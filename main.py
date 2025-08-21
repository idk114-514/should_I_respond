import json
import re
import asyncio
import random
from pathlib import Path
from typing import Dict, Any

import aiofiles
from astrbot.api import logger, AstrBotConfig
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.provider import ProviderRequest, LLMResponse
from astrbot.api.star import Context, Star, register

@register(
    "persona_interest_controller",
    "Gemini",
    "A plugin to dynamically adjust persona based on conversation interest for aiocqhttp.",
    "1.1.2", # 版本号更新
    "https://github.com/your-repo/astrophot_plugin_persona_interest_controller"
)
class PersonaInterestPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.history_cache: Dict[str, list] = {}
        self.history_file = Path("data") / "persona_interest_history.json"
        self.history_lock = asyncio.Lock()
        
        asyncio.create_task(self._load_history())
        logger.info("Persona Interest Controller plugin loaded with self-managed history.")

    async def _load_history(self):
        async with self.history_lock:
            if self.history_file.exists():
                try:
                    async with aiofiles.open(self.history_file, "r", encoding="utf-8") as f:
                        self.history_cache = json.loads(await f.read())
                        logger.info(f"Successfully loaded history from {self.history_file}")
                except Exception as e:
                    logger.error(f"Failed to load history file: {e}")

    async def _save_history(self):
        async with self.history_lock:
            try:
                max_len = self.config.get("history_max_length", 20)
                for session_id in self.history_cache:
                    self.history_cache[session_id] = self.history_cache[session_id][-max_len:]
                
                async with aiofiles.open(self.history_file, "w", encoding="utf-8") as f:
                    await f.write(json.dumps(self.history_cache, indent=2, ensure_ascii=False))
            except Exception as e:
                logger.error(f"Failed to save history file: {e}")

    async def _get_current_persona_prompt(self, event: AstrMessageEvent) -> str:
        try:
            uid = event.unified_msg_origin
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(uid)
            if not curr_cid: return ""
            conversation = await self.context.conversation_manager.get_conversation(uid, curr_cid)
            if not conversation: return ""
            persona_id = conversation.persona_id
            target_persona_name = ""
            if persona_id and persona_id != "[%None]":
                target_persona_name = persona_id
            elif not persona_id:
                target_persona_name = self.context.provider_manager.selected_default_persona.get("name")
            else: return ""
            if not target_persona_name: return ""
            all_personas: list[dict] = self.context.provider_manager.personas
            for persona_dict in all_personas:
                if persona_dict.get("name") == target_persona_name:
                    return persona_dict.get("prompt", "")
            return ""
        except Exception: return ""

    @filter.on_llm_request(priority=10)
    async def interest_analyzer(self, event: AstrMessageEvent, req: ProviderRequest):
        subject_id = event.get_group_id() or event.get_sender_id()
        if subject_id not in self.config.get("whitelist", []):
            return
        if event.get_platform_name() != "aiocqhttp": return

        logger.debug(f"[PIC] Analyzing interest for whitelisted session: {event.unified_msg_origin}.")
        
        analysis_provider_id = self.config.get("analysis_provider_id")
        if not analysis_provider_id: return
        analysis_provider = self.context.get_provider_by_id(analysis_provider_id)
        if not analysis_provider: return

        persona_description = await self._get_current_persona_prompt(event) or req.system_prompt
        
        session_id = event.unified_msg_origin
        current_message = req.prompt or "User sent an empty or non-text message."
        
        user_history_entry = {
            "role": "user",
            "sender_id": event.get_sender_id(),
            "sender_name": event.get_sender_name(),
            "content": f"[Direct Mention] {current_message}" if event.is_at_or_wake_command else current_message
        }
        self.history_cache.setdefault(session_id, []).append(user_history_entry)
        
        history_for_analysis = self.history_cache[session_id][:-1]
        
        def format_history_entry(msg: dict) -> str:
            if msg['role'] == 'user':
                return f"user ({msg.get('sender_name', 'unknown')}/{msg.get('sender_id', '0')}): {msg.get('content')}"
            return f"assistant: {msg.get('content')}"

        formatted_history = "\n".join(map(format_history_entry, history_for_analysis)) or "No previous conversation history."
        current_message_for_analysis = format_history_entry(user_history_entry)
        
        awakening_context_str = ""
        if event.is_at_or_wake_command:
            awakening_context_str = "IMPORTANT CONTEXT: The user has directly awakened you..."
        
        json_str = ""
        try:
            analysis_prompt_template = self.config.get("analysis_system_prompt")
            analysis_prompt = analysis_prompt_template.replace("{awakening_context}", awakening_context_str)
            analysis_prompt = analysis_prompt.replace("{persona}", persona_description)
            analysis_prompt = analysis_prompt.replace("{history}", formatted_history)
            analysis_prompt = analysis_prompt.replace("{current_message}", current_message_for_analysis)
            
            analysis_response = await analysis_provider.text_chat(prompt=analysis_prompt)
            response_text = analysis_response.completion_text
            
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if not json_match: return

            json_str = json_match.group(0)
            analysis_result = json.loads(json_str)

            if not analysis_result.get("should_reply", True):
                logger.info(f"[PIC] Analysis decided not to reply. Reason: {analysis_result.get('reason')}. Stopping event.")
                event.stop_event()
                await self._save_history()
                return

            # --- 关键修改: 随机回复检定 ---
            reply_chance = self.config.get("random_reply_chance", 1.0)
            if random.random() > reply_chance:
                logger.info(f"[PIC] Analysis decided to reply, but failed the random chance roll ({reply_chance * 100}%). Stopping event.")
                event.stop_event()
                await self._save_history()
                return

            interest = analysis_result.get("interest", "normal")
            feeling = analysis_result.get("feeling", "neutral")

            # --- 关键修改: 将情感状态存入事件，以便后续记录 ---
            event.set_extra("pic_emotion_data", {"interest": interest, "feeling": feeling})

            injection_wrapper = f"[[System Note: Your current state is - Interest: '{interest}', Feeling: '{feeling}'. You MUST respond according to this state.]]\n\nUser's message is: \"{req.prompt}\""
            injection_wrapper = f"User's message is: \"{req.prompt}\"\n\n[[System Note: Your current state is - Interest: '{interest}', Feeling: '{feeling}'. You MUST respond according to this state.]]"

            req.prompt = injection_wrapper
            logger.info(f"[PIC] Injected emotion into user prompt.")

        except Exception as e:
            logger.error(f"[PIC] An error occurred during interest analysis: {e}", exc_info=True)

    @filter.on_llm_response(priority=10)
    async def save_llm_reply_to_history(self, event: AstrMessageEvent, resp: LLMResponse):
        subject_id = event.get_group_id() or event.get_sender_id()
        if subject_id not in self.config.get("whitelist", []):
            return

        session_id = event.unified_msg_origin
        bot_reply_str = resp.completion_text

        if bot_reply_str:
            bot_history_entry = {
                "role": "assistant",
                "content": bot_reply_str
            }
            # --- 关键修改: 检查是否需要记录情感状态 ---
            if self.config.get("record_emotion_in_history", False):
                emotion_data = event.get_extra("pic_emotion_data")
                if emotion_data:
                    bot_history_entry["state"] = emotion_data
            
            self.history_cache.setdefault(session_id, []).append(bot_history_entry)
            await self._save_history()
            logger.debug(f"[PIC] Saved bot LLM reply to history for session {session_id}")

    @filter.command_group("sir")
    async def history_ctrl(self, event: AstrMessageEvent):
        """管理本插件的聊天记录缓存"""
        pass

    @history_ctrl.command("view")
    async def view_history(self, event: AstrMessageEvent):
        session_id = event.unified_msg_origin
        history = self.history_cache.get(session_id, [])
        if not history:
            yield event.plain_result("当前会话没有聊天记录。")
            return
        
        # --- 关键修改: 在视图中显示情感状态 ---
        def format_view_entry(msg: dict) -> str:
            if msg['role'] == 'user':
                return f"[{msg.get('role')}] ({msg.get('sender_name', 'unknown')}/{msg.get('sender_id', '0')}): {msg.get('content')}"
            
            state_info = ""
            if msg.get("state"):
                state = msg["state"]
                state_info = f" (State: {state.get('interest', 'N/A')}, {state.get('feeling', 'N/A')})"
            
            return f"[{msg.get('role')}]{state_info}: {msg.get('content')}"

        formatted_history = "--- Chat History ---\n" + "\n".join(map(format_view_entry, history))
        yield event.plain_result(formatted_history)

    @history_ctrl.command("clear")
    async def clear_history(self, event: AstrMessageEvent):
        session_id = event.unified_msg_origin
        if session_id in self.history_cache:
            del self.history_cache[session_id]
            await self._save_history()
            yield event.plain_result("当前会话的聊天记录已清空。")
        else:
            yield event.plain_result("当前会话没有聊天记录可清空。")

    async def terminate(self):
        await self._save_history()
        logger.info("Persona Interest Controller plugin unloaded.")