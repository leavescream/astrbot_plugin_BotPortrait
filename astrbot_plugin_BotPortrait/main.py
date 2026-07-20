import datetime
import random
import shutil
import datetime
from pathlib import Path
from typing import Optional, Any

from astrbot.api import llm_tool, logger, sp
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.star.star_tools import StarTools
from astrbot.api.provider import LLMResponse
from astrbot.core.provider.entities import ProviderRequest
from .core.comfyui import ComfyUIClient
from .core.data import HairstyleDataManager, OutfitDataManager
from .core.generator import HairstyleGenerator, OutfitGenerator
from .core.outfit import OutfitPicker
from astrbot.core.message.message_event_result import ResultContentType


_DEFAULT_PERSONALITY = {
    "personality_key": "default",
    "core_prompt": "",
    "core_negative": "worst quality, low quality, lowres, watermark, blurry",
    "enable_daily_outfit": True,
    "enable_daily_hairstyle": True,
    "auto_save_enabled": False,
    "auto_save_path": "",
    "engine": {
        "workflow_json_path": "workflows/T2I_lora.json",
        "model": "",
        "lora": "",
        "lora_strength": 0.79,
        "lora_clip_strength": 1.0,
        "min_width": 544,
        "max_width": 808,
        "min_height": 544,
        "max_height": 808,
        "steps": 20,
        "cfg": 8.0,
        "sampler_name": "dpmpp_2m",
        "scheduler": "karras",
    },
}


class BotPortraitPlugin(Star):
    """
    BotPortrait - AI 自画像生成插件。

    两种触发方式：
    1. LLM 工具「take_selfie」— 艾可在对话中主动调用，同步生图并发送
    2. 命令「自拍」— 主人手动触发
    """

    # ============================================================
    # [初始化] 插件属性与配置加载
    # ============================================================
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._enabled: bool = True
        self._generating: bool = False
        self._suppress_auto_portrait: bool = False  # take_selfie 工具触发时抑制自动生图

        self.outfit_data: Optional[OutfitDataManager] = None
        self.outfit_picker: Optional[OutfitPicker] = None
        self.outfit_generator: Optional[OutfitGenerator] = None
        self.hairstyle_data: Optional[HairstyleDataManager] = None
        self.hairstyle_generator: Optional[HairstyleGenerator] = None
        self.comfyui: Optional[ComfyUIClient] = None
        self._last_generated_prompt: str = ""
        self._pending_action_tags = None
        self._pending_pic_prompt = None
        self._pending_reply_text = None

    # ============================================================
    # [人格配置] 获取/匹配不同人格的参数
    # ============================================================
    def _get_personality_config(self, key: str, default: Any = None) -> Any:
        configs = self.config.get("personality_configs", [])
        if configs:
            return configs[0].get(key, _DEFAULT_PERSONALITY.get(key, default))
        return _DEFAULT_PERSONALITY.get(key, default)

    def _get_engine_config(self, persona_id: str = None) -> dict:
        return self._get_personality_config("engine", _DEFAULT_PERSONALITY["engine"], persona_id=persona_id)

    async def _resolve_persona_id(self, event: AstrMessageEvent) -> Optional[str]:
        try:
            umo = event.unified_msg_origin
            session_cfg = await sp.get_async(
                scope="umo", scope_id=umo,
                key="session_service_config", default={}
            )
            session_persona_id = session_cfg.get("persona_id") if isinstance(session_cfg, dict) else None
            if session_persona_id:
                logger.debug(f"[人格匹配] [{umo}] session_service_config: {session_persona_id}")
                return session_persona_id

            session_id = await self.context.conversation_manager.get_curr_conversation_id(umo)
            if session_id:
                conversation = await self.context.conversation_manager.get_conversation(umo, session_id)
                persona_id = conversation.persona_id if conversation else None
                if persona_id and persona_id != "[%None]":
                    logger.debug(f"[人格匹配] [{umo}] 会话人格: {persona_id}")
                    return persona_id

            default_persona = await self.context.persona_manager.get_default_persona_v3(umo=umo)
            if default_persona and isinstance(default_persona, dict):
                persona_id = default_persona.get("name")
                if persona_id:
                    logger.debug(f"[人格匹配] [{umo}] 默认人格: {persona_id}")
                    return persona_id

            logger.debug(f"[人格匹配] [{umo}] 未获取到任何人格 ID")
        except Exception as e:
            logger.debug(f"[人格匹配] 获取人格 ID 异常: {e}")
        return None

    def _match_personality_config(self, persona_id: str) -> Optional[dict]:
        if not persona_id:
            return None
        configs = self.config.get("personality_configs", [])
        if not configs:
            return None
        for entry in configs:
            if not isinstance(entry, dict):
                continue
            if entry.get("personality_key") == persona_id:
                return entry
        return None

    def _get_personality_config(self, key: str, default: Any = None, persona_id: str = None) -> Any:
        if persona_id:
            matched = self._match_personality_config(persona_id)
            if matched:
                return matched.get(key, _DEFAULT_PERSONALITY.get(key, default))
            return _DEFAULT_PERSONALITY.get(key, default)
        configs = self.config.get("personality_configs", [])
        if configs:
            return configs[0].get(key, _DEFAULT_PERSONALITY.get(key, default))
        return _DEFAULT_PERSONALITY.get(key, default)

    # ============================================================
    # [生命周期] 插件启动与停止
    # ============================================================
    async def initialize(self):
        try:
            data_dir = StarTools.get_data_dir() / "astrbot_plugin_BotPortrait"
            data_dir.mkdir(parents=True, exist_ok=True)

            pic_guide = self.config.get("pic_guide", "").strip()
            if pic_guide:
                logger.info(f"  翻译说明书已加载: {len(pic_guide)} 字符")
            else:
                logger.warning("  翻译说明书为空，未注入 pic_guide")
            self._pic_guide_content = pic_guide

            try:
                outfit_file = data_dir / "outfit_data.json"
                self.outfit_data = OutfitDataManager(outfit_file)
                pool = self.config.get("outfit_pool", [])
                self.outfit_picker = OutfitPicker(self.outfit_data, pool)
                self.outfit_generator = OutfitGenerator(self.context, self.outfit_data, pool, pic_guide)
                logger.info("  穿搭系统初始化完成")
            except Exception as e:
                logger.error(f"  穿搭系统初始化失败: {e}")

            try:
                hairstyle_file = data_dir / "hairstyle_data.json"
                self.hairstyle_data = HairstyleDataManager(hairstyle_file)
                h_pool = self.config.get("hairstyle_pool", [])
                self.hairstyle_generator = HairstyleGenerator(self.context, self.hairstyle_data, h_pool, pic_guide)
                logger.info("  发型系统初始化完成")
            except Exception as e:
                logger.error(f"  发型系统初始化失败: {e}")

            try:
                api_url = self.config.get("api_url", "http://127.0.0.1:8188")
                timeout = self.config.get("timeout", 30)
                engine_cfg = self._get_engine_config()
                wf_path = self._resolve_workflow_path(engine_cfg.get("workflow_json_path", "workflows/T2I_lora.json"))
                self.comfyui = ComfyUIClient(api_url, wf_path, timeout=timeout)
                logger.info(f"  ComfyUI 客户端初始化完成: {api_url}")
            except Exception as e:
                logger.error(f"  ComfyUI 客户端初始化失败: {e}")
                self.comfyui = None

            self._enabled = True
            logger.info("BotPortrait 插件初始化完成，开关: True")
        except Exception as e:
            logger.error(f"BotPortrait 初始化顶层异常: {e}")
            self._enabled = True

        self.context.activate_llm_tool("take_selfie")
        logger.info("  LLM 工具 take_selfie 已注册")

    async def terminate(self):
        logger.info("BotPortrait 插件已停止")

    # ============================================================
    # [路径解析] 工作流 JSON 路径定位
    # ============================================================
    def _resolve_workflow_path(self, wf_path: str) -> str:
        if Path(wf_path).is_absolute():
            return wf_path
        base = Path(__file__).parent
        return str(base / wf_path)

    # ============================================================
    # [日更系统] 每日穿搭 & 发型生成
    # ============================================================
    def _get_refresh_time(self) -> tuple[int, int]:
        """从配置读取每日刷新时间，返回 (hour, minute)，默认 00:00"""
        refresh_str = self.config.get("daily_refresh_time", "00:00")
        try:
            parts = refresh_str.split(":")
            return int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            return 0, 0

    def _get_effective_date(self) -> str:
        """根据刷新时间计算有效日期：刷新时间前沿用前一天，刷新时间后用今天"""
        refresh_hour, refresh_min = self._get_refresh_time()
        now = datetime.datetime.now()
        if now.hour < refresh_hour or (now.hour == refresh_hour and now.minute < refresh_min):
            eff = datetime.date.today() - datetime.timedelta(days=1)
        else:
            eff = datetime.date.today()
        return eff.isoformat()

    async def _ensure_today_outfit(self, umo: Optional[str] = None, personality_key: str = "echo"):
        if not self.outfit_generator or not self.outfit_picker:
            return
        eff_date = self._get_effective_date()
        if self.outfit_data.get(eff_date, personality_key):
            return
        ok = await self.outfit_generator.generate_today(umo, personality_key=personality_key, effective_date=eff_date)
        if ok:
            return
        static = self.outfit_picker.pick_or_reuse(personality_key)
        if static:
            static.personality_key = personality_key
            self.outfit_data.set(static)
            logger.info(f"静态回退今日穿搭：{static.style}")
            return
        logger.warning("今日穿搭为空（LLM 和静态回退均失败）")

    async def _ensure_today_hairstyle(self, umo: Optional[str] = None, personality_key: str = "echo"):
        if not self.hairstyle_generator:
            return
        eff_date = self._get_effective_date()
        if self.hairstyle_data.get(eff_date, personality_key):
            return
        await self.hairstyle_generator.generate_today(umo, personality_key=personality_key, effective_date=eff_date)

    # ============================================================
    # [生图核心] 构建 SD prompt → ComfyUI 渲染
    # ============================================================
    # 中文动作描述 → 英文翻译映射（用于 take_selfie 传入的 action_desc）
    _CN_ACTION_MAP = {
        "双手在胸前比V字手势，歪头微笑": "both hands making V-signs in front of chest, tilting head with smile",
        "右手托腮，目光看向侧方，若有所思的表情": "right hand on cheek, gazing to the side with thoughtful expression",
        "双手叉腰，微微抬起下巴，自信的表情": "hands on hips, chin slightly lifted, confident expression",
        "双手背在身后，身体微微前倾，好奇地歪头": "hands behind back, body leaning forward slightly, curious head tilt",
        "右手比一个OK手势放在眼前，单眼wink": "making an OK gesture over one eye with right hand, single eye wink",
        "双手轻轻捧脸，露出灿烂的笑容": "gently holding face with both hands, beaming bright smile",
        "右手食指轻点下巴，歪头做思考状": "right index finger lightly touching chin, tilting head in thought",
        "双手高举过头，做出欢呼的姿势，开心地笑着": "both arms raised high in cheer gesture, laughing happily",
        "双臂交叉抱在胸前，面无表情地直视镜头": "arms crossed over chest, expressionless stare into camera",
        "右手在脸侧比出小小的爱心手势，腼腆地微笑": "making a small heart gesture beside face with right hand, shy smile",
        "双手合十放在胸前，微微低头，闭眼祈祷般的表情": "hands clasped together at chest, head slightly lowered, eyes closed like praying",
        "左手叉腰，右手伸出食指指向侧方，挑眉坏笑": "left hand on hip, right index finger pointing sideways, raised eyebrow smirk",
        "双手比V": "both hands making V-sign",
        "歪头": "tilting head",
        "微笑": "smiling",
        "托腮": "hand on cheek",
        "叉腰": "hands on hips",
        "wink": "winking",
        "闭眼": "eyes closed",
        "双手合十": "hands clasped together",
        "比心": "making heart gesture",
        "挥手": "waving hand",
        "叉手": "arms crossed",
        "歪头微笑": "tilting head with smile",
        "双手比V歪头微笑": "both hands making V-sign, tilting head with smile",
        "双手撑脸歪头好奇打量": "resting face on hands, tilting head curiously observing",
        "开心大笑": "laughing happily",
        "自信表情": "confident expression",
        "思考状": "thoughtful expression",
        "双手高举": "both arms raised high",
        "OK手势": "OK gesture",
        "爱心手势": "heart gesture",
        "食指指": "index finger pointing",
        "挑眉": "raised eyebrow",
        "直视镜头": "staring into camera",
        "单眼wink": "single eye wink",
        "双手抱胸": "arms crossed over chest",
    }

    @staticmethod
    def _translate_action(text: str) -> str:
        """将中文动作描述翻译为英文。先查映射表，再尝试通用清洗。"""
        if not text:
            return ""
        # 完整匹配优先
        if text in BotPortraitPlugin._CN_ACTION_MAP:
            return BotPortraitPlugin._CN_ACTION_MAP[text]
        # 尝试逐词/短语匹配
        for cn, en in sorted(BotPortraitPlugin._CN_ACTION_MAP.items(), key=lambda x: -len(x[0])):
            if cn in text:
                text = text.replace(cn, en)
        # 去除残留中文（通用清洗）
        import re
        text = re.sub(r'[\u4e00-\u9fff]+', '', text).strip()
        # 清洗多余空格和逗号
        text = re.sub(r'\s*,\s*', ', ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip(", ")

    async def _generate_portrait(self, action_hint: str = "", umo: str = "", personality_key: str = "echo") -> Optional[str]:
        engine_cfg = self._get_engine_config(personality_key)
        # 中文 → 英文翻译（ComfyUI 需要英文 prompt）
        action_hint = self._translate_action(action_hint)
        outfit_tags = ""
        hairstyle_tags = ""
        today = self._get_effective_date()
        enable_outfit = self._get_personality_config("enable_daily_outfit", True, persona_id=personality_key)
        enable_hairstyle = self._get_personality_config("enable_daily_hairstyle", True, persona_id=personality_key)
        outfit = self.outfit_data.get(today, personality_key) if (self.outfit_data and enable_outfit) else None
        if outfit:
            outfit_tags = outfit.tags
        hs = self.hairstyle_data.get(today, personality_key) if (self.hairstyle_data and enable_hairstyle) else None
        if hs:
            hairstyle_tags = hs.tags
        core = self._get_personality_config("core_prompt", "", persona_id=personality_key)
        extra = action_hint
        if outfit_tags:
            extra += ", " + outfit_tags
        if hairstyle_tags:
            extra += ", " + hairstyle_tags
        prompt = core
        if extra:
            prompt += ", " + extra
        negative = self._get_personality_config("core_negative", "", persona_id=personality_key)
        params = {
            "prompt": prompt,
            "negative_prompt": negative,
            "steps": engine_cfg.get("steps", 20),
            "cfg": engine_cfg.get("cfg", 8.0),
            "sampler_name": engine_cfg.get("sampler_name", "dpmpp_2m"),
            "scheduler": engine_cfg.get("scheduler", "karras"),
            "width": random.randint(engine_cfg.get("min_width", 544), engine_cfg.get("max_width", 808)),
            "height": random.randint(engine_cfg.get("min_height", 544), engine_cfg.get("max_height", 808)),
            "model": engine_cfg.get("model", ""),
            "lora": engine_cfg.get("lora", ""),
            "lora_strength": engine_cfg.get("lora_strength", 0.79),
            "lora_clip_strength": engine_cfg.get("lora_clip_strength", 1.0),
        }
        logger.info(f"[BotPortrait] 动作翻译结果: '{action_hint}'")
        logger.info(f"[BotPortrait] extra 完整长度={len(extra)}: {extra[:200]}...")
        logger.info(f"ComfyUI prompt: {prompt[:120]}...")
        if not self.comfyui:
            logger.error("ComfyUI 客户端未初始化")
            return None
        model_name = params.get("model", "")
        lora_name = params.get("lora", "")
        err = await self.comfyui.validate_model_and_lora(model_name, lora_name)
        if err:
            logger.error(f"模型/LoRA 验证失败: {err}")
            return None
        self._last_generated_prompt = params["prompt"]
        return await self.comfyui.generate(
            prompt=params["prompt"],
            negative=params.get("negative_prompt", ""),
            filename_prefix=personality_key,
            steps=params.get("steps", 20),
            cfg=params.get("cfg", 8.0),
            sampler_name=params.get("sampler_name", "dpmpp_2m"),
            scheduler=params.get("scheduler", "karras"),
            width=params.get("width", 544),
            height=params.get("height", 808),
            model=params.get("model", ""),
            lora=params.get("lora", ""),
            lora_strength=params.get("lora_strength", 0.79),
            lora_clip_strength=params.get("lora_clip_strength", 1.0),
        )

    # ============================================================
    # [LLM 工具] 艾可主动自拍（take_selfie）
    # ============================================================
    @llm_tool(name="take_selfie")
    async def take_selfie(self, event: AstrMessageEvent, action_desc: str):
        """艾可主动自拍一张自画像并发送到对话中。

        Args:
            action_desc(str): 动作/表情描述，如'双手比V歪头微笑'。"""
        if not self._enabled:
            return "自画像功能已关闭，请发送「开启自画像」打开"
        persona_id = await self._resolve_persona_id(event)
        matched_config = self._match_personality_config(persona_id) if persona_id else None
        if not matched_config:
            logger.warning(f"[take_selfie] 未匹配到人格配置: persona_id={persona_id}")
            return "无对应人格配置表"
        personality_key = matched_config.get("personality_key", "default")
        await event.send(event.plain_result("📸 准备中..."))
        umo = event.unified_msg_origin
        await self._ensure_today_outfit(umo, personality_key)
        await self._ensure_today_hairstyle(umo, personality_key)
        await event.send(event.plain_result("🎨 ComfyUI 渲染中..."))
        img_path = await self._generate_portrait(action_desc, umo, personality_key)
        if not img_path:
            return "自画像生成失败（请检查 ComfyUI 是否运行，以及模型/LoRA 配置是否正确）"
        # 标记抑制自动生图：take_selfie 已成功出图，不让 auto 再跟一张
        self._suppress_auto_portrait = True
        if matched_config.get("auto_save_enabled", False):
            save_dir = matched_config.get("auto_save_path", "").strip()
            if save_dir:
                try:
                    save_path = Path(save_dir)
                    save_path.mkdir(parents=True, exist_ok=True)
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    ext = Path(img_path).suffix or ".png"
                    dest = save_path / f"portrait_{ts}{ext}"
                    shutil.copy2(img_path, str(dest))
                    logger.info(f"图片已自动储存: {dest}")
                except Exception as e:
                    logger.error(f"自动储存图片失败: {e}")
        try:
            await event.send(event.image_result(str(img_path)))
            await event.send(event.plain_result("✅"))
            logger.info(f"take_selfie: 已发送 ({action_desc})")
            return "图片已发送"
        except Exception as e:
            logger.error(f"take_selfie 发图失败: {e}")
            return f"自画像已生成但发送失败：{e}"
        finally:
            try:
                p = Path(img_path)
                if p.exists():
                    p.unlink()
                    logger.debug(f"临时图片已清理: {p.name}")
            except Exception as e:
                logger.error(f"临时图片清理失败: {e}")
            if self.config.get("auto_clean_comfyui_output", False):
                output_dir = self.config.get("comfyui_output_path", "").strip()
                if output_dir:
                    try:
                        out_path = Path(output_dir)
                        if out_path.exists() and out_path.is_dir():
                            deleted = 0
                            for f in out_path.iterdir():
                                if f.is_file() and f.name.startswith(f"{personality_key}_"):
                                    f.unlink()
                                    deleted += 1
                            if deleted:
                                logger.info(f"ComfyUI output 已清理 {deleted} 张旧图: {output_dir}")
                        else:
                            logger.warning(f"ComfyUI output 目录不存在: {output_dir}")
                    except Exception as e:
                        logger.error(f"ComfyUI output 清理失败: {e}")

    # ============================================================
    # [LLM 钩子] 注入状态 / 拦截回复 / 自动生图
    # ============================================================
    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """在每次 LLM 请求前注入今日穿搭和发型信息（life_scheduler 风格）"""
        umo = event.unified_msg_origin
        persona_id = await self._resolve_persona_id(event)
        personality_key = persona_id or "echo"
        enable_outfit = self._get_personality_config("enable_daily_outfit", True, persona_id=persona_id)
        enable_hairstyle = self._get_personality_config("enable_daily_hairstyle", True, persona_id=persona_id)
        if enable_outfit:
            await self._ensure_today_outfit(umo, personality_key)
        if enable_hairstyle:
            await self._ensure_today_hairstyle(umo, personality_key)
        today = self._get_effective_date()
        outfit = self.outfit_data.get(today, personality_key) if (self.outfit_data and enable_outfit) else None
        hs = self.hairstyle_data.get(today, personality_key) if (self.hairstyle_data and enable_hairstyle) else None

        now = datetime.datetime.now()
        weekday_map = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        time_tag = f"{now.strftime('%Y-%m-%d')} {now.strftime('%H:%M')} ({weekday_map[now.weekday()]})"
        parts = ["<character_state>"]
        parts.append(f"时间: {time_tag}")
        if outfit:
            parts.append(f"我穿着{outfit.chinese}")
        if hs:
            parts.append(f"我发型是{hs.chinese}")
        parts.append("</character_state>")
        parts.append("<state_following_rules>")
        parts.append("- 当用户问到正在做什么、今天安排、所在场景、穿着或生活状态时，必须以 <character_state> 为准。")
        parts.append("- 不得编造与当前状态冲突的上课、上班、外出、睡觉等状态。")
        parts.append("- 与用户问题无关时无需主动提及这些状态。")
        parts.append("</state_following_rules>")
        state_injection = "\n".join(parts)

        # ── 注入（仅 state，pic_guide 已移交给子 LLM 分镜导演）
        to_inject = "\n\n" + state_injection
        
        injected = False
        if hasattr(req, 'system_prompt') and req.system_prompt:
            if to_inject not in req.system_prompt:
                req.system_prompt += to_inject
                injected = True
        elif hasattr(req, 'prompt_messages') and req.prompt_messages:
            # 优先注入 system 消息
            for msg in req.prompt_messages:
                if msg.get("role") == "system":
                    curr = msg.get("content", "")
                    if to_inject not in curr:
                        msg["content"] = curr + to_inject
                        injected = True
                    break
            # 无 system 消息时，注入到第一条消息开头
            if not injected and req.prompt_messages:
                first = req.prompt_messages[0]
                curr = first.get("content", "")
                first["content"] = to_inject.strip() + "\n\n" + curr
                injected = True

        logger.info(f"[BotPortrait] LLM 注入（子LLM分镜模式）| state: {'已注入' if injected else '失败'}")

    def _extract_action_tags(self, text: str) -> list[str]:
        import re
        pattern = re.compile(r'&&([^&]+?)&&')
        matches = pattern.findall(text)
        seen = set()
        tags = []
        for tag in matches:
            tag = tag.strip()
            if tag and tag not in seen:
                seen.add(tag)
                tags.append(tag)
        return tags

    @filter.on_llm_response(priority=99999)
    async def on_llm_response_auto(self, event: AstrMessageEvent, response: LLMResponse):
        if not self._enabled:
            return
        if not response or not response.completion_text:
            return
        # 子LLM分镜模式：保存完整回复文本供后续子LLM分析，不再提取 <pic> 标签
        self._pending_reply_text = response.completion_text

    @filter.on_decorating_result(priority=99999)
    async def on_decorating_result_auto(self, event: AstrMessageEvent):
        """流式传输兼容：在消息装饰阶段触发生图，支持 STREAMING_FINISH 信号"""
        # 如果 take_selfie 刚生成过图片，跳过自动触发避免重复
        if self._suppress_auto_portrait:
            self._suppress_auto_portrait = False
            logger.info("[BotPortrait] take_selfie 已生图，跳过自动触发")
            return

        # 流式传输兼容：等待 STREAMING_FINISH（流式结束信号）才触发
        result = event.get_result()
        if result and result.result_content_type == ResultContentType.STREAMING_RESULT:
            return  # 流式传输中，等待完成

        # 检测消息是否已被指令处理器匹配 — 如果是，跳过自动生图
        # AstrBot pipeline 在匹配到 @filter.command() 指令时会设置此 extra
        if event.get_extra("handlers_parsed_params", {}):
            logger.debug("[BotPortrait] 消息已被指令处理器匹配，跳过自动生图")
            return

        reply_text = self._pending_reply_text
        if not reply_text:
            return
        try:
            # 子LLM分镜：分析回复内容产出 SD tags
            sd_tags = await self._sub_llm_pic_director(reply_text)
            if not sd_tags:
                logger.debug("[BotPortrait] 子LLM未产出SD tags，跳过生图")
                return

            persona_id = await self._resolve_persona_id(event)
            matched_config = self._match_personality_config(persona_id) if persona_id else None
            personality_key = matched_config.get("personality_key", "echo") if matched_config else "echo"
            prob = self._get_personality_config("trigger_probability", 1.0, persona_id=persona_id)
            import random
            if random.random() > prob:
                logger.debug(f"[BotPortrait] 概率未命中: {random.random():.2f} > {prob}")
                return
            umo = event.unified_msg_origin
            enable_outfit = self._get_personality_config("enable_daily_outfit", True, persona_id=persona_id)
            enable_hairstyle = self._get_personality_config("enable_daily_hairstyle", True, persona_id=persona_id)
            if enable_outfit:
                await self._ensure_today_outfit(umo, personality_key)
            if enable_hairstyle:
                await self._ensure_today_hairstyle(umo, personality_key)
            img_path = await self._generate_portrait(sd_tags, umo, personality_key)
            if not img_path:
                logger.error(f"[BotPortrait] 自动触发生图失败: {sd_tags[:80]}...")
                return
            try:
                await event.send(event.image_result(str(img_path)))
                logger.info(f"[BotPortrait] 自动触发生图已发送")
            except Exception as e:
                logger.error(f"[BotPortrait] 自动发送图片失败: {e}")
            finally:
                from pathlib import Path
                try:
                    p = Path(img_path)
                    if p.exists():
                        p.unlink()
                        logger.debug(f"[BotPortrait] 临时图片已清理: {p.name}")
                except Exception:
                    pass
        finally:
            self._pending_pic_prompt = None
            self._pending_action_tags = None
            self._pending_reply_text = None

    async def _sub_llm_pic_director(self, reply_text: str) -> Optional[str]:
        """子LLM分镜导演：用 pic_guide 规则分析回复，产出 SD tags"""
        if not self._pic_guide_content:
            logger.warning("[BotPortrait] pic_guide 未配置，子LLM分镜跳过")
            return None

        provider = self.context.get_using_provider()
        if not provider:
            logger.warning("[BotPortrait] 无法获取 LLM provider，子LLM分镜跳过")
            return None

        sid = f"botportrait_subllm_{datetime.date.today().isoformat()}"
        nl = chr(10)
        try:
            _sub_prompt = (
                "你是一名自画像分镜导演。严格遵循以下分镜规则，分析用户的回复内容，"
                f"输出一张自画像的 SD tags。{nl * 2}"
                f"【分镜规则】{nl}"
                f"{self._pic_guide_content}{nl * 2}"
                "只输出 SD tags，不要多余的解释。"
            )
            resp = await provider.text_chat(
                prompt=f"根据以下回复内容，输出 SD tags：{nl * 2}{reply_text}",
                system_prompt=_sub_prompt,
                session_id=sid,
            )
            text = self._sub_extract_text(resp)
            if text:
                logger.info(f"[BotPortrait] 子LLM分镜产出: {text[:100]}...")
                return text
            logger.warning("[BotPortrait] 子LLM返回为空")
            return None
        finally:
            await self._sub_cleanup_session(sid)

    @staticmethod
    def _sub_extract_text(resp) -> str:
        if resp is None:
            return ""
        for key in ("completion_text", "completion", "text", "content"):
            val = getattr(resp, key, None)
            if isinstance(val, str) and val.strip():
                return val.strip()
        if isinstance(resp, str) and resp.strip():
            return resp.strip()
        return ""

    async def _sub_cleanup_session(self, sid: str):
        try:
            cid = await self.context.conversation_manager.get_curr_conversation_id(sid)
            if cid:
                await self.context.conversation_manager.delete_conversation(sid, cid)
        except Exception:
            pass

    # ============================================================
    # [命令] 手动触发（自拍 / 穿搭 / 发型 / 状态）
    @filter.command("开启自画像", alias={"开自拍", "开启自拍", "start portrait"})
    async def cmd_enable(self, event: AstrMessageEvent):
        self._enabled = True
        yield event.plain_result("✅ 自画像已开启")

    @filter.command("关闭自画像", alias={"关自拍", "关闭自拍", "stop portrait"})
    async def cmd_disable(self, event: AstrMessageEvent):
        self._enabled = False
        yield event.plain_result("⏸️ 自画像已关闭")

    # ------------------------------------------------------------
    # [穿搭命令]
    # ------------------------------------------------------------
    @filter.command("今日穿搭", alias={"outfit", "穿搭"})
    async def cmd_outfit(self, event: AstrMessageEvent):
        persona_id = await self._resolve_persona_id(event)
        personality_key = persona_id or "echo"
        if not self._get_personality_config("enable_daily_outfit", True, persona_id=persona_id):
            yield event.plain_result("每日穿搭已关闭，请在配置页面开启")
            return
        if not self.outfit_data:
            yield event.plain_result("穿搭系统未初始化")
            return
        today = self._get_effective_date()
        outfit = self.outfit_data.get(today, personality_key)
        if outfit:
            yield event.plain_result(f"👗 今日穿搭：{outfit.style}\n{outfit.chinese}")
        else:
            yield event.plain_result("今日穿搭未生成，尝试手动生成中…")
            await self._ensure_today_outfit(event.unified_msg_origin, personality_key)
            outfit = self.outfit_data.get(today, personality_key)
            if outfit:
                yield event.plain_result(f"👗 今日穿搭：{outfit.style}\n{outfit.chinese}")
            else:
                yield event.plain_result("无穿搭池配置")

    @filter.command("自拍状态", alias={"portrait status", "肖像状态"})
    async def cmd_status(self, event: AstrMessageEvent):
        lines = [
            f"🔘 状态: {'开' if self._enabled else '关'}",
            f"📸 生成中: {'是' if self._generating else '否'}",
        ]
        today = self._get_effective_date()
        persona_id = await self._resolve_persona_id(event)
        personality_key = persona_id or "echo"
        if self.outfit_data:
            if not self._get_personality_config("enable_daily_outfit", True, persona_id=persona_id):
                lines.append("👗 今日穿搭: 关")
            else:
                o = self.outfit_data.get(today, personality_key)
                lines.append(f"👗 今日穿搭: {o.style if o else '未生成'}")
        if self.hairstyle_data:
            if not self._get_personality_config("enable_daily_hairstyle", True, persona_id=persona_id):
                lines.append("💇 今日发型: 关")
            else:
                h = self.hairstyle_data.get(today, personality_key)
                lines.append(f"💇 今日发型: {h.style if h else '未生成'}")
        if self._last_generated_prompt:
            lines.append(f"📝 生图提示词: {self._last_generated_prompt}")
        yield event.plain_result("\n".join(lines))

    @filter.command("重选穿搭", alias={"reroll outfit"})
    async def cmd_reroll(self, event: AstrMessageEvent):
        persona_id = await self._resolve_persona_id(event)
        personality_key = persona_id or "echo"
        if not self._get_personality_config("enable_daily_outfit", True, persona_id=persona_id):
            yield event.plain_result("每日穿搭已关闭，请在配置页面开启")
            return
        if not self.outfit_generator or not self.outfit_picker:
            yield event.plain_result("穿搭系统未初始化")
            return
        today = self._get_effective_date()
        self.outfit_data.remove(today, personality_key) if hasattr(self.outfit_data, "remove") else None
        o = await self.outfit_generator.generate_today(event.unified_msg_origin, personality_key=personality_key, effective_date=today)
        if o:
            o = self.outfit_data.get(today, personality_key)
            yield event.plain_result(f"🔄 已重选：{o.style}\n{o.chinese}")
            return
        static = self.outfit_picker.pick_or_reuse(personality_key)
        if static:
            yield event.plain_result(f"🔄 静态回退：{static.style}\n{static.chinese}")
            return
        yield event.plain_result("穿搭池为空")

    @filter.command("指定穿搭", alias={"set outfit", "穿搭换", "换穿搭"})
    async def cmd_set_outfit(self, event: AstrMessageEvent, style_name: str | None = None):
        persona_id = await self._resolve_persona_id(event)
        personality_key = persona_id or "echo"
        if not self._get_personality_config("enable_daily_outfit", True, persona_id=persona_id):
            yield event.plain_result("每日穿搭已关闭，请在配置页面开启")
            return
        if not style_name or not style_name.strip():
            yield event.plain_result("请指定一个风格名称，例如：指定穿搭 酷飒中性风")
            return
        style_name = style_name.strip()
        if not self.outfit_picker or not self.outfit_generator:
            yield event.plain_result("穿搭系统未初始化")
            return
        today = self._get_effective_date()
        current_outfit = self.outfit_data.get(today, personality_key)
        self.outfit_data.remove(today, personality_key) if hasattr(self.outfit_data, "remove") else None
        o = await self.outfit_generator.generate_today(event.unified_msg_origin, style_hint=style_name, personality_key=personality_key, effective_date=today)
        if o:
            o = self.outfit_data.get(today, personality_key)
            yield event.plain_result(f"🎯 已指定：{o.style}\n{o.chinese}")
            return
        if current_outfit:
            self.outfit_data.set(current_outfit)
        yield event.plain_result(f"不存在「{style_name}」风格")

    # ------------------------------------------------------------
    # [发型命令]
    # ------------------------------------------------------------
    @filter.command("今日发型", alias={"hairstyle", "发型"})
    async def cmd_hairstyle(self, event: AstrMessageEvent):
        persona_id = await self._resolve_persona_id(event)
        personality_key = persona_id or "echo"
        if not self._get_personality_config("enable_daily_hairstyle", True, persona_id=persona_id):
            yield event.plain_result("每日发型已关闭，请在配置页面开启")
            return
        if not self.hairstyle_data:
            yield event.plain_result("发型系统未初始化")
            return
        today = self._get_effective_date()
        hs = self.hairstyle_data.get(today, personality_key)
        if hs:
            yield event.plain_result(f"💇 今日发型：{hs.style}\n{hs.chinese}")
        else:
            yield event.plain_result("今日发型未生成，尝试生成中…")
            await self._ensure_today_hairstyle(event.unified_msg_origin, personality_key)
            hs = self.hairstyle_data.get(today, personality_key)
            if hs:
                yield event.plain_result(f"💇 今日发型：{hs.style}\n{hs.chinese}")
            else:
                yield event.plain_result("无发型池配置")

    @filter.command("重选发型", alias={"reroll hairstyle"})
    async def cmd_reroll_hairstyle(self, event: AstrMessageEvent):
        persona_id = await self._resolve_persona_id(event)
        personality_key = persona_id or "echo"
        if not self._get_personality_config("enable_daily_hairstyle", True, persona_id=persona_id):
            yield event.plain_result("每日发型已关闭，请在配置页面开启")
            return
        if not self.hairstyle_generator:
            yield event.plain_result("发型系统未初始化")
            return
        today = self._get_effective_date()
        if hasattr(self.hairstyle_data, "remove"):
            self.hairstyle_data.remove(today, personality_key)
        hs = await self.hairstyle_generator.generate_today(event.unified_msg_origin, personality_key=personality_key, effective_date=today)
        if hs:
            yield event.plain_result(f"🔄 已重选发型：{hs.style}\n{hs.chinese}")
        else:
            yield event.plain_result("发型池为空")

    @filter.command("指定发型", alias={"set hairstyle"})
    async def cmd_set_hairstyle(self, event: AstrMessageEvent, style_name: str | None = None):
        persona_id = await self._resolve_persona_id(event)
        personality_key = persona_id or "echo"
        if not self._get_personality_config("enable_daily_hairstyle", True, persona_id=persona_id):
            yield event.plain_result("每日发型已关闭，请在配置页面开启")
            return
        if not style_name or not style_name.strip():
            yield event.plain_result("请指定一个发型名称，例如：指定发型 低垂马尾")
            return
        style_name = style_name.strip()
        if not self.hairstyle_generator or not self.hairstyle_data:
            yield event.plain_result("发型系统未初始化")
            return
        today = self._get_effective_date()
        current_hairstyle = self.hairstyle_data.get(today, personality_key)
        if hasattr(self.hairstyle_data, "remove"):
            self.hairstyle_data.remove(today, personality_key)
        hs = await self.hairstyle_generator.generate_today(event.unified_msg_origin, style_hint=style_name, personality_key=personality_key, effective_date=today)
        if hs:
            yield event.plain_result("🎯 已指定发型：" + hs.style + chr(10) + hs.chinese)
        else:
            if current_hairstyle:
                self.hairstyle_data.set(current_hairstyle)
            yield event.plain_result(f"不存在「{style_name}」风格")

    # ------------------------------------------------------------
    # [穿搭/发型开关命令] 仅影响当前人格
    # ------------------------------------------------------------
    @filter.command("开启穿搭", alias={"开启每日穿搭", "outfit on"})
    async def cmd_outfit_on(self, event: AstrMessageEvent):
        persona_id = await self._resolve_persona_id(event)
        if not persona_id:
            yield event.plain_result("未匹配到当前人格，无法操作")
            return
        matched = self._match_personality_config(persona_id)
        if not matched:
            yield event.plain_result("当前人格无独立配置，请在 WebUI 中添加")
            return
        matched["enable_daily_outfit"] = True
        self.config.save_config()
        yield event.plain_result(f"✅ 已为「{persona_id}」开启每日穿搭")

    @filter.command("关闭穿搭", alias={"关闭每日穿搭", "outfit off"})
    async def cmd_outfit_off(self, event: AstrMessageEvent):
        persona_id = await self._resolve_persona_id(event)
        if not persona_id:
            yield event.plain_result("未匹配到当前人格，无法操作")
            return
        matched = self._match_personality_config(persona_id)
        if not matched:
            yield event.plain_result("当前人格无独立配置，请在 WebUI 中添加")
            return
        matched["enable_daily_outfit"] = False
        self.config.save_config()
        yield event.plain_result(f"⏸️ 已为「{persona_id}」关闭每日穿搭")

    @filter.command("开启发型", alias={"开启每日发型", "hairstyle on"})
    async def cmd_hairstyle_on(self, event: AstrMessageEvent):
        persona_id = await self._resolve_persona_id(event)
        if not persona_id:
            yield event.plain_result("未匹配到当前人格，无法操作")
            return
        matched = self._match_personality_config(persona_id)
        if not matched:
            yield event.plain_result("当前人格无独立配置，请在 WebUI 中添加")
            return
        matched["enable_daily_hairstyle"] = True
        self.config.save_config()
        yield event.plain_result(f"✅ 已为「{persona_id}」开启每日发型")

    @filter.command("关闭发型", alias={"关闭每日发型", "hairstyle off"})
    async def cmd_hairstyle_off(self, event: AstrMessageEvent):
        persona_id = await self._resolve_persona_id(event)
        if not persona_id:
            yield event.plain_result("未匹配到当前人格，无法操作")
            return
        matched = self._match_personality_config(persona_id)
        if not matched:
            yield event.plain_result("当前人格无独立配置，请在 WebUI 中添加")
            return
        matched["enable_daily_hairstyle"] = False
        self.config.save_config()
        yield event.plain_result(f"⏸️ 已为「{persona_id}」关闭每日发型")

    # ------------------------------------------------------------
    # [查看库命令]
    # ------------------------------------------------------------
    @filter.command("查看发型库", alias={"发型库", "hair library"})
    async def cmd_hairstyle_library(self, event: AstrMessageEvent):
        if not self.hairstyle_generator:
            yield event.plain_result("发型系统未初始化")
            return
        pool = self.hairstyle_generator.pool
        if not pool:
            yield event.plain_result("发型池为空")
            return
        lines = [f"📋 发型库（共{len(pool)}个）："]
        for i, item in enumerate(pool, 1):
            name = item.split("：")[0].split(":")[0] if "：" in item or ":" in item else item
            lines.append(f"{i}. {name}")
        yield event.plain_result("\n".join(lines))

    @filter.command("查看穿搭库", alias={"穿搭库", "outfit library"})
    async def cmd_outfit_library(self, event: AstrMessageEvent):
        if not self.outfit_picker:
            yield event.plain_result("穿搭系统未初始化")
            return
        pool = self.outfit_picker.pool
        if not pool:
            yield event.plain_result("穿搭池为空")
            return
        lines = [f"📋 穿搭库（共{len(pool)}个）："]
        for i, item in enumerate(pool, 1):
            name = item.split("：")[0].split(":")[0] if "：" in item or ":" in item else item
            lines.append(f"{i}. {name}")
        yield event.plain_result("\n".join(lines))