import asyncio
import datetime
import json
import random
import re
from pathlib import Path
from typing import Optional

from astrbot.api import logger

from .data import DailyHairstyle, DailyOutfit, HairstyleDataManager, OutfitDataManager
from .outfit import _parse_outfit_item


# ─────────────────────────────────────────────
# 基类：life_scheduler 风格的 LLM 生成器
# ─────────────────────────────────────────────


class _BaseGenerator:
    """
    参考 astrbot_plugin_life_scheduler 的 SchedulerGenerator 架构：
    - 一次 LLM 调用 → JSON 输出（含中文 desc + 英文 SD tags）
    - 独立 session_id + 调用后清理
    - JSON 校验 + 重试
    - 支持 pic_guide（翻译说明书）作为 system_prompt 注入
    """

    _EMPTY_RETRIES = 1  # 空完成重试

    def __init__(self, context, data_mgr, pool: list[str], pic_guide: str = ""):
        self.context = context
        self.data_mgr = data_mgr
        self.pool = list(pool or [])
        self.pic_guide = pic_guide
        self._gen_lock = asyncio.Lock()
        self._generating = False

    # ──────── 子类需实现 ────────

    @property
    def _domain(self) -> str:
        """领域名，如 'outfit' / 'hairstyle'"""
        raise NotImplementedError

    def _build_prompt(self, style_name: str, style_desc: str) -> str:
        """构建 LLM prompt"""
        raise NotImplementedError

    def _validate_payload(self, payload: dict, style_name: str) -> tuple[bool, str]:
        """校验 LLM 返回的 JSON 是否合规"""
        raise NotImplementedError

    def _make_data(self, payload: dict, date_str: str, style_name: str, personality_key: str = "") -> object:
        """将解析后的 JSON 转成数据对象"""
        raise NotImplementedError

    # ──────── 公共流程 ────────

    async def generate_today(
        self, umo: Optional[str] = None, style_hint: str = "",
        effective_date: Optional[str] = None, personality_key: str = "echo"
    ) -> Optional[object]:
        """
        参考 life_scheduler.generate_schedule() 的流程：
        1. 今日已有 → 复用
        2. 选风格（避免近期重复）→ 调 LLM（一次调用）→ JSON 解析 → 校验 → 落盘
        3. LLM 失败 → 回退：chinese=风格名, tags=""

        effective_date：有效日期字符串（yyyy-mm-dd）。
        由调用方传入（_get_effective_date()），确保数据存取的 key 一致。
        """
        async with self._gen_lock:
            if self._generating:
                return None
            self._generating = True

        try:
            today = effective_date or datetime.date.today().isoformat()

            # 复用已有
            existing = self.data_mgr.get(today, personality_key)
            if existing:
                return existing

            # 空池
            if not self.pool:
                logger.warning(f"{self._domain}池为空，跳过今日{self._domain}生成")
                return None

            # 选风格
            style_raw = self._pick_style(style_hint, today, personality_key)
            if not style_raw:
                return None

            style_name, style_desc = _parse_outfit_item(style_raw)

            # ── 一次 LLM 调用（life_scheduler 风格） ──
            payload = await self._call_llm_once(style_name, style_desc, umo)

            if payload:
                ok, reason = self._validate_payload(payload, style_name)
                if ok:
                    data = self._make_data(payload, today, style_name, personality_key)
                    self.data_mgr.set(data)
                    logger.info(
                        f"{self._domain}生成成功：{style_name} | chinese={payload.get('chinese','')[:30]}... | tags={payload.get('tags','')}"
                    )
                    return data

                logger.warning(
                    f"{self._domain} JSON 校验失败({reason})，尝试修复..."
                )
                payload = await self._repair_llm(style_name, style_desc, reason, umo)
                if payload:
                    ok, reason2 = self._validate_payload(payload, style_name)
                    if ok:
                        data = self._make_data(payload, today, style_name, personality_key)
                        self.data_mgr.set(data)
                        logger.info(
                            f"{self._domain}修复后生成成功：{style_name}"
                        )
                        return data
                    logger.warning(
                        f"{self._domain}修复后仍校验失败({reason2})"
                    )

            # LLM 完全失败 → 回退
            logger.warning(
                f"{self._domain} LLM 生成失败，回退到静态数据：{style_name}"
            )
            fallback = self._make_fallback(today, style_name, style_desc, personality_key)
            self.data_mgr.set(fallback)
            return fallback

        except Exception as e:
            logger.error(f"{self._domain}生成异常: {e}")
            return None
        finally:
            async with self._gen_lock:
                self._generating = False

    def _pick_style(
        self, style_hint: str, today: datetime.date, personality_key: str = "echo"
    ) -> Optional[str]:
        """从池中选风格，避免近期重复"""
        if style_hint:
            matched = [s for s in self.pool if style_hint.lower() in s.lower()]
            return random.choice(matched) if matched else None

        recent = self.data_mgr.get_recent_styles(personality_key=personality_key, days=3)
        candidates = [s for s in self.pool if s not in recent]
        return random.choice(candidates or self.pool)

    # ──────── LLM 调用（life_scheduler 风格） ────────

    def _get_system_prompt(self) -> Optional[str]:
        """返回 LLM 调用使用的 system prompt。子类可覆盖此方法以使用不同的规则。"""
        return self.pic_guide if self.pic_guide else None

    async def _call_llm_once(
        self, style_name: str, style_desc: str, umo: Optional[str]
    ) -> Optional[dict]:
        """一次 LLM 调用，返回解析后的 JSON dict"""
        prompt = self._build_prompt(style_name, style_desc)
        sid = f"botportrait_{self._domain}_{datetime.date.today().isoformat()}"

        try:
            provider = await self._get_provider(umo)
            if not provider:
                return None

            for attempt in range(self._EMPTY_RETRIES + 1):
                resp = await provider.text_chat(
                    prompt=prompt,
                    system_prompt=self._get_system_prompt(),
                    session_id=sid,
                )
                text = self._extract_text(resp)
                if text:
                    payload = self._extract_json(text)
                    if payload:
                        return payload
                if attempt < self._EMPTY_RETRIES:
                    logger.warning(
                        f"{self._domain} LLM 返回为空，重试..."
                    )
            return None
        finally:
            await self._cleanup_session(sid)

    async def _repair_llm(
        self, style_name: str, style_desc: str, reason: str, umo: Optional[str]
    ) -> Optional[dict]:
        """修复调用：告诉 LLM 上次哪里不合规"""
        prompt = self._build_prompt(style_name, style_desc)
        repair = (
            f"\n\n⚠️ 上次输出不合规，原因：{reason}\n"
            "请严格按 JSON 格式输出，确保字段名和值正确。"
        )
        sid = f"botportrait_{self._domain}_repair_{datetime.date.today().isoformat()}"

        try:
            provider = await self._get_provider(umo)
            if not provider:
                return None
            resp = await provider.text_chat(
                prompt=prompt + repair,
                system_prompt=self._get_system_prompt(),
                session_id=sid,
            )
            text = self._extract_text(resp)
            return self._extract_json(text) if text else None
        finally:
            await self._cleanup_session(sid)

    async def _get_provider(self, umo: Optional[str]):
        """获取 LLM provider"""
        if umo:
            provider_id = await self.context.get_current_chat_provider_id(umo=umo)
            if provider_id:
                return self.context.get_provider_by_id(provider_id)
        return self.context.get_using_provider()

    async def _cleanup_session(self, sid: str):
        """清理 LLM 会话（life_scheduler 做法）"""
        try:
            cid = await self.context.conversation_manager.get_curr_conversation_id(sid)
            if cid:
                await self.context.conversation_manager.delete_conversation(sid, cid)
        except Exception:
            pass

    # ──────── 解析工具 ────────

    @staticmethod
    def _extract_text(resp) -> str:
        if resp is None:
            return ""
        for key in ("completion_text", "completion", "text", "content"):
            val = getattr(resp, key, None)
            if isinstance(val, str) and val.strip():
                return val.strip()
        if isinstance(resp, str) and resp.strip():
            return resp.strip()
        return ""

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """从 LLM 输出中提取 JSON"""
        # 移除 markdown 代码块标记
        text = text.strip()
        text = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^```\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)

        start = text.find("{")
        if start == -1:
            return None

        brace = 0
        in_string = False
        escape = False
        for i, ch in enumerate(text[start:], start=start):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    brace += 1
                elif ch == "}":
                    brace -= 1
                    if brace == 0:
                        try:
                            data = json.loads(text[start : i + 1])
                            return data if isinstance(data, dict) else None
                        except Exception:
                            return None
        return None

    def _make_fallback(
        self, date_str: str, style_name: str, style_desc: str, personality_key: str = ""
    ) -> object:
        """LLM 失败时的静态回退"""
        raise NotImplementedError


# ─────────────────────────────────────────────
# 穿搭生成器
# ─────────────────────────────────────────────


class OutfitGenerator(_BaseGenerator):
    """每日穿搭生成器——life_scheduler 风格，一次 LLM 调用"""

    @property
    def _domain(self) -> str:
        return "outfit"

    def _get_system_prompt(self) -> Optional[str]:
        """穿搭生成器使用服装专用规则，不用分镜导演规则"""
        return (
            "【穿搭标签生成规则】\n\n"
            "你负责将一段中文穿搭描述翻译为英文 SD 服装标签。\n\n"
            "约束：\n"
            "1. 只写服装本身——类型、款式、颜色、图案、材质、配饰\n"
            "2. 禁止写场景（beach、sunny day、indoor 等）\n"
            "3. 禁止写动作/姿势（standing、sitting、pose 等）\n"
            "4. 禁止写表情/眼神（looking at viewer、smile 等）\n"
            "5. 禁止写人物外貌（1girl、white hair 等）\n"
            "6. 纯英文逗号分隔，禁止任何包装格式\n\n"
            "示例：\n"
            '  输入："黑色细带比基尼，三角杯，金属环装饰，弹力涤纶微光泽"\n'
            '  输出："black string bikini, triangle cups, metal ring accents, shiny polyester"\n'
            '  输入："红色波点比基尼，荷叶边，棉质"\n'
            '  输出："red polka dot bikini, ruffled trim, cotton"\n'
            "只输出纯标签，不要多余的解释。"
        )

    def _build_prompt(self, style_name: str, style_desc: str) -> str:
        return (
            f"穿搭风格：{style_name}\n"
            f"风格参考：{style_desc}\n\n"
            "请根据规则书中的穿搭标签规则，将上述穿搭描述翻译为 SD 服装标签，返回以下 JSON：\n"
            "1. chinese：一段中文穿搭描述，150字以内，包含具体款式、颜色、材质\n"
            "2. tags：2-5个英文逗号分隔的 SD 生图标签（只写服装类型、颜色、款式、材质，不写场景/动作/人物外貌）\n\n"
            "必须返回以下 JSON（不要 Markdown，不要代码块，不要解释）：\n"
            f'{{"style": "{style_name}", "chinese": "...", "tags": "..."}}'
        )

    def _validate_payload(self, payload: dict, style_name: str) -> tuple[bool, str]:
        if not payload:
            return False, "未能解析出 JSON"
        chinese = str(payload.get("chinese", "")).strip()
        tags = str(payload.get("tags", "")).strip()
        if not chinese:
            return False, "chinese 不能为空"
        if not tags:
            return False, "tags 不能为空"
        tag_count = len([t for t in tags.split(",") if t.strip()])
        if tag_count < 2:
            return False, f"tags 标签数不足（{tag_count}），至少需要 2 个"
        return True, ""

    def _make_data(self, payload: dict, date_str: str, style_name: str, personality_key: str = "echo") -> DailyOutfit:
        return DailyOutfit(
            date=date_str,
            personality_key=personality_key,
            style=payload.get("style", style_name),
            chinese=payload.get("chinese", ""),
            tags=payload.get("tags", ""),
        )

    def _make_fallback(
        self, date_str: str, style_name: str, style_desc: str, personality_key: str = "echo"
    ) -> DailyOutfit:
        return DailyOutfit(
            date=date_str,
            personality_key=personality_key,
            style=style_name,
            chinese=style_desc or style_name,
            tags="",
        )


# ─────────────────────────────────────────────
# 发型生成器——与穿搭完全相同的架构
# ─────────────────────────────────────────────


class HairstyleGenerator(_BaseGenerator):
    """每日发型生成器——与 OutfitGenerator 逻辑一致"""

    @property
    def _domain(self) -> str:
        return "hairstyle"

    def _get_system_prompt(self) -> Optional[str]:
        """发型生成器使用发型专用规则，不用分镜导演规则"""
        return (
            "【发型标签生成规则】\n\n"
            "你负责将一段中文发型描述翻译为英文 SD 发型标签。\n\n"
            "约束：\n"
            "1. 只写发型本身——造型、长度、束发方式、纹理、刘海\n"
            "2. 禁止写场景（beach、sunny day、indoor 等）\n"
            "3. 禁止写动作/姿势（standing、sitting、pose 等）\n"
            "4. 禁止写表情/眼神（looking at viewer、smile 等）\n"
            "5. 禁止写人物外貌（1girl、white hair 等）\n"
            "6. 纯英文逗号分隔，禁止任何包装格式\n\n"
            "示例：\n"
            '  输入："高马尾，清爽利落，发束向上竖起，富有活力"\n'
            '  输出："high ponytail, swept back, voluminous"\n'
            "只输出纯标签，不要多余的解释。"
        )

    def _build_prompt(self, style_name: str, style_desc: str) -> str:
        return (
            f"发型风格：{style_name}\n"
            f"风格参考：{style_desc}\n\n"
            "请根据规则书中的发型标签规则，将上述发型描述翻译为 SD 发型标签，返回以下 JSON：\n"
            "1. chinese：一段中文发型描述，50字以内，贴合风格定位\n"
            "2. tags：2-4个英文逗号分隔的 SD 生图标签（只写发型类型、长度、造型，不写场景/动作/人物外貌）\n\n"
            "必须返回以下 JSON（不要 Markdown，不要代码块，不要解释）：\n"
            f'{{"style": "{style_name}", "chinese": "...", "tags": "..."}}'
        )

    def _validate_payload(self, payload: dict, style_name: str) -> tuple[bool, str]:
        if not payload:
            return False, "未能解析出 JSON"
        chinese = str(payload.get("chinese", "")).strip()
        tags = str(payload.get("tags", "")).strip()
        if not chinese:
            return False, "chinese 不能为空"
        if not tags:
            return False, "tags 不能为空"
        return True, ""

    def _make_data(
        self, payload: dict, date_str: str, style_name: str, personality_key: str = "echo"
    ) -> DailyHairstyle:
        return DailyHairstyle(
            date=date_str,
            personality_key=personality_key,
            style=payload.get("style", style_name),
            chinese=payload.get("chinese", ""),
            tags=payload.get("tags", ""),
        )

    def _make_fallback(
        self, date_str: str, style_name: str, style_desc: str, personality_key: str = "echo"
    ) -> DailyHairstyle:
        return DailyHairstyle(
            date=date_str,
            personality_key=personality_key,
            style=style_name,
            chinese=style_desc or style_name,
            tags="",
        )
