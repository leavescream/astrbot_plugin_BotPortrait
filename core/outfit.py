import datetime
import random
from typing import Optional

from astrbot.api import logger

from .data import DailyOutfit, OutfitDataManager

def _parse_outfit_item(item: str) -> tuple[str, str]:
    """解析穿搭条目，返回 (风格名称, 描述)

    支持格式：
    - "风格名：描述"
    - "风格名: 描述"
    - "风格名/描述"
    """
    item = item.strip()
    for s in ["：", ":", "/"]:
        if s in item:
            parts = item.split(s, 1)
            return parts[0].strip(), parts[1].strip()
    return item, item


class OutfitPicker:
    """
    每日穿搭选择器。
    从配置池中随机选取今日穿搭，避免连续重复。
    成功时返回 DailyOutfit；失败时返回 None。
    """

    def __init__(self, data_mgr: OutfitDataManager, outfit_pool: list[str]):
        self._data = data_mgr
        self.pool = list(outfit_pool or [])

    def pick_or_reuse(self, personality_key: str = "echo") -> Optional[DailyOutfit]:
        """
        获取今日穿搭：
        - 如果今日已有记录，直接复用
        - 否则从池中随机选择（避免与近期重复）
        - 失败返回 None
        """
        today = datetime.date.today().isoformat()

        # 如果今天已有，直接返回
        existing = self._data.get(today, personality_key)
        if existing:
            return existing

        # 没有穿搭池 -> 无法生成
        if not self.pool:
            logger.warning("穿搭池为空，无法生成每日穿搭")
            return None

        # 获取近期风格，避免重复
        recent = self._data.get_recent_styles(days=3)
        candidates = [s for s in self.pool if s not in recent]
        if not candidates:
            candidates = self.pool

        raw = random.choice(candidates)
        return self._parse_and_save(today, raw, personality_key)

    def pick_or_reuse_with_style(self, style_hint: str, personality_key: str = "echo") -> Optional[DailyOutfit]:
        """
        获取今日穿搭，但优先使用指定风格。
        """
        today = datetime.date.today().isoformat()

        existing = self._data.get(today, personality_key)
        if existing:
            return existing

        if not self.pool:
            return None

        # 尝试匹配指定风格
        matched = [s for s in self.pool if style_hint.lower() in s.lower()]
        if matched:
            raw = random.choice(matched)
        else:
            recent = self._data.get_recent_styles(days=3)
            candidates = [s for s in self.pool if s not in recent]
            raw = random.choice(candidates or self.pool)

        return self._parse_and_save(today, raw, personality_key)

    def _parse_and_save(self, date_str: str, raw: str, personality_key: str = "echo") -> DailyOutfit:
        """解析穿搭条目并保存"""
        style, desc = _parse_outfit_item(raw)
        tags = ""  # 静态回退不生成 tags，由主流程 LLM 完成

        outfit = DailyOutfit(
            date=date_str,
            personality_key=personality_key,
            style=style,
            chinese=desc or raw,
            tags=tags,
        )
        self._data.set(outfit)
        logger.info(f"今日穿搭已选定：{style} | {tags}")
        return outfit

    def update_pool(self, new_pool: list[str]) -> None:
        """运行时更新穿搭池"""
        self.pool = list(new_pool or [])
