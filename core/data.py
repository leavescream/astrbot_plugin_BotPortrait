import datetime
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class DailyOutfit:
    """单日穿搭数据——life_scheduler 风格，一次 LLM 调用生成所有字段"""

    date: str  # yyyy-mm-dd
    personality_key: str = ""  # 所属人格
    style: str = ""      # 风格名称，如"比基尼泳装风"
    chinese: str = ""    # 中文穿搭描述（给 LLM system prompt 用）
    tags: str = ""       # 英文 SD 生图标签（给 ComfyUI 用）

    @classmethod
    def from_dict(cls, d: dict) -> "DailyOutfit":
        return cls(
            date=d.get("date", ""),
            personality_key=d.get("personality_key", "echo"),
            style=d.get("style", ""),
            chinese=d.get("chinese", d.get("description", "")),
            tags=d.get("tags", d.get("prompt_tags", "")),
        )


@dataclass(slots=True)
class DailyHairstyle:
    """单日发型数据——与 DailyOutfit 逻辑一致"""

    date: str  # yyyy-mm-dd
    personality_key: str = ""  # 所属人格
    style: str = ""      # 风格名称
    chinese: str = ""    # 中文发型描述（给 LLM system prompt 用）
    tags: str = ""       # 英文 SD 生图标签（给 ComfyUI 用）

    @classmethod
    def from_dict(cls, d: dict) -> "DailyHairstyle":
        return cls(
            date=d.get("date", ""),
            personality_key=d.get("personality_key", "echo"),
            style=d.get("style", ""),
            chinese=d.get("chinese", d.get("description", "")),
            tags=d.get("tags", d.get("prompt_tags", "")),
        )


class OutfitDataManager:
    """
    穿搭数据持久化。
    每日随机从穿搭池挑选，避免连续重复。
    以 JSON 文件存储，跨重启保持一致。
    """

    def __init__(self, json_path: Path):
        self._path = json_path
        self._data: dict[str, DailyOutfit] = {}
        self.load()

    # ---------- CRUD（所有方法新增 personality_key） ----------

    def _key(self, date_str: str, personality_key: str) -> str:
        return f"{date_str}_{personality_key}"

    def get(self, date_str: str, personality_key: str = "echo") -> Optional[DailyOutfit]:
        return self._data.get(self._key(date_str, personality_key))

    def has(self, date_str: str, personality_key: str = "echo") -> bool:
        return self._key(date_str, personality_key) in self._data

    def set(self, outfit: DailyOutfit) -> None:
        self._data[self._key(outfit.date, outfit.personality_key)] = outfit
        self.save(keep_date=outfit.date)

    def reload(self) -> None:
        """从磁盘重新加载所有穿搭数据"""
        self.load()

    def remove(self, date_str: str, personality_key: str = "echo") -> None:
        """删除指定日期+人格的穿搭数据"""
        self._data.pop(self._key(date_str, personality_key), None)
        self.save(keep_date=date_str)

    def today(self, personality_key: str = "echo") -> Optional[DailyOutfit]:
        return self._data.get(self._key(datetime.date.today().isoformat(), personality_key))

    # ---------- 持久化 ----------

    def load(self) -> None:
        if not self._path.exists():
            self._data.clear()
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            self._data.clear()
            return
        data: dict[str, DailyOutfit] = {}
        for stored_key, item in raw.items():
            if not isinstance(item, dict):
                continue
            try:
                outfit = DailyOutfit.from_dict(item)
                # 旧格式：key=日期 → 转为新格式 key=日期_人格
                if "_" not in stored_key and outfit.personality_key:
                    new_key = self._key(outfit.date, outfit.personality_key)
                else:
                    new_key = stored_key
                data[new_key] = outfit
            except Exception:
                continue
        self._data = data

    def save(self, keep_date: str = None) -> None:
        # 只保留当日数据（所有人格），旧记录自动清理
        today = keep_date or datetime.date.today().isoformat()
        self._data = {k: v for k, v in self._data.items() if k.startswith(today)}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        payload = {k: asdict(o) for k, o in self._data.items()}
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._path)

    def get_recent_styles(self, personality_key: str = "echo", days: int = 3) -> list[str]:
        """获取最近几天指定人格的穿搭风格（用于去重）"""
        styles: list[str] = []
        today = datetime.date.today()
        for i in range(1, days + 1):
            d = (today - datetime.timedelta(days=i)).isoformat()
            outfit = self._data.get(self._key(d, personality_key))
            if outfit and outfit.style:
                styles.append(outfit.style)
        return styles


class HairstyleDataManager:
    """
    发型数据持久化。
    每日随机从发型池挑选，避免连续重复。
    以 JSON 文件存储，跨重启保持一致。
    """

    def __init__(self, json_path: Path):
        self._path = json_path
        self._data: dict[str, DailyHairstyle] = {}
        self.load()

    # ---------- CRUD（所有方法新增 personality_key） ----------

    def _key(self, date_str: str, personality_key: str) -> str:
        return f"{date_str}_{personality_key}"

    def get(self, date_str: str, personality_key: str = "echo") -> Optional[DailyHairstyle]:
        return self._data.get(self._key(date_str, personality_key))

    def has(self, date_str: str, personality_key: str = "echo") -> bool:
        return self._key(date_str, personality_key) in self._data

    def set(self, hairstyle: DailyHairstyle) -> None:
        self._data[self._key(hairstyle.date, hairstyle.personality_key)] = hairstyle
        self.save(keep_date=hairstyle.date)

    def reload(self) -> None:
        """从磁盘重新加载所有发型数据"""
        self.load()

    def remove(self, date_str: str, personality_key: str = "echo") -> None:
        """删除指定日期+人格的发型数据"""
        self._data.pop(self._key(date_str, personality_key), None)
        self.save(keep_date=date_str)

    def today(self, personality_key: str = "echo") -> Optional[DailyHairstyle]:
        return self._data.get(self._key(datetime.date.today().isoformat(), personality_key))

    # ---------- 持久化 ----------

    def load(self) -> None:
        if not self._path.exists():
            self._data.clear()
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            self._data.clear()
            return
        data: dict[str, DailyHairstyle] = {}
        for stored_key, item in raw.items():
            if not isinstance(item, dict):
                continue
            try:
                hs = DailyHairstyle.from_dict(item)
                # 旧格式：key=日期 → 转为新格式 key=日期_人格
                if "_" not in stored_key and hs.personality_key:
                    new_key = self._key(hs.date, hs.personality_key)
                else:
                    new_key = stored_key
                data[new_key] = hs
            except Exception:
                continue
        self._data = data

    def save(self, keep_date: str = None) -> None:
        # 只保留当日数据（所有人格），旧记录自动清理
        today = keep_date or datetime.date.today().isoformat()
        self._data = {k: v for k, v in self._data.items() if k.startswith(today)}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        payload = {k: asdict(o) for k, o in self._data.items()}
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._path)

    def get_recent_styles(self, personality_key: str = "echo", days: int = 3) -> list[str]:
        """获取最近几天指定人格的发型风格（用于去重）"""
        styles: list[str] = []
        today = datetime.date.today()
        for i in range(1, days + 1):
            d = (today - datetime.timedelta(days=i)).isoformat()
            hs = self._data.get(self._key(d, personality_key))
            if hs and hs.style:
                styles.append(hs.style)
        return styles
