import asyncio
import datetime
import json
import os
import random
import tempfile
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import aiohttp
from astrbot.api import logger

# === 默认固定提示词（可在配置面板中覆盖） ===
DEFAULT_CORE_PROMPT = (
    "1girl, white hair, blue eyes, grayish black mechanical body, "
    "mechanical parts, loose strands of hair, black ribbon, movie lighting"
)
DEFAULT_CORE_NEGATIVE = (
    "nsfw, lowres, bad anatomy, bad hands, text, error, "
    "missing fingers, extra digit, fewer digits, cropped, "
    "worst quality, low quality, normal quality, jpeg artifacts, "
    "signature, watermark, username, blurry"
)


class ComfyUIClient:
    """
    ComfyUI API 客户端。
    封装工作流提交、轮询、图片下载。
    """

    def __init__(
        self,
        api_url: str,
        workflow_path: str,
        core_prompt: str = "",
        core_negative: str = "",
        timeout: int = 120,
        poll_interval: float = 1.0,
    ):
        self._api_url = api_url.rstrip("/")
        self._workflow_path = workflow_path
        self._core_prompt = core_prompt or DEFAULT_CORE_PROMPT
        self._core_negative = core_negative or DEFAULT_CORE_NEGATIVE
        self._timeout = timeout
        self._poll_interval = poll_interval

        # 运行时缓存：工作流模板（惰性加载）
        self._workflow_cache: Optional[dict] = None

    # ---------- 工作流加载 ----------

    def _load_workflow(self) -> dict:
        """加载工作流 JSON 文件"""
        if self._workflow_cache is not None:
            return self._workflow_cache

        path = Path(self._workflow_path)
        if not path.exists():
            raise FileNotFoundError(f"工作流文件不存在: {self._workflow_path}")

        with open(path, encoding="utf-8") as f:
            workflow = json.load(f)

        self._workflow_cache = workflow
        return workflow

    def clear_workflow_cache(self):
        """清除工作流缓存（工作流文件更新后调用）"""
        self._workflow_cache = None

    # ---------- 提示词注入 ----------

    def inject_prompt(
        self,
        workflow: dict,
        positive: str,
        negative: str = "",
        filename_prefix: str = "echo",
        **engine_kwargs,
    ) -> dict:
        """
        将提示词注入工作流。

        关键改进：通过 KSampler 的连线追踪 positive/negative 分别连到哪个
        CLIPTextEncode 节点，按实际连线分配，而非按遍历顺序猜测。
        """
        wf = json.loads(json.dumps(workflow))  # 深拷贝

        # 第一步：找到 KSampler，读取它的 positive/negative 连线目标节点 ID
        sampler_node = None
        target_positive_id = None
        target_negative_id = None

        for node_id, node in wf.items():
            if not isinstance(node, dict):
                continue
            cls_type = node.get("class_type", "")
            inputs = node.get("inputs", {})

            if cls_type in (
                "KSampler", "KSamplerAdvanced",
                "SamplerCustom", "SamplerCustomAdvanced",
            ):
                sampler_node = node
                pos_conn = inputs.get("positive")
                neg_conn = inputs.get("negative")
                if isinstance(pos_conn, list) and len(pos_conn) >= 1:
                    target_positive_id = str(pos_conn[0])
                if isinstance(neg_conn, list) and len(neg_conn) >= 1:
                    target_negative_id = str(neg_conn[0])
                break  # 找到第一个采样器就够

        # 第二步：根据连线 ID 分配正负节点
        positive_node = None
        negative_node = None
        empty_latent_node = None
        checkpoint_node = None
        lora_node = None

        for node_id, node in wf.items():
            if not isinstance(node, dict):
                continue
            cls_type = node.get("class_type", "")
            inputs = node.get("inputs", {})

            if cls_type == "CLIPTextEncode":
                if target_positive_id and node_id == target_positive_id:
                    positive_node = node
                elif target_negative_id and node_id == target_negative_id:
                    negative_node = node
                elif positive_node is None:
                    # 后备：找不到连线时按首个赋值
                    positive_node = node
                elif negative_node is None:
                    negative_node = node

            elif cls_type == "EmptyLatentImage":
                empty_latent_node = node
            elif cls_type == "CheckpointLoaderSimple":
                checkpoint_node = node
            elif cls_type == "LoraLoader":
                lora_node = node

        # 第三步：注入提示词
        effective_negative = negative or self._core_negative

        if positive_node:
            positive_node["inputs"]["text"] = positive
        if negative_node:
            negative_node["inputs"]["text"] = effective_negative
        elif positive_node:
            # 只有一个 CLIPTextEncode 时合并在正向里
            positive_node["inputs"]["text"] = f"{positive}\n{effective_negative}"

        # 第四步：注入采样器参数，并将 seed 设为一个随机值
        if sampler_node:
            inputs = sampler_node["inputs"]
            inputs["seed"] = random.getrandbits(64)  # 随机种子，范围 0 ~ 2^64-1
            for key in ("steps", "cfg", "sampler_name", "scheduler"):
                if key in engine_kwargs:
                    inputs[key] = (
                        int(engine_kwargs[key])
                        if key in ("steps",)
                        else float(engine_kwargs[key])
                        if key == "cfg"
                        else engine_kwargs[key]
                    )

        # 第五步：注入图片尺寸
        if empty_latent_node:
            inputs = empty_latent_node["inputs"]
            if "width" in engine_kwargs:
                inputs["width"] = int(engine_kwargs["width"])
            if "height" in engine_kwargs:
                inputs["height"] = int(engine_kwargs["height"])

        # 第六步：注入模型和 LoRA
        if checkpoint_node and "model" in engine_kwargs and engine_kwargs["model"]:
            checkpoint_node["inputs"]["ckpt_name"] = engine_kwargs["model"]
        if lora_node and "lora" in engine_kwargs and engine_kwargs["lora"]:
            lora_node["inputs"]["lora_name"] = engine_kwargs["lora"]
            if "lora_strength" in engine_kwargs:
                lora_node["inputs"]["strength_model"] = engine_kwargs["lora_strength"]
            if "lora_clip_strength" in engine_kwargs:
                lora_node["inputs"]["strength_clip"] = engine_kwargs["lora_clip_strength"]

        # 第七步：为 SaveImage 节点设置唯一 prefix，防止 ComfyUI output 文件互相覆盖
        for node_id, node in wf.items():
            if not isinstance(node, dict):
                continue
            if node.get("class_type") == "SaveImage":
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                node["inputs"]["filename_prefix"] = f"{filename_prefix}_{ts}"
                break

        return wf

    # ---------- 模型/LoRA 存在性验证 ----------

    @staticmethod
    def _extract_names_from_field(field_value) -> list:
        """
        从 ComfyUI /object_info 返回的字段中提取名称列表。
        兼容多种返回格式：
        - ["STRING", ["name1", "name2"]]      → 取索引 1
        - [["name1", "name2"], {"default": ...}] → 取索引 0
        - ["STRING", {"mylist": [...], ...}]   → 取 mylist
        - ["name1", "name2"]                   → 本身就是列表
        """
        if not isinstance(field_value, list):
            return []

        # 情况 1：字段值本身就是字符串列表
        if field_value and all(isinstance(v, str) for v in field_value):
            return field_value

        # 情况 2：遍历子元素，寻找包含最多字符串的列表
        best = []
        for item in field_value:
            if isinstance(item, list):
                names = [s for s in item if isinstance(s, str)]
                if len(names) > len(best):
                    best = names
            elif isinstance(item, dict):
                # 兼容 {"mylist": [...], ...} 格式
                for val in item.values():
                    if isinstance(val, list):
                        names = [s for s in val if isinstance(s, str)]
                        if len(names) > len(best):
                            best = names

        return best

    async def validate_model_and_lora(
        self, model_name: str = "", lora_name: str = ""
    ) -> Optional[str]:
        """
        验证 ComfyUI 中是否存在指定的模型和 LoRA。
        模型和 LoRA 一起检查，返回错误信息或 None（验证通过）。
        - model_name 为空时跳过模型检查
        - lora_name 为空时跳过 LoRA 检查
        """
        missing = []

        try:
            async with aiohttp.ClientSession() as session:
                # 查模型
                if model_name:
                    url = f"{self._api_url}/object_info/CheckpointLoaderSimple"
                    async with session.get(url, timeout=10) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            ckpt_info = data.get("CheckpointLoaderSimple", {})
                            field = ckpt_info.get("input", {}).get("required", {}).get("ckpt_name", [])
                            available = self._extract_names_from_field(field)
                            if model_name not in available:
                                missing.append(f"模型「{model_name}」")
                        else:
                            missing.append(f"无法连接 ComfyUI 查询模型列表（HTTP {resp.status}）")

                # 查 LoRA
                if lora_name:
                    url = f"{self._api_url}/object_info/LoraLoader"
                    async with session.get(url, timeout=10) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            lora_info = data.get("LoraLoader", {})
                            field = lora_info.get("input", {}).get("required", {}).get("lora_name", [])
                            available = self._extract_names_from_field(field)
                            if lora_name not in available:
                                missing.append(f"LoRA「{lora_name}」")
                        else:
                            missing.append(f"无法连接 ComfyUI 查询 LoRA 列表（HTTP {resp.status}）")

        except aiohttp.ClientError as e:
            return f"无法连接 ComfyUI（{e}）"
        except Exception as e:
            return f"验证模型/LoRA 时异常: {e}"

        if missing:
            return "ComfyUI 中未找到: " + "、".join(missing)
        return None

    # ---------- API 调用 ----------

    async def generate(self, prompt: str, negative: str = "", filename_prefix: str = "echo", **engine_kwargs) -> Optional[str]:
        """
        完整生成流程：
        1. 加载工作流
        2. 注入提示词和参数
        3. 提交到 ComfyUI
        4. 轮询直到完成
        5. 下载图片
        6. 返回本地文件路径
        """
        workflow = self._load_workflow()
        wf = self.inject_prompt(workflow, prompt, negative, filename_prefix=filename_prefix, **engine_kwargs)

        # 提交工作流
        prompt_id = await self._queue_prompt(wf)
        if not prompt_id:
            logger.error("ComfyUI 队列提交失败")
            return None

        # 轮询等待完成
        logger.info(f"ComfyUI 任务已提交: {prompt_id}，等待生成...")
        images = await self._wait_for_completion(prompt_id)
        if not images:
            logger.error("ComfyUI 生成未返回图片")
            return None

        # 下载第一张图片
        img_info = images[0]
        return await self._download_image(img_info)

    async def _queue_prompt(self, workflow: dict) -> Optional[str]:
        """提交工作流到 ComfyUI，返回 prompt_id"""
        payload = {
            "client_id": str(uuid.uuid4()),
            "prompt": workflow,
        }
        url = f"{self._api_url}/prompt"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=30) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        logger.error(f"ComfyUI POST /prompt 失败 [{resp.status}]: {text}")
                        return None
                    data = await resp.json()
                    return data.get("prompt_id")
        except aiohttp.ClientError as e:
            logger.error(f"ComfyUI 请求异常: {e}")
            return None

    async def _wait_for_completion(self, prompt_id: str) -> Optional[list[dict]]:
        """轮询 ComfyUI history 直到任务完成"""
        url = f"{self._api_url}/history/{prompt_id}"
        deadline = asyncio.get_event_loop().time() + self._timeout

        while asyncio.get_event_loop().time() < deadline:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=10) as resp:
                        if resp.status != 200:
                            await asyncio.sleep(self._poll_interval)
                            continue
                        data = await resp.json()
                        prompt_data = data.get(prompt_id)
                        if not prompt_data:
                            await asyncio.sleep(self._poll_interval)
                            continue

                        # ---- 修复：检测任务是否已完成（无论成功/失败） ----
                        status = prompt_data.get("status", {})
                        completed = status.get("completed", False)

                        outputs = prompt_data.get("outputs", {})
                        images = []
                        for node_id, node_out in outputs.items():
                            for img in node_out.get("images", []):
                                if img.get("type") == "output":
                                    images.append(img)

                        if images:
                            logger.info(f"ComfyUI 生成完成，获得 {len(images)} 张图片")
                            return images

                        # 任务已完成但无图片 → 生成失败
                        if completed:
                            messages = status.get("messages", [])
                            error_msg = ""
                            for msg in messages:
                                if isinstance(msg, list) and len(msg) >= 2:
                                    error_msg = str(msg[1])
                            logger.error(
                                f"ComfyUI 任务失败 (prompt_id={prompt_id[:8]}): {error_msg}"
                            )
                            return None

                        # 仍在生成
                        await asyncio.sleep(self._poll_interval)
            except aiohttp.ClientError:
                await asyncio.sleep(self._poll_interval)
                continue

        logger.error(f"ComfyUI 生成超时 ({self._timeout}s): {prompt_id}")
        return None

    async def _download_image(self, img_info: dict) -> Optional[str]:
        """从 ComfyUI 下载图片到本地临时文件"""
        filename = img_info.get("filename")
        subfolder = img_info.get("subfolder", "")
        img_type = img_info.get("type", "output")

        if not filename:
            logger.error("图片信息缺少 filename")
            return None

        parsed = urlparse(self._api_url)
        view_url = (
            f"{parsed.scheme}://{parsed.netloc}/view"
            f"?filename={filename}&subfolder={subfolder}&type={img_type}"
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(view_url, timeout=30) as resp:
                    if resp.status != 200:
                        logger.error(f"下载图片失败 [{resp.status}]")
                        return None
                    img_data = await resp.read()

            ext = os.path.splitext(filename)[1] or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(img_data)
                local_path = tmp.name

            logger.info(f"图片已下载: {local_path}")
            return local_path
        except aiohttp.ClientError as e:
            logger.error(f"图片下载异常: {e}")
            return None
        except Exception as e:
            logger.error(f"保存图片失败: {e}")
            return None
