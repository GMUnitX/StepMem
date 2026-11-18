"""通用工具：设备/精度解析、增量解码、JSON 序列化。"""

from __future__ import annotations

import logging
from dataclasses import is_dataclass, fields
from typing import Any, Iterable, Sequence

import numpy as np

LOGGER = logging.getLogger("stepmem")


def resolve_device(requested: str) -> str:
    """把 ``auto`` 解析为当前可用的最佳设备。"""
    import torch

    requested = (requested or "auto").lower()
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(requested: str, device: str):
    """选择加载精度：CPU 默认 float32；GPU 默认 bf16（不支持时 fp16）。"""
    import torch

    requested = (requested or "auto").lower()
    mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "float64": torch.float64,
        "double": torch.float64,
    }
    if requested != "auto":
        if requested not in mapping:
            raise ValueError(f"不支持的 dtype: {requested}")
        return mapping[requested]

    device = device.lower()
    if device.startswith("cuda"):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def safe_decode(tokenizer, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
    """兼容不同 tokenizer 的 decode 签名。"""
    ids = [int(i) for i in ids]
    if not ids:
        return ""
    try:
        return tokenizer.decode(
            ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        try:
            return tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        except TypeError:
            return tokenizer.decode(ids)


class IncrementalDecoder:
    """把整段 token 序列的 decode 结果转成增量文本。

    byte-level BPE 在跨 token 的 UTF-8 字符尚未完整时可能先解码出替换字符，
    后续 token 到达后会修正。这里用公共前缀法生成增量，并允许在发生修正时
    重新发送修正后的片段。
    """

    def __init__(self, tokenizer, skip_special_tokens: bool = True) -> None:
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens
        self._ids: list[int] = []
        self._text = ""

    @property
    def text(self) -> str:
        return self._text

    @property
    def ids(self) -> list[int]:
        return list(self._ids)

    def reset(self) -> None:
        self._ids = []
        self._text = ""

    def peek(self, ids: Sequence[int]) -> str:
        return safe_decode(self.tokenizer, ids, self.skip_special_tokens)

    def update(self, ids: Sequence[int]) -> str:
        """更新到给定的完整 token 序列，返回自上次 update 以来的新增文本。"""
        self._ids = [int(i) for i in ids]
        new_text = safe_decode(self.tokenizer, self._ids, self.skip_special_tokens)
        old_text = self._text
        if new_text == old_text:
            return ""
        limit = min(len(old_text), len(new_text))
        common = 0
        while common < limit and old_text[common] == new_text[common]:
            common += 1
        self._text = new_text
        return new_text[common:]


def find_stop_string(text: str, stop_strings: Iterable[str]) -> tuple[int, str] | None:
    """返回最早出现的停止串（位置, 停止串）。"""
    best: tuple[int, str] | None = None
    for stop in stop_strings:
        if not stop:
            continue
        idx = text.find(stop)
        if idx < 0:
            continue
        if best is None or idx < best[0] or (idx == best[0] and len(stop) > len(best[1])):
            best = (idx, stop)
    return best


def tokenizer_special_strings(tokenizer, extra_tokens: Iterable[str] = ()) -> list[str]:
    """收集需要从记忆文本中剔除的特殊 token 字符串，长串优先。"""
    candidates: set[str] = set()
    for token in extra_tokens or ():
        if isinstance(token, str) and token:
            candidates.add(token)

    values = getattr(tokenizer, "all_special_tokens", None)
    if values:
        try:
            candidates.update(str(t) for t in values if isinstance(t, str) and t)
        except TypeError:  # pragma: no cover - 异常 tokenizer
            pass

    additional = getattr(tokenizer, "additional_special_tokens", None)
    if additional:
        try:
            candidates.update(str(t) for t in additional if isinstance(t, str) and t)
        except TypeError:  # pragma: no cover
            pass

    for attr in (
        "eos_token",
        "bos_token",
        "pad_token",
        "unk_token",
        "sep_token",
        "cls_token",
        "mask_token",
    ):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, str) and value:
            candidates.add(value)
    # 长串先替换，避免短串是长串子串时留下残余。
    return sorted(candidates, key=len, reverse=True)


def sanitize_memory_text(
    text: str,
    tokenizer,
    extra_tokens: Iterable[str] = (),
    *,
    strip: bool = True,
) -> str:
    """移除记忆文本中的特殊 token（EOS/BOS/PAD/UNK/chat 模板标记等）。

    StepMem 只保存“可读内容”，不保存控制 token。该函数用于：
    * 生成步骤写入记忆前；
    * 记忆训练写入 assistant 字段前；
    * 旧版记忆注入前。
    """
    if not isinstance(text, str):
        raise TypeError("记忆文本必须是 str")
    if not text:
        return ""
    result = text
    for token in tokenizer_special_strings(tokenizer, extra_tokens):
        if token and token in result:
            result = result.replace(token, "")
    return result.strip() if strip else result


def strip_matched_prefix(memory_text: str, piece: str) -> str:
    """若记忆文本以 piece 开头则去掉该前缀，用于可见注入避免重复首 token。

    会同时尝试 piece 的首尾空白变体，以及记忆文本自身的前导空白，
    以兼容 BPE 把空格并入 token 的情况。
    """
    if not memory_text or not piece:
        return memory_text
    variants = {
        piece,
        piece.strip(),
        piece.lstrip(),
        piece.rstrip(),
    }
    variants = sorted((v for v in variants if v), key=len, reverse=True)

    for variant in variants:
        if memory_text.startswith(variant):
            return memory_text[len(variant) :]

    stripped = memory_text.lstrip()
    if stripped != memory_text:
        for variant in variants:
            if stripped.startswith(variant):
                return stripped[len(variant) :]
    return memory_text


def parameter_device(model):
    """模型参数所在设备（兼容 device_map 分片时取第一个参数）。"""
    import torch

    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover
        return torch.device("cpu")


def hidden_size_of(model) -> int:
    """从常见配置字段推断隐藏维度。"""
    config = getattr(model, "config", None)
    for name in ("hidden_size", "n_embd", "d_model", "dim"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    embeddings = model.get_input_embeddings()
    if embeddings is not None and hasattr(embeddings, "embedding_dim"):
        return int(embeddings.embedding_dim)
    raise ValueError("无法从模型配置推断 hidden_size，请在自定义模型中显式提供")


def max_positions_of(model) -> int:
    config = getattr(model, "config", None)
    for name in ("max_position_embeddings", "n_positions", "max_sequence_length", "seq_length"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return 2048


def to_jsonable(obj: Any) -> Any:
    """递归转换为可 JSON 序列化的结构（numpy/torch/dataclass 均可）。"""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # torch.Tensor 等
    if hasattr(obj, "detach") and hasattr(obj, "cpu"):
        tensor = obj.detach().cpu()
        return tensor.tolist() if tensor.ndim else tensor.item()
    return str(obj)
