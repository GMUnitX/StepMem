"""注意力探针与步骤边界判定。

``reduce_attention`` 把 HF 返回的每层注意力 ``[1, heads, 1, kv_len]`` 归约成一个
一维权重向量；``StepTracker`` 以纯 NumPy 状态机实现相邻 token 的余弦相似度判定，
不依赖具体模型，便于单元测试。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .config import SteppingConfig


@dataclass(frozen=True)
class AttentionSample:
    """某个 token 的注意力分布及其在上下文中的绝对位置范围。

    ``vector[j]`` 对应上下文绝对位置 ``start_position + j``。正常情况下
    ``start_position == 0``；使用滑动窗口/裁剪 KV cache 时 start 会大于 0，
    余弦比较前按绝对位置求交集，避免错位。
    """

    vector: np.ndarray
    start_position: int

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector, dtype=np.float32).reshape(-1)
        object.__setattr__(self, "vector", vector)
        object.__setattr__(self, "start_position", int(self.start_position))

    @property
    def length(self) -> int:
        return int(self.vector.shape[0])

    @property
    def end_position(self) -> int:
        return self.start_position + self.length


def resolve_attention_layers(
    layers: Optional[Sequence[int]], num_layers: int
) -> list[int]:
    """解析层选择，支持负数索引。``None`` 表示全部层。"""
    if num_layers <= 0:
        raise ValueError("num_layers 必须 > 0")
    if layers is None:
        return list(range(num_layers))
    resolved: list[int] = []
    for layer in layers:
        idx = int(layer)
        if idx < 0:
            idx += num_layers
        if not (0 <= idx < num_layers):
            raise ValueError(
                f"attention_layers 中的 {layer} 超出模型层数范围 "
                f"[-{num_layers}, {num_layers - 1}]"
            )
        if idx not in resolved:
            resolved.append(idx)
    if not resolved:
        raise ValueError("attention_layers 不能为空")
    return resolved


def reduce_attention(
    attentions: Sequence,
    layers: Optional[Sequence[int]] = (-1,),
    head_reduction: str = "mean",
) -> np.ndarray:
    """把每层注意力归约为一个 CPU float32 向量。

    参数
    ----
    attentions:
        transformers 返回的 tuple，每层形状 ``[batch, heads, q_len, kv_len]``。
        解码阶段 q_len 通常为 1，这里取最后一行。
    layers:
        参与平均的层索引；``(-1,)`` 表示最后一层，``None`` 表示全部层。
    head_reduction:
        ``mean``（默认）或 ``first``（仅调试用）。
    """
    if not attentions:
        raise ValueError("attentions 为空")
    import torch

    num_layers = len(attentions)
    layer_indices = resolve_attention_layers(layers, num_layers)
    if head_reduction not in {"mean", "first"}:
        raise ValueError("head_reduction 必须是 'mean' 或 'first'")

    total = None
    with torch.inference_mode():
        for layer_idx in layer_indices:
            attn = attentions[layer_idx]
            if attn is None:
                raise ValueError(f"第 {layer_idx} 层没有返回注意力权重")
            if attn.dim() != 4:
                raise ValueError(
                    f"注意力张量维度应为 4，收到 {tuple(attn.shape)}"
                )
            if attn.shape[0] != 1:
                raise ValueError("StepMem 目前只支持 batch_size=1")
            row = attn[0, :, -1, :].to(torch.float32)  # [heads, kv_len]
            vec = row.mean(dim=0) if head_reduction == "mean" else row[0]
            # 逐层回传 CPU 再累加：兼容 device_map 跨多张 GPU 分层放置的情况。
            vec_cpu = vec.detach().to("cpu", torch.float32).numpy()
            total = vec_cpu if total is None else total + vec_cpu
    assert total is not None
    total = total / float(len(layer_indices))
    return np.ascontiguousarray(total, dtype=np.float32)


def cosine_overlap(
    a: Optional[AttentionSample], b: Optional[AttentionSample], eps: float = 1e-8
) -> Optional[float]:
    """两个注意力分布在绝对位置交集上的余弦相似度。

    返回 ``None`` 表示无法比较（无交集、零向量或非有限值）。
    """
    if a is None or b is None:
        return None
    lo = max(a.start_position, b.start_position)
    hi = min(a.end_position, b.end_position)
    if hi <= lo:
        return None
    av = a.vector[lo - a.start_position : hi - a.start_position]
    bv = b.vector[lo - b.start_position : hi - b.start_position]
    if av.size == 0 or bv.size == 0:
        return None
    if not (np.all(np.isfinite(av)) and np.all(np.isfinite(bv))):
        return None
    na = float(np.linalg.norm(av))
    nb = float(np.linalg.norm(bv))
    if na < eps or nb < eps:
        return None
    sim = float(np.dot(av, bv) / (na * nb))
    return max(-1.0, min(1.0, sim))


@dataclass
class ClosedStep:
    """已完成步骤的不可变快照。"""

    index: int
    start_position: int
    end_position: int
    token_ids: list[int]
    key: np.ndarray
    start_reason: str
    start_similarity: Optional[float]


@dataclass
class StepDecision:
    """一次 ``observe`` 的判定结果。"""

    started: bool
    closed: Optional[ClosedStep]
    similarity: Optional[float]
    reason: str  # initial | threshold | max_tokens | ""
    step_index: int


class StepTracker:
    """逐 token 的步骤切分状态机。

    生命周期::

        tracker = StepTracker(config)
        decision = tracker.observe(position, token_id, hidden, attention)
        ...
        final_step = tracker.finish()

    第一个被观察的 token 永远开启第一个步骤，其隐藏状态既是
    查询键也是该步骤的存储键。
    """

    def __init__(self, config: SteppingConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._index = -1
        self._start_pos = 0
        self._start_key: Optional[np.ndarray] = None
        self._start_reason = ""
        self._start_sim: Optional[float] = None
        self._token_ids: list[int] = []
        self._last_attn: Optional[AttentionSample] = None
        self._last_pos = 0
        self._finished = False

    @property
    def has_open_step(self) -> bool:
        return self._index >= 0 and not self._finished

    @property
    def current_index(self) -> int:
        return self._index

    @property
    def current_token_count(self) -> int:
        return len(self._token_ids)

    @property
    def current_start_position(self) -> int:
        return self._start_pos

    def _begin(
        self,
        position: int,
        hidden: np.ndarray,
        reason: str,
        similarity: Optional[float],
    ) -> None:
        self._index += 1
        self._start_pos = int(position)
        self._start_key = np.asarray(hidden, dtype=np.float32).reshape(-1).copy()
        self._start_reason = reason
        self._start_sim = similarity
        self._token_ids = []

    def _close(self, end_position: int) -> ClosedStep:
        assert self._start_key is not None
        return ClosedStep(
            index=self._index,
            start_position=self._start_pos,
            end_position=int(end_position),
            token_ids=list(self._token_ids),
            key=self._start_key.copy(),
            start_reason=self._start_reason,
            start_similarity=self._start_sim,
        )

    def observe(
        self,
        position: int,
        token_id: int,
        hidden: np.ndarray,
        attention: Optional[AttentionSample],
    ) -> StepDecision:
        """观察一个刚生成并完成前向的 token。"""
        if self._finished:
            raise RuntimeError("StepTracker.finish() 之后不能再 observe")
        position = int(position)
        token_id = int(token_id)
        hidden = np.asarray(hidden, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(hidden)):
            raise ValueError("hidden state 包含 NaN/Inf")

        # 第一个 token：强制开启第一个步骤。
        if self._index < 0:
            self._begin(position, hidden, "initial", None)
            self._token_ids.append(token_id)
            self._last_attn = attention
            self._last_pos = position
            return StepDecision(
                started=True,
                closed=None,
                similarity=None,
                reason="initial",
                step_index=self._index,
            )

        similarity = cosine_overlap(self._last_attn, attention)

        reason = ""
        if self.config.max_step_tokens and len(self._token_ids) >= self.config.max_step_tokens:
            reason = "max_tokens"
        elif (
            similarity is not None
            and similarity < self.config.boundary_threshold
            and len(self._token_ids) >= self.config.min_step_tokens
        ):
            reason = "threshold"

        if reason:
            closed = self._close(self._last_pos)
            self._begin(position, hidden, reason, similarity)
            self._token_ids.append(token_id)
            self._last_attn = attention
            self._last_pos = position
            return StepDecision(
                started=True,
                closed=closed,
                similarity=similarity,
                reason=reason,
                step_index=self._index,
            )

        self._token_ids.append(token_id)
        self._last_attn = attention
        self._last_pos = position
        return StepDecision(
            started=False,
            closed=None,
            similarity=similarity,
            reason="",
            step_index=self._index,
        )

    def finish(self) -> Optional[ClosedStep]:
        """结束生成，返回尚未关闭的步骤（没有则返回 None）。"""
        if self._finished:
            return None
        self._finished = True
        if self._index < 0 or not self._token_ids:
            return None
        return self._close(self._last_pos)
