"""StepMem 的数据结构与流式事件。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np

from .utils import to_jsonable


@dataclass
class MemoryRecord:
    """向量库中的一条步骤记忆。文本保持模型生成时的原始形态。"""

    id: int
    text: str
    session_id: str
    step_index: int
    token_count: int
    created_at: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalHit:
    """一次检索结果。similarity 为余弦相似度（键已归一化）。"""

    record: MemoryRecord
    similarity: float


@dataclass
class RetrievalInfo:
    """一次检索决策记录。

    position 是触发检索的 token 在上下文中的绝对位置；写入模式下通常是步骤首
    token，非写入模式下是每个生成 token。
    """

    similarity: Optional[float]
    threshold: float
    accepted: bool
    reason: str
    record_id: Optional[int] = None
    record_text: Optional[str] = None
    injected_tokens: int = 0
    position: Optional[int] = None


@dataclass
class StepInfo:
    """生成结果中的一个步骤。位置均为上下文中的绝对 token 位置。"""

    index: int
    start_position: int
    end_position: int
    token_ids: list[int]
    text: str
    boundary_similarity: Optional[float]
    boundary_reason: str
    stored: bool = False
    record_id: Optional[int] = None
    retrieval: Optional[RetrievalInfo] = None


@dataclass
class InjectionInfo:
    """一次成功的记忆注入。text 是实际写入回复/上下文的文本。

    step_index=-1 表示非写入模式（按 token 检索，没有步骤）。
    """

    step_index: int
    record_id: int
    similarity: float
    text: str
    token_count: int
    position: int


@dataclass
class GenerationStats:
    prompt_tokens: int
    generated_tokens: int
    steps: int
    boundaries: int
    retrievals: int
    injections: int
    injected_tokens: int
    stored_steps: int
    elapsed_seconds: float
    tokens_per_second: float
    finish_reason: str


@dataclass
class GenerationResult:
    """一次完整生成的可审计结果。"""

    text: str
    prompt_text: str
    context_text: str
    steps: list[StepInfo]
    injections: list[InjectionInfo]
    stats: GenerationStats
    session_id: str
    retrievals: list[RetrievalInfo] = field(default_factory=list)

    def to_dict(self, include_context: bool = True, include_token_ids: bool = False) -> dict[str, Any]:
        data = to_jsonable(self)
        if not include_context:
            data.pop("context_text", None)
        for step in data.get("steps", []):
            tokens = step.get("token_ids") or []
            step["token_count"] = len(tokens)
            if not include_token_ids:
                step.pop("token_ids", None)
        return data


@dataclass
class ProfileResult:
    """用于阈值校准的原始统计结果。"""

    text: str
    token_ids: list[int]
    token_texts: list[str]
    attention_similarities: list[Optional[float]]
    hidden_states: np.ndarray
    finish_reason: str

    def summary(self) -> dict[str, Any]:
        sims = [s for s in self.attention_similarities if s is not None]
        return {
            "generated_tokens": len(self.token_ids),
            "attention_similarity_count": len(sims),
            "attention_similarity_min": float(np.min(sims)) if sims else None,
            "attention_similarity_max": float(np.max(sims)) if sims else None,
            "finish_reason": self.finish_reason,
        }


# ----------------------------------------------------------------------------
# 流式事件
# ----------------------------------------------------------------------------


@dataclass
class TokenEvent:
    """新生成一个 token。text 为相对上次事件的增量文本。

    ``generated_text`` 是截至当前的完整可见输出（可能包含已注入的记忆文本），
    而不是仅模型生成的 token 文本。非写入模式 ``step_index`` 为 -1。
    """

    token_id: int
    text: str
    context_position: int
    step_index: int
    generated_text: str


@dataclass
class StepStartEvent:
    step_index: int
    context_position: int
    reason: str  # initial | threshold | max_tokens
    similarity: Optional[float]


@dataclass
class StepEndEvent:
    step_index: int
    text: str
    token_count: int
    stored: bool
    record_id: Optional[int]
    is_final: bool = False


@dataclass
class RetrievalEvent:
    step_index: int  # -1 表示非写入模式的逐 token 检索
    similarity: Optional[float]
    accepted: bool
    reason: str
    record_id: Optional[int] = None
    record_text: Optional[str] = None
    position: Optional[int] = None


@dataclass
class InjectionEvent:
    """记忆注入事件：text 同时写入可见回复和 KV cache。

    流式消费方需要把 ``text`` 追加到可见输出；``DoneEvent.result.text`` 已包含它。
    非写入模式 ``step_index`` 为 -1。
    """

    step_index: int
    record_id: int
    similarity: float
    text: str
    token_count: int
    position: int


@dataclass
class DoneEvent:
    result: GenerationResult


Event = Union[
    TokenEvent,
    StepStartEvent,
    StepEndEvent,
    RetrievalEvent,
    InjectionEvent,
    DoneEvent,
]
