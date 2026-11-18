"""StepMem · 步骤记忆拓展。

用模型生成过程的注意力与隐藏状态驱动可扩展步骤记忆的 Hugging Face 推理引擎：
支持步骤切分、步骤原文记忆、蒸馏数据训练、只读服务与可见记忆注入。
"""

from __future__ import annotations

from .config import (
    RetrievalConfig,
    SamplingConfig,
    StepMemConfig,
    SteppingConfig,
    StoreConfig,
)
from .engine import StepMemEngine
from .errors import (
    AttentionUnavailableError,
    ContextOverflowError,
    DimensionMismatchError,
    StepMemError,
    StoreError,
)
from .sampling import Sampler
from .stepping import (
    AttentionSample,
    ClosedStep,
    StepDecision,
    StepTracker,
    cosine_overlap,
    reduce_attention,
    resolve_attention_layers,
)
from .store import VectorStore
from .training import (
    MemoryTrainer,
    MemoryTrainingError,
    MemoryTrainingResult,
    MemoryTrainingStats,
    load_memory_dataset,
    validate_memory_record,
)
from .types import (
    DoneEvent,
    GenerationResult,
    GenerationStats,
    InjectionEvent,
    InjectionInfo,
    MemoryRecord,
    ProfileResult,
    RetrievalEvent,
    RetrievalHit,
    RetrievalInfo,
    StepEndEvent,
    StepInfo,
    StepStartEvent,
    TokenEvent,
)
from .utils import sanitize_memory_text, strip_matched_prefix

__version__ = "0.3.0"

__all__ = [
    "StepMemEngine",
    "StepMemConfig",
    "SteppingConfig",
    "RetrievalConfig",
    "StoreConfig",
    "SamplingConfig",
    "VectorStore",
    "StepTracker",
    "AttentionSample",
    "ClosedStep",
    "StepDecision",
    "reduce_attention",
    "resolve_attention_layers",
    "cosine_overlap",
    "Sampler",
    "MemoryTrainer",
    "MemoryTrainingResult",
    "MemoryTrainingStats",
    "MemoryTrainingError",
    "load_memory_dataset",
    "validate_memory_record",
    "sanitize_memory_text",
    "strip_matched_prefix",
    "MemoryRecord",
    "RetrievalHit",
    "RetrievalInfo",
    "StepInfo",
    "InjectionInfo",
    "GenerationStats",
    "GenerationResult",
    "ProfileResult",
    "TokenEvent",
    "StepStartEvent",
    "StepEndEvent",
    "RetrievalEvent",
    "InjectionEvent",
    "DoneEvent",
    "StepMemError",
    "AttentionUnavailableError",
    "ContextOverflowError",
    "StoreError",
    "DimensionMismatchError",
    "__version__",
]
