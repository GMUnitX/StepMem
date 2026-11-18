"""StepMem 异常类型。"""

from __future__ import annotations


class StepMemError(Exception):
    """StepMem 的基础异常。"""


class AttentionUnavailableError(StepMemError):
    """无法获取注意力权重或隐藏状态。

    注意力探针要求模型使用 eager attention，并且 transformers 版本支持
    ``output_attentions``。部分模型/后端（例如 FlashAttention）不会返回注意力权重。
    """


class ContextOverflowError(StepMemError):
    """上下文长度超过模型可处理的上限。"""


class StoreError(StepMemError):
    """向量存储读写失败。"""


class DimensionMismatchError(StoreError):
    """查询键/存储键的维度与索引维度不一致。"""
