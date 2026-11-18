"""阈值校准：用真实生成过程估计注意力边界阈值与检索阈值。

步骤边界阈值与检索阈值高度依赖模型和文本分布。StepMem 不采用固定假设，而是
实际运行一段无记忆生成，统计：

* 相邻 token 注意力分布的余弦相似度 → 按目标平均步骤长度取分位数；
* 步骤首 token 隐藏状态的成对余弦相似度 → 取同主题分位数作为检索阈值参考，
  同时报告跨主题分布，便于判断两个分布的分离程度。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from .engine import StepMemEngine
from .types import ProfileResult


@dataclass
class CalibrationResult:
    report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.report


def _percentiles(values: Sequence[float], points: Sequence[float]) -> dict[str, Optional[float]]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {f"p{p:g}": None for p in points}
    out = np.percentile(arr, list(points))
    return {f"p{p:g}": float(v) for p, v in zip(points, out)}


def _pairwise_cosines(states: np.ndarray) -> list[float]:
    if states.ndim != 2 or states.shape[0] < 2:
        return []
    norms = np.linalg.norm(states, axis=1, keepdims=True)
    # 隐藏状态一般不为零；若出现零向量则跳过。
    valid = norms[:, 0] > 1e-8
    states = states[valid]
    norms = norms[valid]
    if states.shape[0] < 2:
        return []
    normalized = states / norms
    matrix = normalized @ normalized.T
    iu, ju = np.triu_indices(matrix.shape[0], k=1)
    return [float(v) for v in matrix[iu, ju]]


def calibrate(
    engine: StepMemEngine,
    prompts: Sequence[str],
    *,
    max_new_tokens: int = 64,
    target_step_tokens: int = 8,
    seed: Optional[int] = None,
) -> CalibrationResult:
    """对一组 prompt 做校准，返回可直接打印/序列化的报告。"""
    if not prompts:
        raise ValueError("校准至少需要一个 prompt")
    if target_step_tokens < 1:
        raise ValueError("target_step_tokens 必须 >= 1")
    if max_new_tokens < 2:
        raise ValueError("max_new_tokens 至少为 2 才能得到相邻相似度")

    profiles: list[ProfileResult] = []
    per_prompt: list[dict[str, Any]] = []
    for idx, prompt in enumerate(prompts):
        profile = engine.profile(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            sampling={"do_sample": False, "seed": seed},
        )
        profiles.append(profile)
        attn = [s for s in profile.attention_similarities if s is not None]
        per_prompt.append(
            {
                "index": idx,
                "prompt": prompt,
                "generated_tokens": len(profile.token_ids),
                "finish_reason": profile.finish_reason,
                "attention_similarity": _percentiles(
                    attn, (5, 10, 12.5, 25, 50, 75, 90, 95)
                ),
                "hidden_pair": _percentiles(
                    _pairwise_cosines(profile.hidden_states), (50, 90, 95, 99)
                ),
            }
        )

    attention_sims = [
        s
        for profile in profiles
        for s in profile.attention_similarities
        if s is not None
    ]
    within_hidden: list[float] = []
    for profile in profiles:
        within_hidden.extend(_pairwise_cosines(profile.hidden_states))
    cross_hidden: list[float] = []
    for i in range(len(profiles)):
        for j in range(i + 1, len(profiles)):
            a = profiles[i].hidden_states
            b = profiles[j].hidden_states
            if a.shape[0] == 0 or b.shape[0] == 0:
                continue
            an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
            bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
            cross_hidden.extend((an @ bn.T).reshape(-1).tolist())

    attn_p = _percentiles(
        attention_sims, (1, 5, 10, 12.5, 25, 50, 75, 90, 95, 99)
    )
    within_p = _percentiles(within_hidden, (5, 25, 50, 75, 90, 95, 99))
    cross_p = _percentiles(cross_hidden, (50, 90, 95, 99, 99.9))

    suggested_boundary = None
    if attention_sims:
        q = 100.0 / float(target_step_tokens)
        suggested_boundary = float(np.clip(np.percentile(attention_sims, q), 0.0, 1.0))

    suggested_retrieval = None
    if within_hidden:
        within_p95 = float(np.percentile(within_hidden, 95))
        cross_floor = (
            float(np.percentile(cross_hidden, 99)) if cross_hidden else 0.0
        )
        suggested_retrieval = float(
            np.clip(max(within_p95, cross_floor), 0.0, 0.99)
        )

    separation = None
    if within_hidden and cross_hidden:
        separation = float(
            np.percentile(within_hidden, 90)
            - np.percentile(cross_hidden, 99)
        )

    report = {
        "model": engine.model_id,
        "prompts": per_prompt,
        "attention_similarity": {
            "count": len(attention_sims),
            "percentiles": attn_p,
        },
        "hidden_within_prompt": {
            "count": len(within_hidden),
            "percentiles": within_p,
        },
        "hidden_cross_prompt": {
            "count": len(cross_hidden),
            "percentiles": cross_p,
        },
        "separation_p90_within_minus_p99_cross": separation,
        "target_step_tokens": int(target_step_tokens),
        "suggested_boundary_threshold": suggested_boundary,
        "suggested_retrieval_threshold": suggested_retrieval,
        "notes": [
            "boundary_threshold 取相邻注意力相似度的 100/target_step_tokens 分位数，"
            "含义是约每 target_step_tokens 个 token 切一刀。",
            "retrieval_threshold 取同 prompt 隐藏状态成对相似度的 p95，"
            "并确保不低于跨 prompt 的 p99；实际使用应结合任务精度需求调整。",
            "隐藏状态成对相似度来自所有 token，而不是仅步骤首 token，"
            "因此只是阈值量级参考。",
        ],
    }
    return CalibrationResult(report=report)
