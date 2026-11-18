"""逐 token 采样：温度、top-k、top-p、min-p、重复惩罚与 n-gram 禁止。"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import torch

from .config import SamplingConfig

LOGGER = logging.getLogger("stepmem")


def _banned_ngram_tokens(context: Sequence[int], n: int) -> set[int]:
    if n <= 0 or len(context) < n:
        return set()
    if n == 1:
        return set(int(t) for t in context)
    prefix = tuple(int(t) for t in context[-(n - 1) :])
    banned: set[int] = set()
    for i in range(len(context) - n + 1):
        if tuple(context[i : i + n - 1]) == prefix:
            banned.add(int(context[i + n - 1]))
    return banned


def _generator_matches(generator: Optional[torch.Generator], device) -> bool:
    if generator is None:
        return True
    try:
        gen_device = generator.device
        target = torch.device(device)
        if gen_device.type != target.type:
            return False
        if (
            gen_device.index is not None
            and target.index is not None
            and gen_device.index != target.index
        ):
            return False
        return True
    except Exception:  # pragma: no cover
        return False


class Sampler:
    """无状态的采样器；RNG 由调用方传入，便于复现。"""

    def __init__(self, config: SamplingConfig) -> None:
        self.config = config

    def sample(
        self,
        logits: torch.Tensor,
        context_ids: Sequence[int],
        generator: Optional[torch.Generator] = None,
    ) -> int:
        cfg = self.config
        if not torch.is_tensor(logits):
            logits = torch.as_tensor(logits)
        logits = logits.detach().to(torch.float32).reshape(-1).clone()
        if logits.numel() == 0:
            raise ValueError("logits 为空")
        if not _generator_matches(generator, logits.device):
            # 多卡 device_map 等场景下 logits 可能位于与生成器不同的设备。
            generator = None
        original = logits.clone()
        vocab = logits.numel()

        # 1) 重复惩罚（贪心与采样都生效）
        if cfg.repetition_penalty != 1.0:
            ids = sorted({int(t) for t in context_ids if 0 <= int(t) < vocab})
            if ids:
                index = torch.tensor(ids, device=logits.device, dtype=torch.long)
                values = logits.index_select(0, index)
                values = torch.where(
                    values > 0,
                    values / cfg.repetition_penalty,
                    values * cfg.repetition_penalty,
                )
                logits.index_copy_(0, index, values)

        # 2) 禁止重复 n-gram
        if cfg.no_repeat_ngram_size > 0:
            banned = _banned_ngram_tokens(context_ids, cfg.no_repeat_ngram_size)
            banned = {t for t in banned if 0 <= t < vocab}
            if banned and len(banned) < vocab:
                index = torch.tensor(sorted(banned), device=logits.device, dtype=torch.long)
                logits.index_fill_(0, index, float("-inf"))

        greedy = (not cfg.do_sample) or cfg.temperature <= 0.0
        if greedy:
            token = int(torch.argmax(logits).item())
            return token if torch.isfinite(logits[token]) else int(torch.argmax(original).item())

        logits = logits / cfg.temperature

        # 3) top-k
        if 0 < cfg.top_k < vocab:
            kth = torch.topk(logits, cfg.top_k).values[-1]
            logits = torch.where(logits < kth, float("-inf"), logits)

        # 4) top-p（nucleus，至少保留 1 个 token）
        if cfg.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            keep = torch.ones_like(probs, dtype=torch.bool)
            if probs.numel() > 1:
                keep[1:] = cumulative[:-1] < cfg.top_p
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask[sorted_idx[keep]] = True
            logits = logits.masked_fill(~mask, float("-inf"))

        # 5) min-p
        if cfg.min_p > 0.0:
            probs = torch.softmax(logits, dim=-1)
            threshold = cfg.min_p * probs.max()
            logits = logits.masked_fill(probs < threshold, float("-inf"))

        probs = torch.softmax(logits, dim=-1)
        if not torch.isfinite(probs).all() or float(probs.sum()) <= 0.0:
            # 过滤过猛时回退到原始分布的 argmax，保证总能产出 token。
            return int(torch.argmax(original).item())

        token = torch.multinomial(probs, num_samples=1, generator=generator)
        return int(token.item())


def make_generator(seed: Optional[int], device) -> Optional[torch.Generator]:
    """创建与采样设备一致的随机数生成器（仅当显式设置 seed 时）。

    个别后端（如部分 MPS 版本）不支持设备专属 Generator，此时返回 None，
    由调用方退化为全局 RNG（并可用 torch.manual_seed 复现）。
    """
    if seed is None:
        return None
    try:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        return generator
    except Exception:  # pragma: no cover - 取决于后端
        return None
