"""记忆训练：把外部蒸馏数据写入 StepMem 步骤记忆库。

训练不更新模型参数。每个样本执行：

1. 用 user 字段填充对话 prompt（默认套 chat template）；
2. 模型前向 prompt，采样出第一个 token；
3. 前向该 token，读取最后一层隐藏状态作为**查询键**；
4. 将 assistant 字段的原文（净化特殊 token 后）作为**记忆内容**写入向量库。

这样在推理时，相似 user 意图的首 token 隐藏状态可以检索到蒸馏答案；
命中后答案会直接写入可见回复，并跳过该段内容的逐 token 生成。
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Optional

import numpy as np
import torch

from .errors import StoreError
from .sampling import Sampler, make_generator
from .utils import safe_decode, to_jsonable

if TYPE_CHECKING:  # pragma: no cover
    from .engine import StepMemEngine

LOGGER = logging.getLogger("stepmem")

ProgressCallback = Callable[[int, int, int], None]  # done, total, stored


@dataclass
class MemoryTrainingError:
    """单条训练样本的错误（不包含完整正文，避免日志泄露数据）。"""

    index: int
    message: str


@dataclass
class MemoryTrainingStats:
    total: int
    processed: int
    stored: int
    skipped_empty: int = 0
    skipped_duplicate: int = 0
    skipped_eos: int = 0
    failed: int = 0
    tokens_stored: int = 0
    elapsed_seconds: float = 0.0
    dry_run: bool = False


@dataclass
class MemoryTrainingResult:
    session_id: str
    stats: MemoryTrainingStats
    errors: list[MemoryTrainingError]

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def validate_memory_record(record: Any, index: int = 0) -> tuple[str, str, dict[str, Any]]:
    """校验一条 ``{"user": ..., "assistant": ...}`` 记录。

    同时接受 ``prompt`` / ``response`` 别名；``metadata`` 可选且必须是 dict。
    """
    if not isinstance(record, Mapping):
        raise ValueError("记录必须是 JSON object")
    user = record.get("user", record.get("prompt"))
    assistant = record.get("assistant", record.get("response"))
    if not isinstance(user, str) or not user.strip():
        raise ValueError("user/prompt 必须是非空字符串")
    if not isinstance(assistant, str) or not assistant.strip():
        raise ValueError("assistant/response 必须是非空字符串")
    metadata = record.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("metadata 必须是 JSON object")
    return user, assistant, dict(metadata)


def _parse_dataset_text(text: str, suffix: str = "") -> list[dict[str, Any]]:
    text = text.lstrip("\ufeff")
    stripped = text.strip()
    if not stripped:
        return []

    # JSON 文件或 JSON 数组
    if suffix == ".json" or stripped.startswith("["):
        data = json.loads(text)
        if isinstance(data, Mapping):
            data = [data]
        if not isinstance(data, list):
            raise ValueError("JSON 数据集必须是对象或对象数组")
        return [dict(item) if isinstance(item, Mapping) else item for item in data]

    # 单行/多行 JSON object（JSONL 只有一行时也会命中这里）
    if stripped.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, Mapping):
            return [dict(data)]
        if isinstance(data, list):
            return [dict(item) if isinstance(item, Mapping) else item for item in data]
        if data is not None:
            raise ValueError("JSON 数据集必须是对象或对象数组")

    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"第 {line_no} 行不是合法 JSON: {exc}") from exc
        if not isinstance(item, Mapping):
            raise ValueError(f"第 {line_no} 行必须是 JSON object")
        records.append(dict(item))
    return records


def load_memory_dataset(source) -> list[dict[str, Any]]:
    """加载记忆训练数据集。

    支持：
    * ``.jsonl`` / ``.ndjson`` / 普通文本：每行一个 JSON object；
    * ``.json``：对象数组或单个对象；
    * 字符串/Path 路径，或直接传入 JSON 文本；
    * 单个 dict，或 dict 的可迭代对象。
    """
    if isinstance(source, Mapping):
        return [dict(source)]
    if isinstance(source, (str, Path)):
        raw = str(source)
        path = Path(raw)
        if path.exists():
            if not path.is_file():
                raise ValueError(f"数据集路径不是文件: {path}")
            text = path.read_text(encoding="utf-8-sig")
            return _parse_dataset_text(text, path.suffix.lower())
        # 不是现存文件时，允许直接传入 JSON/JSONL 文本。
        if "\n" in raw or raw.lstrip().startswith(("{", "[")):
            return _parse_dataset_text(raw)
        raise FileNotFoundError(f"找不到数据集文件: {raw}")
    if isinstance(source, Iterable):
        records = []
        for item in source:
            if not isinstance(item, Mapping):
                raise ValueError("数据集元素必须是 JSON object")
            records.append(dict(item))
        return records
    raise TypeError("dataset 必须是路径、JSON 文本、dict 或 dict 可迭代对象")


class MemoryTrainer:
    """把（用户段，助手内容）蒸馏数据写入 StepMem 记忆库。"""

    def __init__(self, engine: "StepMemEngine") -> None:
        self.engine = engine

    def _build_prompt(
        self,
        user: str,
        *,
        use_chat_template: bool,
        system_prompt: Optional[str],
    ) -> tuple[str, list[int]]:
        engine = self.engine
        if use_chat_template:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user})
            prompt_text = engine.format_chat(messages)
            prompt_ids = engine._encode(prompt_text, add_special_tokens=False)
        else:
            prompt_text = (
                f"{system_prompt}\n\n{user}" if system_prompt else user
            )
            prompt_ids = engine._encode(prompt_text, add_special_tokens=True)
        return prompt_text, prompt_ids

    def train(
        self,
        dataset,
        *,
        sampling=None,
        use_chat_template: bool = True,
        system_prompt: Optional[str] = None,
        skip_eos: bool = False,
        max_samples: Optional[int] = None,
        dedupe: bool = True,
        strict: bool = False,
        dry_run: bool = False,
        save: bool = True,
        progress: Optional[ProgressCallback] = None,
    ) -> MemoryTrainingResult:
        """训练记忆库。

        sampling:
            首 token 的采样配置；默认贪心（do_sample=False），保证训练键稳定。
        use_chat_template:
            True 时用 tokenizer 的 chat template 填充 user 段；False 时直接用
            user 原文（可配合 system_prompt 拼接）。
        skip_eos:
            首 token 为 EOS 时是否跳过。默认 False（仍读取其隐藏状态并保存），
            因为“模型想停止”本身也是可检索的意图。
        max_samples:
            只处理前 N 条，便于小规模试验。
        dedupe:
            同一次训练中按 (user, 净化后 assistant) 去重。
        strict:
            True 时遇到任何样本错误立即抛出；False 时记录到 errors 并继续。
        dry_run:
            只解析/校验/清洗数据集，不加载额外计算、不写库。
        save:
            训练结束后若配置了 store.path 且非只读，立即落盘。
        progress:
            回调 ``(done, total, stored)``，便于 CLI 显示进度。
        """
        engine = self.engine
        store = engine.store
        if store.read_only:
            raise StoreError("记忆库处于只读模式，无法进行记忆训练")

        records = load_memory_dataset(dataset)
        if max_samples is not None:
            max_samples = int(max_samples)
            if max_samples < 1:
                raise ValueError("max_samples 必须 >= 1")
            records = records[:max_samples]
        total = len(records)
        session_id = f"train-{uuid.uuid4().hex[:8]}"
        started = time.perf_counter()
        errors: list[MemoryTrainingError] = []

        if sampling is None:
            sampling_cfg = replace(
                engine.config.sampling,
                do_sample=False,
                max_new_tokens=1,
                stop_strings=(),
            )
        else:
            sampling_cfg = engine._merge_sampling(sampling)
            sampling_cfg = replace(
                sampling_cfg, max_new_tokens=1, stop_strings=()
            )
        sampler = Sampler(sampling_cfg)
        generator = make_generator(sampling_cfg.seed, engine.device)
        if sampling_cfg.seed is not None and generator is None:
            torch.manual_seed(int(sampling_cfg.seed))
        eos_ids = engine._eos_ids(sampling_cfg)

        stored = 0
        skipped_empty = 0
        skipped_duplicate = 0
        skipped_eos = 0
        failed = 0
        tokens_stored = 0
        seen: set[tuple[str, str]] = set()
        device = engine.device

        for index, raw_record in enumerate(records):
            def notify() -> None:
                if progress is not None:
                    progress(index + 1, total, stored)

            try:
                user, assistant, extra_metadata = validate_memory_record(
                    raw_record, index
                )
            except Exception as exc:
                failed += 1
                errors.append(MemoryTrainingError(index, str(exc)))
                if strict:
                    raise ValueError(f"第 {index + 1} 条训练样本无效: {exc}") from exc
                notify()
                continue

            assistant_text = engine.clean_memory_text(assistant, strip=True)
            if not assistant_text:
                skipped_empty += 1
                notify()
                continue

            dedupe_key = (user, assistant_text)
            if dedupe and dedupe_key in seen:
                skipped_duplicate += 1
                notify()
                continue

            try:
                _, prompt_ids = self._build_prompt(
                    user,
                    use_chat_template=use_chat_template,
                    system_prompt=system_prompt,
                )
                if not prompt_ids:
                    raise ValueError("user prompt 编码结果为空")
                if len(prompt_ids) >= engine.max_positions:
                    raise ValueError(
                        f"user prompt 长度 {len(prompt_ids)} 超过上限 "
                        f"{engine.max_positions}"
                    )

                if dry_run:
                    if dedupe:
                        seen.add(dedupe_key)
                    notify()
                    continue

                # 1) 填充 user 段并前向
                prefill = engine._forward(
                    torch.tensor([prompt_ids], dtype=torch.long, device=device),
                    None,
                    capture=False,
                    position_start=0,
                )
                with torch.inference_mode():
                    first_token = sampler.sample(
                        prefill.logits[0, -1].float(), prompt_ids, generator
                    )

                if skip_eos and first_token in eos_ids:
                    skipped_eos += 1
                    notify()
                    continue

                # 2) 生成第一个 token（只前向一次），读取最后一层隐藏状态
                outputs = engine._forward(
                    torch.tensor([[first_token]], dtype=torch.long, device=device),
                    prefill.past_key_values,
                    capture=False,
                    capture_hidden_states=True,
                    position_start=len(prompt_ids),
                )
                key = engine._last_hidden_np(outputs)
                if not np.all(np.isfinite(key)):
                    raise ValueError("首 token 隐藏状态包含 NaN/Inf")

                try:
                    token_count = len(
                        engine._encode(assistant_text, add_special_tokens=False)
                    )
                except Exception:  # pragma: no cover - 极少见 tokenizer 问题
                    token_count = 0

                metadata = dict(extra_metadata)
                metadata.update(
                    {
                        "source": "memory_train",
                        "train_session": session_id,
                        "dataset_index": index,
                        "chat_template": bool(use_chat_template),
                        "user_preview": user[:200],
                        "assistant_chars": len(assistant_text),
                        "first_token_id": int(first_token),
                        "first_token_text": safe_decode(
                            engine.tokenizer, [first_token], skip_special_tokens=True
                        ),
                    }
                )

                # 3) assistant 原文作为记忆内容，首 token 隐藏状态作为键
                engine.store.add(
                    key,
                    assistant_text,
                    session_id=session_id,
                    step_index=index,
                    token_count=token_count,
                    metadata=metadata,
                    autosave=False,  # 批量训练结束后统一落盘
                )
                stored += 1
                tokens_stored += token_count
                if dedupe:
                    seen.add(dedupe_key)
            except Exception as exc:
                failed += 1
                errors.append(MemoryTrainingError(index, str(exc)))
                if strict:
                    raise RuntimeError(
                        f"第 {index + 1} 条训练样本失败: {exc}"
                    ) from exc
                LOGGER.warning("记忆训练样本 #%d 失败: %s", index + 1, exc)
            notify()

        if not dry_run and save and not store.read_only and engine.config.store.path:
            store.save()

        elapsed = time.perf_counter() - started
        processed = total - skipped_empty - skipped_duplicate
        stats = MemoryTrainingStats(
            total=total,
            processed=processed,
            stored=stored,
            skipped_empty=skipped_empty,
            skipped_duplicate=skipped_duplicate,
            skipped_eos=skipped_eos,
            failed=failed,
            tokens_stored=tokens_stored,
            elapsed_seconds=elapsed,
            dry_run=bool(dry_run),
        )
        return MemoryTrainingResult(
            session_id=session_id, stats=stats, errors=errors
        )
