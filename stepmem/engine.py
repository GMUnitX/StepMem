"""StepMem 推理引擎。

生成循环以 token 为单位维护 KV cache，并在每个解码步抓取：
* 最后一层（可配置）注意力分布（对多头平均）；
* 最后一层隐藏状态（步骤首 token 作为检索/存储键）。

检测到步骤边界后：先关闭并写入上一段步骤，再以新步骤首 token 的隐藏状态检索，
命中且高于阈值时把记忆原文 token 化后追加进上下文（同时扩展 KV cache），
随后继续生成。检索键在注入前计算，与存储键处于同一表示空间。
"""

from __future__ import annotations

import inspect
import logging
import threading
import time
import uuid
from dataclasses import fields, replace
from typing import Any, Iterator, Mapping, Optional, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import SamplingConfig, StepMemConfig
from .errors import (
    AttentionUnavailableError,
    ContextOverflowError,
    DimensionMismatchError,
    StepMemError,
)
from .sampling import Sampler, make_generator
from .stepping import (
    AttentionSample,
    StepTracker,
    cosine_overlap,
    reduce_attention,
)
from .store import VectorStore
from .types import (
    DoneEvent,
    GenerationResult,
    GenerationStats,
    InjectionEvent,
    InjectionInfo,
    ProfileResult,
    RetrievalEvent,
    RetrievalInfo,
    StepEndEvent,
    StepInfo,
    StepStartEvent,
    TokenEvent,
)
from .utils import (
    IncrementalDecoder,
    find_stop_string,
    hidden_size_of,
    max_positions_of,
    parameter_device,
    resolve_device,
    resolve_dtype,
    safe_decode,
    sanitize_memory_text,
    strip_matched_prefix,
)

LOGGER = logging.getLogger("stepmem")


def _normalize_messages(messages) -> list[dict[str, str]]:
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    elif isinstance(messages, Mapping):
        messages = [messages]
    try:
        items = list(messages)
    except TypeError as exc:
        raise TypeError("messages 必须是 str、dict 或消息列表") from exc
    if not items:
        raise ValueError("messages 不能为空")
    normalized: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise TypeError("每条消息必须是 dict")
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not role:
            raise ValueError("消息缺少合法 role")
        if not isinstance(content, str):
            raise TypeError("StepMem 只支持文本 content")
        normalized.append({"role": role, "content": content})
    return normalized


class StepMemEngine:
    """加载 Hugging Face 因果语言模型并启用 StepMem 长期记忆。"""

    def __init__(
        self,
        config: StepMemConfig | Mapping[str, Any] | str,
        *,
        model=None,
        tokenizer=None,
        store: Optional[VectorStore] = None,
    ) -> None:
        if isinstance(config, StepMemConfig):
            self.config = config
        elif isinstance(config, str):
            self.config = StepMemConfig(model=config)
        else:
            self.config = StepMemConfig.from_dict(dict(config))
        self.config.validate()

        logging.basicConfig(level=getattr(logging, self.config.log_level, logging.WARNING))
        LOGGER.setLevel(getattr(logging, self.config.log_level, logging.WARNING))

        if self.config.torch_threads:
            torch.set_num_threads(int(self.config.torch_threads))

        self._lock = threading.RLock()
        self._device = resolve_device(self.config.device)

        # 写入模式需要注意力切分步骤；非写入模式（只读服务）走逐 token 隐藏态检索，
        # 因此可以保留 SDPA / FlashAttention 等优化。
        self._write_mode = not (
            store.read_only if store is not None else self.config.store.read_only
        )

        # ---------------- tokenizer ----------------
        if tokenizer is None:
            if not self.config.model:
                raise ValueError("未提供 tokenizer/model 时必须指定 config.model")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model,
                trust_remote_code=self.config.trust_remote_code,
                use_fast=True,
            )
        else:
            self.tokenizer = tokenizer
        self._ensure_pad_token()

        # ---------------- model ----------------
        if model is None:
            if not self.config.model:
                raise ValueError("未提供 model 时必须指定 config.model")
            dtype = resolve_dtype(self.config.dtype, self._device)
            load_kwargs = dict(self.config.load_kwargs)
            load_kwargs.setdefault("trust_remote_code", self.config.trust_remote_code)
            requested_attn = self.config.attn_implementation
            if self._write_mode:
                # 注意力探针必须使用 eager attention 才能拿到每层权重。
                load_kwargs["attn_implementation"] = "eager"
                if requested_attn not in {"auto", "eager"}:
                    LOGGER.info(
                        "写入模式需要 eager attention 切分步骤，已覆盖 "
                        "attn_implementation=%r",
                        requested_attn,
                    )
            elif requested_attn not in {"auto", ""}:
                # 非写入模式保留用户指定的 SDPA / FlashAttention。
                load_kwargs["attn_implementation"] = requested_attn
            # 降低 CPU 加载峰值内存（需要 accelerate；transformers 新版本常用默认值）。
            try:
                import accelerate  # noqa: F401

                load_kwargs.setdefault("low_cpu_mem_usage", True)
            except ImportError:
                pass
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model, dtype=dtype, **load_kwargs
                )
            except TypeError as exc:
                if "dtype" not in str(exc).lower():  # pragma: no cover
                    raise
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model, torch_dtype=dtype, **load_kwargs
                )
            if "device_map" not in load_kwargs:
                self.model.to(self._device)
        else:
            self.model = model
            if not self._write_mode:
                current_attn = getattr(
                    self.model.config, "_attn_implementation", "unknown"
                )
                LOGGER.info("非写入模式沿用模型现有注意力实现: %s", current_attn)
                requested = self.config.attn_implementation
                if requested not in {"auto", ""} and requested != current_attn:
                    LOGGER.warning(
                        "传入模型使用 %s，而配置请求 %s；非写入模式不会重建模型注意力实现",
                        current_attn,
                        requested,
                    )

        self.model.eval()
        try:
            self.model.config.use_cache = True
        except Exception:  # pragma: no cover
            pass
        if self._write_mode:
            self._force_eager_attention()

        self._input_device = parameter_device(self.model)
        self._hidden_size = hidden_size_of(self.model)
        model_max = max_positions_of(self.model)
        self._max_positions = int(
            self.config.max_context_tokens or model_max
        )
        if self._max_positions < 2:
            raise ValueError("max_context_tokens 过小")
        self._forward_params = set(inspect.signature(self.model.forward).parameters)
        self._supports_logits_to_keep = "logits_to_keep" in self._forward_params
        self.model_id = (
            self.config.model
            or str(getattr(self.model.config, "_name_or_path", "") or "")
            or self.model.__class__.__name__
        )

        # ---------------- memory ----------------
        if store is not None:
            if int(store.dim) != int(self._hidden_size):
                raise DimensionMismatchError(
                    f"VectorStore 维度 {store.dim} 与模型隐藏维度 {self._hidden_size} 不一致"
                )
            self.store = store
        elif self.config.store.path:
            self.store = VectorStore.open(
                self.config.store.path,
                self._hidden_size,
                config=self.config.store,
                model_id=self.model_id,
            )
        else:
            self.store = VectorStore(
                self._hidden_size, self.config.store, model_id=self.model_id
            )
        self._write_mode = not self.store.read_only

        LOGGER.info(
            "StepMem 就绪: model=%s device=%s hidden=%d max_positions=%d "
            "mode=%s attn=%s store=%s",
            self.model_id,
            self._input_device,
            self._hidden_size,
            self._max_positions,
            "write" if self._write_mode else "read-only",
            getattr(self.model.config, "_attn_implementation", "unknown"),
            self.store,
        )

    # ------------------------------------------------------------------ 属性

    @property
    def write_mode(self) -> bool:
        """True=写入模式（eager + 步骤切分）；False=只读模式（逐 token 检索）。"""
        return self._write_mode

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def device(self):
        return self._input_device

    @property
    def max_positions(self) -> int:
        return self._max_positions

    # ------------------------------------------------------------------ 初始化辅助

    def _ensure_pad_token(self) -> None:
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            eos = getattr(self.tokenizer, "eos_token_id", None)
            if eos is not None:
                try:
                    self.tokenizer.pad_token = (
                        self.tokenizer.convert_ids_to_tokens(eos)
                        if hasattr(self.tokenizer, "convert_ids_to_tokens")
                        else eos
                    )
                except Exception:  # pragma: no cover - 少数 tokenizer 只读
                    LOGGER.debug("无法设置 pad_token，继续使用 batch_size=1")

    def _force_eager_attention(self) -> None:
        current = getattr(self.model.config, "_attn_implementation", None)
        if current == "eager":
            return
        setter = getattr(self.model, "set_attn_implementation", None)
        if callable(setter):
            try:
                setter("eager")
                LOGGER.debug("已通过 set_attn_implementation 切换到 eager attention")
                return
            except Exception as exc:  # pragma: no cover
                LOGGER.debug("set_attn_implementation 失败: %s", exc)
        try:
            self.model.config._attn_implementation = "eager"
        except Exception:  # pragma: no cover
            pass

    # ------------------------------------------------------------------ 前向

    def _forward(
        self,
        input_ids: torch.Tensor,
        past_key_values,
        *,
        capture: bool,
        position_start: int,
        capture_hidden_states: bool = False,
    ):
        n = int(input_ids.shape[-1])
        if n <= 0:
            raise ValueError("input_ids 不能为空")
        total = int(position_start) + n
        device = input_ids.device
        # attention_mask 必须与 KV cache 中实际保存的 key 数对齐；滑动窗口模型
        # 可能裁剪 cache，此时 mask 长度会小于绝对位置总数。
        cache_len = self._cache_seq_len(past_key_values)
        mask_len = total if cache_len is None else cache_len + n
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
            "attention_mask": torch.ones((1, mask_len), dtype=torch.long, device=device),
        }
        if "position_ids" in self._forward_params:
            kwargs["position_ids"] = torch.arange(
                position_start, total, dtype=torch.long, device=device
            ).unsqueeze(0)
        if "cache_position" in self._forward_params:
            kwargs["cache_position"] = torch.arange(
                position_start, total, dtype=torch.long, device=device
            )
        if self._supports_logits_to_keep:
            kwargs["logits_to_keep"] = 1
        if capture:
            kwargs["output_attentions"] = True
            kwargs["output_hidden_states"] = True
        elif capture_hidden_states:
            kwargs["output_hidden_states"] = True
        with torch.inference_mode():
            outputs = self.model(**kwargs)
        if capture:
            if getattr(outputs, "attentions", None) is None:
                raise AttentionUnavailableError(
                    "模型没有返回注意力权重。请确认 transformers 版本支持 "
                    "output_attentions，并使用 eager attention（StepMem 默认已强制）。"
                )
        if capture or capture_hidden_states:
            if getattr(outputs, "hidden_states", None) is None:
                raise AttentionUnavailableError("模型没有返回隐藏状态")
        return outputs

    def _last_hidden_np(self, outputs) -> np.ndarray:
        """取最后一个位置的最后一层隐藏状态（CPU float32）。"""
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise AttentionUnavailableError("模型没有返回隐藏状态")
        with torch.inference_mode():
            return (
                hidden_states[-1][0, -1]
                .detach()
                .to("cpu", torch.float32)
                .numpy()
                .copy()
            )

    @staticmethod
    def _cache_seq_len(past_key_values) -> Optional[int]:
        """返回 cache 中已保存的 key 数；无法判断时返回 None。"""
        if past_key_values is None:
            return 0
        getter = getattr(past_key_values, "get_seq_length", None)
        if callable(getter):
            try:
                return int(getter())
            except Exception:  # pragma: no cover - 特殊 cache 实现
                pass
        try:  # legacy tuple cache: (k, v) 的倒数第二维是序列长度
            return int(past_key_values[0][0].shape[-2])
        except Exception:  # pragma: no cover
            return None

    # ------------------------------------------------------------------ 编码/解码

    def _encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        try:
            ids = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        except TypeError:  # pragma: no cover - 极少数自定义 tokenizer
            ids = self.tokenizer.encode(text)
        if isinstance(ids, torch.Tensor):
            ids = ids.reshape(-1).tolist()
        else:
            ids = list(ids)
            if ids and isinstance(ids[0], (list, tuple)):
                ids = list(ids[0])
        return [int(i) for i in ids]

    def _decode(self, ids: Sequence[int], skip_special_tokens: Optional[bool] = None) -> str:
        if skip_special_tokens is None:
            skip_special_tokens = self.config.store.skip_special_tokens
        return safe_decode(self.tokenizer, ids, skip_special_tokens)

    def clean_memory_text(self, text: str, *, strip: bool = False) -> str:
        """净化将要写入/注入的记忆文本（默认移除特殊 token，不裁剪内容）。"""
        if not isinstance(text, str):
            raise TypeError("记忆文本必须是 str")
        if not self.config.store.sanitize_special_tokens:
            return text.strip() if strip else text
        return sanitize_memory_text(
            text,
            self.tokenizer,
            self.config.store.extra_special_tokens,
            strip=strip,
        )

    def format_chat(self, messages) -> str:
        """把消息列表套用 tokenizer 的 chat template，返回待 token 化的文本。"""
        msgs = _normalize_messages(messages)
        if self.config.system_prompt and (
            not msgs or msgs[0].get("role") != "system"
        ):
            msgs = [{"role": "system", "content": self.config.system_prompt}] + msgs

        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise ValueError(
                "当前 tokenizer 不支持 chat template；请使用 complete() 直接传入文本，"
                "或在 StepMemConfig.chat_template 中提供模板"
            )
        if self.config.chat_template is None and getattr(
            self.tokenizer, "chat_template", None
        ) is None:
            raise ValueError(
                "tokenizer 没有 chat template；请使用 complete() 或配置 chat_template"
            )

        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.config.chat_template:
            kwargs["chat_template"] = self.config.chat_template
        try:
            rendered = self.tokenizer.apply_chat_template(msgs, **kwargs)
        except TypeError:
            if not self.config.chat_template:
                raise
            old = getattr(self.tokenizer, "chat_template", None)
            try:
                self.tokenizer.chat_template = self.config.chat_template
                rendered = self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            finally:
                self.tokenizer.chat_template = old
        if not isinstance(rendered, str) or not rendered:
            raise ValueError("chat template 渲染结果为空")
        return rendered

    # ------------------------------------------------------------------ 公共生成 API

    def _merge_sampling(self, sampling) -> SamplingConfig:
        base = self.config.sampling
        if sampling is None:
            return base
        if isinstance(sampling, SamplingConfig):
            return sampling
        if isinstance(sampling, Mapping):
            allowed = {f.name for f in fields(SamplingConfig)}
            unknown = sorted(set(sampling) - allowed)
            if unknown:
                raise ValueError(f"未知采样参数: {unknown}")
            values = {f.name: getattr(base, f.name) for f in fields(SamplingConfig)}
            values.update(sampling)
            return SamplingConfig(**values)
        raise TypeError("sampling 必须是 SamplingConfig、dict 或 None")

    def _eos_ids(self, sampling: SamplingConfig) -> set[int]:
        values: list[Any] = [
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(getattr(self.model, "generation_config", None), "eos_token_id", None),
            *sampling.eos_token_ids,
        ]
        result: set[int] = set()
        for value in values:
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    try:
                        result.add(int(item))
                    except (TypeError, ValueError):
                        continue
            else:
                try:
                    result.add(int(value))
                except (TypeError, ValueError):
                    continue
        return result

    def stream_chat(self, messages, sampling=None) -> Iterator:
        """按 chat template 生成，流式返回事件（TokenEvent/Step*/Retrieval*/Done）。"""
        prompt_text = self.format_chat(messages)
        prompt_ids = self._encode(prompt_text, add_special_tokens=False)
        if not prompt_ids:
            raise ValueError("chat template 编码结果为空")
        return self._stream_ids(prompt_ids, prompt_text, sampling)

    def chat(self, messages, sampling=None) -> GenerationResult:
        result = None
        for event in self.stream_chat(messages, sampling=sampling):
            if isinstance(event, DoneEvent):
                result = event.result
        if result is None:  # pragma: no cover - 正常路径总有 DoneEvent
            raise StepMemError("生成结束但未收到 DoneEvent")
        return result

    def stream_complete(self, prompt: str, sampling=None) -> Iterator:
        """按原始文本补全（不套 chat template）。"""
        if not isinstance(prompt, str):
            raise TypeError("prompt 必须是 str；对话请使用 chat()/stream_chat()")
        prompt_ids = self._encode(prompt, add_special_tokens=True)
        if not prompt_ids:
            raise ValueError("prompt 编码结果为空")
        return self._stream_ids(prompt_ids, prompt, sampling)

    def complete(self, prompt: str, sampling=None) -> GenerationResult:
        result = None
        for event in self.stream_complete(prompt, sampling=sampling):
            if isinstance(event, DoneEvent):
                result = event.result
        if result is None:  # pragma: no cover
            raise StepMemError("生成结束但未收到 DoneEvent")
        return result

    def train_memory(self, dataset, **kwargs):
        """用 ``{"user": ..., "assistant": ...}`` 数据集训练/扩充步骤记忆。

        对每个样本：填充 user 段（默认套 chat template）→ 模型生成第一个 token →
        读取其最后一层隐藏状态作为查询键 → 存入 assistant 字段原文（净化特殊 token）。
        不更新模型参数。详见 ``stepmem.training.MemoryTrainer``。
        """
        from .training import MemoryTrainer

        return MemoryTrainer(self).train(dataset, **kwargs)

    def _stream_ids(
        self, prompt_ids: list[int], prompt_text: str, sampling
    ) -> Iterator:
        sampling_cfg = self._merge_sampling(sampling)
        if len(prompt_ids) >= self._max_positions:
            raise ContextOverflowError(
                f"prompt 长度 {len(prompt_ids)} 已达到上限 {self._max_positions}"
            )

        def generator() -> Iterator:
            with self._lock:
                session = _GenerationSession(
                    self, prompt_ids, prompt_text, sampling_cfg
                )
                yield from session.events()

        return generator()

    # ------------------------------------------------------------------ 校准

    def profile(
        self,
        prompt: Optional[str] = None,
        messages=None,
        max_new_tokens: int = 64,
        sampling=None,
    ) -> ProfileResult:
        """无记忆、无注入地跑一段生成，采集注意力相似度与隐藏状态。

        用于 ``stepmem calibrate`` 估计合适的边界/检索阈值。需要 eager attention；
        非写入模式默认使用 SDPA/FlashAttention，会拒绝该调用。
        """
        attn_impl = getattr(self.model.config, "_attn_implementation", None)
        if not self.write_mode and attn_impl != "eager":
            raise StepMemError(
                "profile/阈值校准需要注意力权重；请使用写入模式，或显式以 "
                "attn_implementation='eager' 加载模型"
            )
        if messages is not None:
            prompt_text = self.format_chat(messages)
            prompt_ids = self._encode(prompt_text, add_special_tokens=False)
        else:
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("profile 需要非空 prompt 或 messages")
            prompt_text = prompt
            prompt_ids = self._encode(prompt, add_special_tokens=True)
        if not prompt_ids:
            raise ValueError("prompt 编码结果为空")
        sampling_cfg = self._merge_sampling(sampling)
        sampling_cfg = replace(sampling_cfg, max_new_tokens=int(max_new_tokens))
        if len(prompt_ids) >= self._max_positions:
            raise ContextOverflowError(
                f"prompt 长度 {len(prompt_ids)} 已达到上限 {self._max_positions}"
            )
        with self._lock:
            return self._profile_ids(prompt_ids, sampling_cfg)

    def _profile_ids(
        self, prompt_ids: list[int], sampling_cfg: SamplingConfig
    ) -> ProfileResult:
        device = self._input_device
        eos_ids = self._eos_ids(sampling_cfg)
        sampler = Sampler(sampling_cfg)
        generator = make_generator(sampling_cfg.seed, device)
        if sampling_cfg.seed is not None and generator is None:
            torch.manual_seed(int(sampling_cfg.seed))
        ctx = list(prompt_ids)
        next_position = len(prompt_ids)
        outputs = self._forward(
            torch.tensor([prompt_ids], dtype=torch.long, device=device),
            None,
            capture=False,
            position_start=0,
        )
        past = outputs.past_key_values
        logits = outputs.logits[0, -1].float()
        decoder = IncrementalDecoder(self.tokenizer, skip_special_tokens=True)
        token_ids: list[int] = []
        hidden_states: list[np.ndarray] = []
        similarities: list[Optional[float]] = []
        previous_sample: Optional[AttentionSample] = None
        finish_reason = "max_new_tokens"

        for _ in range(sampling_cfg.max_new_tokens):
            if next_position >= self._max_positions:
                finish_reason = "context_limit"
                break
            with torch.inference_mode():
                token = sampler.sample(logits, ctx, generator)
            if token in eos_ids:
                finish_reason = "eos"
                break
            ctx.append(token)
            token_ids.append(token)
            outputs = self._forward(
                torch.tensor([[token]], dtype=torch.long, device=device),
                past,
                capture=True,
                position_start=next_position,
            )
            past = outputs.past_key_values
            position = next_position
            next_position += 1
            logits = outputs.logits[0, -1].float()
            with torch.inference_mode():
                vector = reduce_attention(
                    outputs.attentions,
                    layers=self.config.stepping.attention_layers,
                    head_reduction=self.config.stepping.head_reduction,
                )
                hidden = (
                    outputs.hidden_states[-1][0, -1]
                    .detach()
                    .to("cpu", torch.float32)
                    .numpy()
                    .copy()
                )
            sample = AttentionSample(
                vector=vector, start_position=position + 1 - vector.shape[0]
            )
            similarities.append(cosine_overlap(previous_sample, sample))
            previous_sample = sample
            hidden_states.append(np.asarray(hidden, dtype=np.float32))
            decoder.update(token_ids)
            if sampling_cfg.stop_strings and find_stop_string(
                decoder.text, sampling_cfg.stop_strings
            ):
                finish_reason = "stop_string"
                break

        text = decoder.text
        stop = (
            find_stop_string(text, sampling_cfg.stop_strings)
            if sampling_cfg.stop_strings
            else None
        )
        if stop is not None:
            text = text[: stop[0]]
        token_texts = [self._decode([t], skip_special_tokens=True) for t in token_ids]
        hidden_array = (
            np.stack(hidden_states, axis=0)
            if hidden_states
            else np.empty((0, self._hidden_size), dtype=np.float32)
        )
        return ProfileResult(
            text=text,
            token_ids=token_ids,
            token_texts=token_texts,
            attention_similarities=similarities,
            hidden_states=hidden_array,
            finish_reason=finish_reason,
        )

    # ------------------------------------------------------------------ 生命周期

    def save_memory(self) -> None:
        if self.store.read_only:
            LOGGER.debug("记忆库处于只读模式，跳过保存")
            return
        if self.config.store.path:
            self.store.save()
            LOGGER.info("记忆已保存: %s", self.config.store.path)

    def close(self) -> None:
        try:
            self.save_memory()
        except Exception as exc:  # 关闭阶段不掩盖原始异常
            LOGGER.error("保存 StepMem 记忆失败: %s", exc)

    def __enter__(self) -> "StepMemEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class _GenerationSession:
    """一次生成过程中所有可变状态的容器。"""

    def __init__(
        self,
        engine: StepMemEngine,
        prompt_ids: list[int],
        prompt_text: str,
        sampling: SamplingConfig,
    ) -> None:
        self.engine = engine
        self.cfg = engine.config
        self.sampling = sampling
        self.prompt_ids = list(prompt_ids)
        self.prompt_text = prompt_text
        self.session_id = uuid.uuid4().hex[:12]

        self.ctx_ids: list[int] = list(prompt_ids)
        self.next_position = len(prompt_ids)
        self.past_key_values = None
        self.logits: Optional[torch.Tensor] = None

        self.sampler = Sampler(sampling)
        self.generator = make_generator(sampling.seed, engine.device)
        if sampling.seed is not None and self.generator is None:
            torch.manual_seed(int(sampling.seed))
        self.eos_ids = engine._eos_ids(sampling)
        self.write_mode = engine.write_mode
        self.tracker = StepTracker(self.cfg.stepping) if self.write_mode else None
        self.decoder = IncrementalDecoder(
            engine.tokenizer, skip_special_tokens=True
        )

        self.steps: list[StepInfo] = []
        self.retrievals: list[RetrievalInfo] = []
        self.injections: list[InjectionInfo] = []
        self.injected_record_ids: set[int] = set()
        self.last_written_id: Optional[int] = None
        self.generated_ids: list[int] = []
        self.generated_text = ""
        self._emitted_len = 0
        self._text_before_last_token = ""
        self._final_trim_chars = 0
        self._memory_spans: list[tuple[int, str]] = []
        self._final_output_text: Optional[str] = None
        self.finish_reason: Optional[str] = None
        self._start_time = time.perf_counter()

    # ------------------------------------------------------------------ 主循环

    def events(self) -> Iterator:
        device = self.engine.device
        outputs = self.engine._forward(
            torch.tensor([self.prompt_ids], dtype=torch.long, device=device),
            None,
            capture=False,
            position_start=0,
        )
        self.past_key_values = outputs.past_key_values
        self.logits = outputs.logits[0, -1].float()

        if self.write_mode:
            yield from self._run_write_loop(device)
        else:
            yield from self._run_read_only_loop(device)

        if self.finish_reason is None:
            self.finish_reason = "max_new_tokens"

        if self.write_mode:
            assert self.tracker is not None
            closed = self.tracker.finish()
            if closed is not None:
                yield from self._close_step(closed, is_final=True)

        yield DoneEvent(result=self._build_result())

    def _run_write_loop(self, device) -> Iterator:
        """写入模式：eager 注意力 + 步骤切分 + 步骤边界检索。"""
        assert self.tracker is not None
        for _ in range(self.sampling.max_new_tokens):
            if self.next_position >= self.engine.max_positions:
                self.finish_reason = "context_limit"
                break

            with torch.inference_mode():
                token = self.sampler.sample(
                    self.logits, self.ctx_ids, self.generator
                )

            if token in self.eos_ids:
                self.finish_reason = "eos"
                break

            # 采样出的 token 先进入上下文，再前向以获得隐藏状态与注意力。
            self.ctx_ids.append(token)
            self.generated_ids.append(token)
            outputs = self.engine._forward(
                torch.tensor([[token]], dtype=torch.long, device=device),
                self.past_key_values,
                capture=True,
                position_start=self.next_position,
            )
            self.past_key_values = outputs.past_key_values
            position = self.next_position
            self.next_position += 1
            self.logits = outputs.logits[0, -1].float()

            self._text_before_last_token = self.decoder.text
            self.decoder.update(self.generated_ids)
            self.generated_text = self.decoder.text

            # 停止串检查发生在步骤观测之前：触发停止的 token 不进入步骤记忆。
            stop = (
                find_stop_string(self.generated_text, self.sampling.stop_strings)
                if self.sampling.stop_strings
                else None
            )
            if stop is not None:
                stop_index = stop[0]
                self.finish_reason = "stop_string"
                # 只裁剪已经属于已观测步骤的那部分停止串前缀。
                self._final_trim_chars = max(
                    0, len(self._text_before_last_token) - stop_index
                )
                visible = self._compose_output_text(
                    self.generated_text[:stop_index]
                )
                self._final_output_text = visible
                delta = self._emit_delta(visible)
                yield TokenEvent(
                    token_id=token,
                    text=delta,
                    context_position=position,
                    step_index=max(self.tracker.current_index, 0),
                    generated_text=visible,
                )
                break

            with torch.inference_mode():
                vector = reduce_attention(
                    outputs.attentions,
                    layers=self.cfg.stepping.attention_layers,
                    head_reduction=self.cfg.stepping.head_reduction,
                )
            hidden = self.engine._last_hidden_np(outputs)
            sample = AttentionSample(
                vector=vector, start_position=position + 1 - vector.shape[0]
            )
            decision = self.tracker.observe(position, token, hidden, sample)

            composed = self._compose_output_text()
            delta = self._emit_delta(composed)
            yield TokenEvent(
                token_id=token,
                text=delta,
                context_position=position,
                step_index=decision.step_index,
                generated_text=composed,
            )

            if decision.closed is not None:
                yield from self._close_step(decision.closed, is_final=False)

            if decision.started:
                step = StepInfo(
                    index=decision.step_index,
                    start_position=position,
                    end_position=position,
                    token_ids=[],
                    text="",
                    boundary_similarity=decision.similarity,
                    boundary_reason=decision.reason,
                )
                self.steps.append(step)
                yield StepStartEvent(
                    step_index=decision.step_index,
                    context_position=position,
                    reason=decision.reason,
                    similarity=decision.similarity,
                )
                should_retrieve = self.cfg.retrieval.enabled and not (
                    decision.reason == "initial"
                    and not self.cfg.retrieval.retrieve_on_initial_step
                )
                if should_retrieve:
                    yield from self._retrieve_and_inject(
                        step, hidden, position, token
                    )

    def _run_read_only_loop(self, device) -> Iterator:
        """非写入模式：不切分步骤，每个生成 token 的隐藏态都检索。"""
        for _ in range(self.sampling.max_new_tokens):
            if self.next_position >= self.engine.max_positions:
                self.finish_reason = "context_limit"
                break

            with torch.inference_mode():
                token = self.sampler.sample(
                    self.logits, self.ctx_ids, self.generator
                )

            if token in self.eos_ids:
                self.finish_reason = "eos"
                break

            self.ctx_ids.append(token)
            self.generated_ids.append(token)
            outputs = self.engine._forward(
                torch.tensor([[token]], dtype=torch.long, device=device),
                self.past_key_values,
                capture=False,
                capture_hidden_states=True,
                position_start=self.next_position,
            )
            self.past_key_values = outputs.past_key_values
            position = self.next_position
            self.next_position += 1
            self.logits = outputs.logits[0, -1].float()

            self._text_before_last_token = self.decoder.text
            self.decoder.update(self.generated_ids)
            self.generated_text = self.decoder.text

            stop = (
                find_stop_string(self.generated_text, self.sampling.stop_strings)
                if self.sampling.stop_strings
                else None
            )
            if stop is not None:
                self.finish_reason = "stop_string"
                visible = self._compose_output_text(
                    self.generated_text[: stop[0]]
                )
                self._final_output_text = visible
                delta = self._emit_delta(visible)
                yield TokenEvent(
                    token_id=token,
                    text=delta,
                    context_position=position,
                    step_index=-1,
                    generated_text=visible,
                )
                break

            hidden = self.engine._last_hidden_np(outputs)
            composed = self._compose_output_text()
            delta = self._emit_delta(composed)
            yield TokenEvent(
                token_id=token,
                text=delta,
                context_position=position,
                step_index=-1,
                generated_text=composed,
            )

            if self.cfg.retrieval.enabled:
                yield from self._retrieve_and_inject(
                    None,
                    hidden,
                    position,
                    token,
                    emit_rejected=False,
                )

    # ------------------------------------------------------------------ 步骤收尾

    def _emit_delta(self, visible_text: str) -> str:
        if len(visible_text) < self._emitted_len:
            # decode 修正导致文本回缩：重发修正后的内容。
            self._emitted_len = 0
        delta = visible_text[self._emitted_len :]
        self._emitted_len = len(visible_text)
        return delta

    def _compose_output_text(self, generated_text: Optional[str] = None) -> str:
        """把模型生成文本与注入的记忆片段按插入位置组合成可见输出。"""
        gen = self.generated_text if generated_text is None else generated_text
        if not self._memory_spans:
            return gen
        parts: list[str] = []
        last = 0
        for position, text in self._memory_spans:
            position = max(0, min(int(position), len(gen)))
            if position < last:
                position = last
            parts.append(gen[last:position])
            parts.append(text)
            last = position
        parts.append(gen[last:])
        return "".join(parts)

    def _insert_memory_span(self, text: str) -> str:
        """把注入文本插入输出流，并同步流式游标。"""
        if text:
            self._memory_spans.append((len(self.generated_text), text))
        composed = self._compose_output_text()
        # 注入文本由 InjectionEvent 单独发出，因此这里推进游标，避免下一个
        # TokenEvent 重复发送。
        self._emitted_len = len(composed)
        return composed

    def _should_store(self, closed, text: str) -> bool:
        if self.engine.store.read_only:
            return False
        if not text.strip():
            return False
        if len(closed.token_ids) < self.cfg.store.min_tokens_to_store:
            return False
        return True

    def _close_step(self, closed, *, is_final: bool) -> Iterator:
        step = self.steps[closed.index]
        text = self.engine._decode(
            closed.token_ids,
            skip_special_tokens=self.cfg.store.skip_special_tokens,
        )
        if is_final and self._final_trim_chars > 0:
            trim = self._final_trim_chars
            text = text[:-trim] if trim < len(text) else ""
        text = self.engine.clean_memory_text(text)
        step.text = text
        step.token_ids = list(closed.token_ids)
        step.end_position = int(closed.end_position)

        stored = False
        record_id: Optional[int] = None
        # 单次回复的首个步骤（initial）只用于首个 token 的检索键，即使处于
        # 写入模式也不写入记忆库，避免把回复开头本身变成可召回的“记忆”。
        if closed.start_reason != "initial" and self._should_store(closed, text):
            if self._is_final_step_allowed(is_final):
                try:
                    record = self.engine.store.add(
                        closed.key,
                        text,
                        session_id=self.session_id,
                        step_index=closed.index,
                        token_count=len(closed.token_ids),
                        metadata={
                            "model": self.engine.model_id,
                            "start_reason": closed.start_reason,
                            "start_similarity": closed.start_similarity,
                        },
                    )
                except Exception as exc:
                    LOGGER.error("写入步骤记忆失败: %s", exc)
                else:
                    stored = True
                    record_id = record.id
                    self.last_written_id = record.id
                    step.stored = True
                    step.record_id = record.id
        yield StepEndEvent(
            step_index=closed.index,
            text=text,
            token_count=len(closed.token_ids),
            stored=stored,
            record_id=record_id,
            is_final=is_final,
        )

    def _is_final_step_allowed(self, is_final: bool) -> bool:
        return bool(self.cfg.store.store_final_step) if is_final else True

    # ------------------------------------------------------------------ 检索 + 注入

    def _reject_retrieval(
        self,
        step: Optional[StepInfo],
        similarity: Optional[float],
        reason: str,
        record=None,
        *,
        position: Optional[int] = None,
        emit: bool = True,
    ) -> Iterator:
        record_id = record.id if record is not None else None
        record_text = record.text if record is not None else None
        if position is None and step is not None:
            position = step.start_position
        info = RetrievalInfo(
            similarity=similarity,
            threshold=self.cfg.retrieval.threshold,
            accepted=False,
            reason=reason,
            record_id=record_id,
            record_text=record_text,
            position=position,
        )
        self.retrievals.append(info)
        if step is not None:
            step.retrieval = info
        if emit:
            yield RetrievalEvent(
                step_index=step.index if step is not None else -1,
                similarity=similarity,
                accepted=False,
                reason=reason,
                record_id=record_id,
                record_text=record_text,
                position=position,
            )

    def _append_memory_tokens(self, ids: list[int], position: int) -> None:
        outputs = self.engine._forward(
            torch.tensor([ids], dtype=torch.long, device=self.engine.device),
            self.past_key_values,
            capture=False,
            position_start=position,
        )
        self.past_key_values = outputs.past_key_values
        self.logits = outputs.logits[0, -1].float()
        self.ctx_ids.extend(ids)
        self.next_position += len(ids)

    def _retrieve_and_inject(
        self,
        step: Optional[StepInfo],
        query_key: np.ndarray,
        position: int,
        start_token_id: int,
        *,
        emit_rejected: bool = True,
    ) -> Iterator:
        cfg = self.cfg.retrieval
        step_index = step.index if step is not None else -1
        excluded: set[int] = set()
        # exclude_injected 对两种模式一致生效（可选功能，默认关闭）：
        # True 时同一条记录在一次生成中最多注入一次；False 时允许重复注入。
        # 非写入模式逐 token 检索，关闭去重时同一记忆可能在连续 token 上
        # 重复注入，由调用方按需权衡。
        if cfg.exclude_injected:
            excluded.update(self.injected_record_ids)
        if cfg.exclude_last_written and self.last_written_id is not None:
            excluded.add(self.last_written_id)

        top_k = 2 if cfg.min_score_margin > 0.0 else 1
        try:
            hits = self.engine.store.search(
                query_key,
                top_k=top_k,
                exclude_ids=excluded,
                exclude_session=self.session_id if cfg.exclude_session else None,
            )
        except Exception as exc:
            LOGGER.error("StepMem 检索失败: %s", exc)
            yield from self._reject_retrieval(
                step, None, "store_error", position=position, emit=emit_rejected
            )
            return

        if not hits:
            yield from self._reject_retrieval(
                step, None, "no_candidates", position=position, emit=emit_rejected
            )
            return

        best = hits[0]
        similarity = float(best.similarity)
        if not (similarity > cfg.threshold):
            yield from self._reject_retrieval(
                step,
                similarity,
                "below_threshold",
                best.record,
                position=position,
                emit=emit_rejected,
            )
            return

        if cfg.min_score_margin > 0.0 and len(hits) > 1:
            margin = similarity - float(hits[1].similarity)
            if margin < cfg.min_score_margin:
                LOGGER.debug(
                    "记忆召回歧义: top1=%.4f top2=%.4f margin=%.4f < %.4f",
                    similarity,
                    float(hits[1].similarity),
                    margin,
                    cfg.min_score_margin,
                )
                yield from self._reject_retrieval(
                    step,
                    similarity,
                    "ambiguous",
                    best.record,
                    position=position,
                    emit=emit_rejected,
                )
                return

        memory_text = self.engine.clean_memory_text(best.record.text)
        if not memory_text.strip():
            yield from self._reject_retrieval(
                step,
                similarity,
                "empty_text",
                best.record,
                position=position,
                emit=emit_rejected,
            )
            return

        # 统一语义：命中记忆始终同时写入可见回复与 KV cache。
        first_piece = self.engine._decode(
            [start_token_id], skip_special_tokens=True
        )
        visible_memory = (
            strip_matched_prefix(memory_text, first_piece)
            if cfg.strip_first_token
            else memory_text
        )
        if not visible_memory.strip():
            yield from self._reject_retrieval(
                step,
                similarity,
                "nothing_to_inject",
                best.record,
                position=position,
                emit=emit_rejected,
            )
            return

        inject_text = f"{cfg.prefix}{visible_memory}{cfg.suffix}"
        try:
            inject_ids = self.engine._encode(inject_text, add_special_tokens=False)
        except Exception as exc:
            LOGGER.error("记忆文本 token 化失败: %s", exc)
            yield from self._reject_retrieval(
                step,
                similarity,
                "encode_error",
                best.record,
                position=position,
                emit=emit_rejected,
            )
            return
        if not inject_ids:
            yield from self._reject_retrieval(
                step,
                similarity,
                "empty_text",
                best.record,
                position=position,
                emit=emit_rejected,
            )
            return
        if len(inject_ids) > cfg.max_inject_tokens:
            inject_ids = inject_ids[: cfg.max_inject_tokens]
        # 以实际进入 KV cache 的 token 解码结果作为可见文本，保证回复与上下文一致。
        inject_text = self.engine._decode(
            inject_ids, skip_special_tokens=self.cfg.store.skip_special_tokens
        )
        if not inject_text.strip():
            yield from self._reject_retrieval(
                step,
                similarity,
                "nothing_to_inject",
                best.record,
                position=position,
                emit=emit_rejected,
            )
            return
        if self.next_position + len(inject_ids) > self.engine.max_positions:
            yield from self._reject_retrieval(
                step,
                similarity,
                "context_limit",
                best.record,
                position=position,
                emit=emit_rejected,
            )
            return

        start = self.next_position
        self._append_memory_tokens(inject_ids, start)
        self.injected_record_ids.add(best.record.id)
        self._insert_memory_span(inject_text)

        self.injections.append(
            InjectionInfo(
                step_index=step_index,
                record_id=best.record.id,
                similarity=similarity,
                text=inject_text,
                token_count=len(inject_ids),
                position=start,
            )
        )
        info = RetrievalInfo(
            similarity=similarity,
            threshold=cfg.threshold,
            accepted=True,
            reason="injected",
            record_id=best.record.id,
            record_text=best.record.text,
            injected_tokens=len(inject_ids),
            position=position,
        )
        self.retrievals.append(info)
        if step is not None:
            step.retrieval = info
        yield RetrievalEvent(
            step_index=step_index,
            similarity=similarity,
            accepted=True,
            reason="injected",
            record_id=best.record.id,
            record_text=best.record.text,
            position=position,
        )
        yield InjectionEvent(
            step_index=step_index,
            record_id=best.record.id,
            similarity=similarity,
            text=inject_text,
            token_count=len(inject_ids),
            position=start,
        )

    # ------------------------------------------------------------------ 结果

    def _build_result(self) -> GenerationResult:
        elapsed = time.perf_counter() - self._start_time
        if self._final_output_text is not None:
            text = self._final_output_text
        else:
            generated_view = self.generated_text
            if self.finish_reason == "stop_string" and self.sampling.stop_strings:
                stop = find_stop_string(generated_view, self.sampling.stop_strings)
                if stop is not None:
                    generated_view = generated_view[: stop[0]]
            text = self._compose_output_text(generated_view)
        generated = len(self.generated_ids)
        stats = GenerationStats(
            prompt_tokens=len(self.prompt_ids),
            generated_tokens=generated,
            steps=len(self.steps),
            boundaries=sum(
                1
                for step in self.steps
                if step.boundary_reason in {"threshold", "max_tokens"}
            ),
            retrievals=len(self.retrievals),
            injections=len(self.injections),
            injected_tokens=sum(i.token_count for i in self.injections),
            stored_steps=sum(1 for step in self.steps if step.stored),
            elapsed_seconds=elapsed,
            tokens_per_second=(generated / elapsed) if elapsed > 0 else 0.0,
            finish_reason=self.finish_reason or "max_new_tokens",
        )
        return GenerationResult(
            text=text,
            prompt_text=self.prompt_text,
            context_text=self.engine._decode(
                self.ctx_ids, skip_special_tokens=False
            ),
            steps=list(self.steps),
            injections=list(self.injections),
            stats=stats,
            session_id=self.session_id,
            retrievals=list(self.retrievals),
        )
