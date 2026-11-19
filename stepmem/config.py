"""StepMem 配置对象。

所有阈值、存储策略和采样参数都集中在这里，便于通过 CLI / JSON 复现实验。
配置对象是不可变思维下的普通 dataclass：构造时完成类型转换和合法性校验。
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Optional

_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
_DTYPES = {"auto", "float32", "float16", "bfloat16", "float64"}
_STORE_BACKENDS = {"numpy", "faiss"}
_HEAD_REDUCTIONS = {"mean", "first"}


def _build(cls, data):
    """用 dict 构造 dataclass，并拒绝未知字段（避免配置拼写错误被静默忽略）。"""
    if data is None:
        return cls()
    if isinstance(data, cls):
        return data
    if not isinstance(data, dict):
        raise TypeError(f"{cls.__name__} 需要 dict 或实例，收到 {type(data).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"{cls.__name__} 存在未知配置项: {unknown}")
    return cls(**data)


def _as_int_tuple(value, name: str) -> Optional[tuple[int, ...]]:
    if value is None:
        return None
    if isinstance(value, int):
        return (int(value),)
    try:
        items = tuple(int(v) for v in value)
    except TypeError as exc:  # pragma: no cover - 仅在错误输入时触发
        raise TypeError(f"{name} 必须是整数序列") from exc
    if not items:
        raise ValueError(f"{name} 不能为空；若要使用全部层请设为 None")
    return items


def _as_str_tuple(value, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = (value,)
    items = tuple(str(v) for v in value)
    for item in items:
        if not item:
            raise ValueError(f"{name} 不能包含空字符串")
    # 去重且保序
    return tuple(dict.fromkeys(items))


def _as_int_tuple_values(value, name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, int):
        value = (value,)
    try:
        return tuple(int(v) for v in value)
    except TypeError as exc:  # pragma: no cover
        raise TypeError(f"{name} 必须是整数序列") from exc


@dataclass
class SteppingConfig:
    """步骤切分参数。

    仅在写入模式（store.read_only=False）生效；非写入模式不读取注意力、不切分步骤。

    attention_layers:
        - ``None``：对全部层的注意力取平均；
        - ``(-1,)``（默认）：只使用最后一层，与查询键/存储键所在层一致；
        - ``(0, 12, -1)``：对指定层取平均。
    head_reduction:
        多头归约方式，默认 ``mean``（对多头取平均）。
    boundary_threshold:
        相邻 token 注意力分布的余弦相似度 **低于** 该值即切分。默认值基于
        Qwen2.5-0.5B 最后一层注意力的经验分布（约 0.72~0.78），不同模型建议用
        ``stepmem calibrate`` 重新校准。
    min_step_tokens:
        步骤最短 token 数。未达到该长度时忽略边界信号，避免碎片化。
    max_step_tokens:
        步骤最长 token 数，0 表示不限制。超长时强制切分。
    """

    attention_layers: Optional[tuple[int, ...]] = (-1,)
    head_reduction: str = "mean"
    boundary_threshold: float = 0.75
    min_step_tokens: int = 1
    max_step_tokens: int = 0

    def __post_init__(self) -> None:
        self.attention_layers = _as_int_tuple(self.attention_layers, "attention_layers")
        self.head_reduction = str(self.head_reduction).lower()
        self.boundary_threshold = float(self.boundary_threshold)
        self.min_step_tokens = int(self.min_step_tokens)
        self.max_step_tokens = int(self.max_step_tokens)
        self.validate()

    def validate(self) -> None:
        if self.head_reduction not in _HEAD_REDUCTIONS:
            raise ValueError(
                f"head_reduction 必须是 {sorted(_HEAD_REDUCTIONS)} 之一，"
                f"收到 {self.head_reduction!r}"
            )
        if not (0.0 <= self.boundary_threshold <= 1.0):
            raise ValueError("boundary_threshold 必须位于 [0, 1]")
        if self.min_step_tokens < 1:
            raise ValueError("min_step_tokens 必须 >= 1")
        if self.max_step_tokens < 0:
            raise ValueError("max_step_tokens 必须 >= 0")
        if self.max_step_tokens and self.max_step_tokens < self.min_step_tokens:
            raise ValueError("max_step_tokens 不能小于 min_step_tokens")


@dataclass
class RetrievalConfig:
    """检索与记忆注入参数。

    threshold:
        最高余弦相似度必须 **严格大于** 该值才会把记忆写入回复与上下文。
    min_score_margin:
        top-1 与 top-2 的相似度差必须 >= 该值，否则视为歧义并放弃召回。
        0 表示关闭，仅使用 top-1 检索。
    max_inject_tokens:
        单次写入回复/上下文的最大 token 数，超出部分截断。
    prefix / suffix:
        记忆文本的可选装饰（会同时出现在回复与上下文中）。默认均为空，
        即直接注入净化后的原始步骤文本。
    strip_first_token:
        写入前，若记忆文本以当前步骤首 token 的文本开头，则去掉该前缀，
        避免回复重复首 token。
    exclude_last_written:
        刚写入的上一段步骤几乎必然与当前查询键高度相似，默认排除以避免自激循环。
    exclude_injected:
        同一次生成中同一条记忆不重复注入。
    exclude_session:
        是否排除本次生成所属会话的步骤（默认 False，允许同一会话内使用早期步骤）。
    retrieve_on_initial_step:
        首个 token 开启步骤后是否立即检索（仅写入模式）。

    记忆一旦通过阈值与歧义门控，就会同时进入可见回复和 KV cache，不再存在
    “只进上下文、不出现在回复里”的静默注入模式。
    """

    enabled: bool = True
    threshold: float = 0.95
    min_score_margin: float = 0.0
    max_inject_tokens: int = 4096
    prefix: str = ""
    suffix: str = ""
    exclude_last_written: bool = True
    exclude_injected: bool = False
    exclude_session: bool = False
    retrieve_on_initial_step: bool = True
    strip_first_token: bool = True

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.threshold = float(self.threshold)
        self.min_score_margin = float(self.min_score_margin)
        self.max_inject_tokens = int(self.max_inject_tokens)
        self.prefix = str(self.prefix)
        self.suffix = str(self.suffix)
        self.strip_first_token = bool(self.strip_first_token)
        self.validate()

    def validate(self) -> None:
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("retrieval.threshold 必须位于 [0, 1]")
        if not (0.0 <= self.min_score_margin <= 1.0):
            raise ValueError("retrieval.min_score_margin 必须位于 [0, 1]")
        if self.max_inject_tokens < 1:
            raise ValueError("max_inject_tokens 必须 >= 1")


@dataclass
class StoreConfig:
    """向量库与记忆文本参数。

    path 为 None 时仅驻留内存；设置路径后支持跨进程持久化。
    read_only=True 时进入非写入模式：只加载/检索记忆，不切分步骤、逐 token 检索，
    禁止 add/save/clear，适合纯服务阶段。
    sanitize_special_tokens 会移除 tokenizer 注册的特殊 token 字符串
    （EOS/BOS/PAD/UNK 及 chat template token 等），确保保存/注入的记忆干净。
    extra_special_tokens 用于补充自定义特殊字符串。
    autosave/save_every 控制自动落盘频率，显式 ``save()`` 总是可用。
    min_tokens_to_store 过滤过短的步骤；store_final_step 决定生成结束时
    是否也保存最后一个步骤（默认保存）。
    """

    path: Optional[str] = None
    backend: str = "numpy"
    autosave: bool = True
    save_every: int = 32
    min_tokens_to_store: int = 1
    store_final_step: bool = True
    skip_special_tokens: bool = True
    read_only: bool = False
    sanitize_special_tokens: bool = True
    extra_special_tokens: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.path is not None:
            self.path = str(self.path)
        self.backend = str(self.backend).lower()
        self.autosave = bool(self.autosave)
        self.save_every = int(self.save_every)
        self.min_tokens_to_store = int(self.min_tokens_to_store)
        self.store_final_step = bool(self.store_final_step)
        self.skip_special_tokens = bool(self.skip_special_tokens)
        self.read_only = bool(self.read_only)
        self.sanitize_special_tokens = bool(self.sanitize_special_tokens)
        self.extra_special_tokens = _as_str_tuple(
            self.extra_special_tokens, "extra_special_tokens"
        )
        self.validate()

    def validate(self) -> None:
        if self.backend not in _STORE_BACKENDS:
            raise ValueError(
                f"store.backend 必须是 {sorted(_STORE_BACKENDS)} 之一，"
                f"收到 {self.backend!r}"
            )
        if self.save_every < 1:
            raise ValueError("store.save_every 必须 >= 1")
        if self.min_tokens_to_store < 1:
            raise ValueError("store.min_tokens_to_store 必须 >= 1")


@dataclass
class SamplingConfig:
    """逐 token 采样参数。temperature <= 0 或 do_sample=False 时退化为贪心。"""

    max_new_tokens: int = 4096
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    seed: Optional[int] = None
    stop_strings: tuple[str, ...] = ()
    eos_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        self.max_new_tokens = int(self.max_new_tokens)
        self.do_sample = bool(self.do_sample)
        self.temperature = float(self.temperature)
        self.top_p = float(self.top_p)
        self.top_k = int(self.top_k)
        self.min_p = float(self.min_p)
        self.repetition_penalty = float(self.repetition_penalty)
        self.no_repeat_ngram_size = int(self.no_repeat_ngram_size)
        if self.seed is not None:
            self.seed = int(self.seed)
        self.stop_strings = _as_str_tuple(self.stop_strings, "stop_strings")
        self.eos_token_ids = _as_int_tuple_values(self.eos_token_ids, "eos_token_ids")
        self.validate()

    def validate(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens 必须 >= 0")
        if self.temperature < 0:
            raise ValueError("temperature 必须 >= 0")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p 必须位于 (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k 必须 >= 0")
        if not (0.0 <= self.min_p <= 1.0):
            raise ValueError("min_p 必须位于 [0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty 必须 > 0")
        if self.no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size 必须 >= 0")


@dataclass
class StepMemConfig:
    """StepMem 引擎的完整配置。

    model 可以是 Hugging Face Hub 名称或本地目录；也可以直接向
    ``StepMemEngine`` 传入已加载的 model/tokenizer 对象（离线测试常用）。

    attn_implementation:
        ``"auto"`` 表示沿用 HF 默认注意力实现（通常 SDPA，安装 FlashAttention
        时为 FA2）。**非写入模式（store.read_only=True）**会保留该优化；
        **写入模式**需要读取注意力权重切分步骤，因此会强制 eager。
    """

    model: str = ""
    device: str = "auto"
    dtype: str = "auto"
    attn_implementation: str = "auto"
    trust_remote_code: bool = False
    torch_threads: Optional[int] = None
    max_context_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    chat_template: Optional[str] = None
    log_level: str = "WARNING"
    stepping: SteppingConfig = field(default_factory=SteppingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    load_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.model = str(self.model or "")
        self.device = str(self.device)
        self.dtype = str(self.dtype).lower()
        self.attn_implementation = str(self.attn_implementation).lower()
        self.trust_remote_code = bool(self.trust_remote_code)
        if self.torch_threads is not None:
            self.torch_threads = int(self.torch_threads)
        if self.max_context_tokens is not None:
            self.max_context_tokens = int(self.max_context_tokens)
        if self.system_prompt is not None:
            self.system_prompt = str(self.system_prompt)
        if self.chat_template is not None:
            self.chat_template = str(self.chat_template)
        self.log_level = str(self.log_level).upper()
        self.stepping = _build(SteppingConfig, self.stepping)
        self.retrieval = _build(RetrievalConfig, self.retrieval)
        self.store = _build(StoreConfig, self.store)
        self.sampling = _build(SamplingConfig, self.sampling)
        if not isinstance(self.load_kwargs, dict):
            raise TypeError("load_kwargs 必须是 dict")
        self.validate()

    def validate(self) -> None:
        if self.dtype not in _DTYPES:
            raise ValueError(f"dtype 必须是 {sorted(_DTYPES)} 之一，收到 {self.dtype!r}")
        if self.log_level not in _LOG_LEVELS:
            raise ValueError(
                f"log_level 必须是 {sorted(_LOG_LEVELS)} 之一，收到 {self.log_level!r}"
            )
        if self.torch_threads is not None and self.torch_threads < 1:
            raise ValueError("torch_threads 必须 >= 1")
        if self.max_context_tokens is not None and self.max_context_tokens < 1:
            raise ValueError("max_context_tokens 必须 >= 1")
        self.stepping.validate()
        self.retrieval.validate()
        self.store.validate()
        self.sampling.validate()

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StepMemConfig":
        data = dict(data)
        for key, sub in (
            ("stepping", SteppingConfig),
            ("retrieval", RetrievalConfig),
            ("store", StoreConfig),
            ("sampling", SamplingConfig),
        ):
            if key in data:
                data[key] = _build(sub, data[key])
        return _build(cls, data)

    @classmethod
    def from_json_file(cls, path: str) -> "StepMemConfig":
        import json

        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def to_json_file(self, path: str) -> None:
        import json

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)
