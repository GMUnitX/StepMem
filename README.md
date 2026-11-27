# StepMem · 步骤记忆拓展，用生成过程内部状态驱动可扩展步骤记忆的 Hugging Face 推理引擎（CPU / GPU）

**优化算力成本，构建高效智能**

StepMem 在模型自回归生成过程中直接使用模型内部状态做记忆检索，不引入额外嵌入模型，也不计算用户输入侧的表示。引擎有两种运行模式：

- **写入模式**（`store.read_only=False`）：使用 eager attention，读取相邻 token 注意力分布切分步骤；步骤完成时，以步骤首 token 的最后一层隐藏状态为键写入向量库。
- **非写入模式**（`store.read_only=True`）：不切分步骤，恢复 SDPA / FlashAttention 等注意力优化；**每个生成 token 的隐藏状态都发起一次检索**。

无论哪种模式，命中并高于阈值、通过歧义门控的记忆都会同时写入可见回复和 KV cache。**命中的记忆内容也不必再由模型逐 token 重新生成，从而节省解码算力。同时，由于记忆内容由模型自身隐藏状态命中且步骤来源于模型历史生成，上下文割裂问题得到解决。另外，由于未命中部分及查询键的生成仍然存在随机性且召回的是可复用步骤，模型并未丧失创造力。**

因此，**在通用对话甚至 Agent 场景下，都能召回可用步骤。生产环境用少量但覆盖广的数据预热数据库即能节省大量算力（虽然检索存在消耗，但总体上只要命中率达到一定程度，检索开销低于召回部分的自回归生成，且生产环境下用户基数保证了大量重复问题的存在）。同时可作为热拔插补丁，临时修复模型的不良回答时无需微调模型而仅需增加数据库条目固定回答。**

单次回复的首个步骤会被强制切分用于首次检索，但即使处于写入模式也不会写入记忆库。

在此基础上，项目还提供：

- **只读服务**：训练一次，长期只读加载，逐 token 检索；
- **记忆训练**：把大模型蒸馏出的问答/算式/事实数据批量写入；
- **可靠性控制**：相似度阈值、top1-top2 歧义门控、特殊 token 净化；
- **可扩展记忆库**：步骤记忆 + 外部蒸馏记忆共用同一检索空间。

模型参数始终不被修改。

---

## 0.StepMem 算力成本对比实验报告

**[StepMem 算力成本对比实验报告](https://gmunitx.com/index.php/2026/11/27/stepmem-%e7%ae%97%e5%8a%9b%e6%88%90%e6%9c%ac%e5%af%b9%e6%af%94%e5%ae%9e%e9%aa%8c%e6%8a%a5%e5%91%8a/)**

---

## 1. 核心能力

- 纯 Hugging Face 模型加载，**CPU / CUDA / MPS** 均可运行
- **双模式运行**：写入模式 eager + 步骤切分；非写入模式 SDPA/FA + 逐 token 检索
- 逐 token KV cache 解码；注入记忆时只对新 token 做前向并扩展 cache
- 注意力权重在设备端完成多头归约，仅回传小向量；默认取最后一层
- 步骤边界由相邻 token 注意力分布余弦相似度判定，支持绝对位置对齐
- 查询键 / 存储键均取生成 token 的最后一层隐藏状态，天然同空间
- **可见记忆注入**：命中记忆直接进入回复与 KV cache，跳过该段内容的逐 token 生成
- **首个步骤不入库**：首次检索仍执行，但不会把回复开头写成记忆
- **只读记忆模式**：加载已训练记忆，禁止写入/清空/保存，适合纯服务
- **记忆训练**：`{"user": ..., "assistant": ...}` JSONL/JSON 蒸馏数据批量入库
- **歧义门控**：top-1 与 top-2 相似度差不足时放弃召回，降低误命中
- **特殊 token 净化**：写入、训练、注入前移除 EOS/BOS/PAD/UNK/chat 模板标记
- 流式事件（token / 步骤 / 检索 / 注入 / 完成）
- 向量库默认 NumPy + BLAS，可选 FAISS，支持原子化磁盘持久化
- 重复注入可配置：去重关闭时同记录可重复注入，开启后单次生成内只注入一次
- 内置阈值校准、CLI、示例、48 项离线测试

---

## 2. 安装

```bash
# 在项目根目录
pip install -e .

# 可选：FAISS 向量索引
pip install -e ".[faiss]"

# 可选：开发依赖
pip install -e ".[dev]"
```

依赖：`torch>=2.1`、`transformers>=4.48`、`numpy>=1.24`。

---

## 3. 快速开始

### 3.1 普通生成 + 步骤记忆

```python
from stepmem import (
    StepMemEngine, StepMemConfig, SteppingConfig, RetrievalConfig,
    StoreConfig, SamplingConfig,
)

cfg = StepMemConfig(
    model="Qwen2.5-0.5B-Instruct",       # 本地目录或 Hub 名称
    device="auto",                        # cuda > mps > cpu
    dtype="auto",                         # CPU→float32, GPU→bf16/fp16
    stepping=SteppingConfig(
        attention_layers=(-1,),           # 最后一层注意力（默认）
        boundary_threshold=0.75,
        min_step_tokens=2,
    ),
    retrieval=RetrievalConfig(
        enabled=True,
        threshold=0.95,
        max_inject_tokens=4096,
    ),
    store=StoreConfig(path="./stepmem_memory"),  # 可写 → 写入模式
    sampling=SamplingConfig(max_new_tokens=128, do_sample=False),
)
# 写入模式：eager 注意力 + 步骤切分；单次回复的首个步骤不入库。

with StepMemEngine(cfg) as engine:
    result = engine.chat("用三句话解释什么是光合作用。")
    print(result.text)
    for item in result.injections:
        print(f"注入记忆 #{item.record_id} sim={item.similarity:.3f}: {item.text!r}")
    print(result.stats)
```

### 3.2 训练蒸馏记忆

```python
with StepMemEngine(cfg) as engine:
    result = engine.train_memory(
        "examples/data/distill_qa.jsonl",   # 或 list[dict]
        sampling={"do_sample": False},      # 首 token 贪心，键更稳定
        dedupe=True,
    )
    print(result.stats)
```

### 3.3 只读服务（记忆进入回复）

```python
serve_cfg = StepMemConfig(
    model="Qwen2.5-0.5B-Instruct",
    device="auto",
    attn_implementation="auto",  # 非写入模式保留 SDPA / FlashAttention
    stepping=SteppingConfig(attention_layers=(-1,), boundary_threshold=0.75),
    retrieval=RetrievalConfig(
        enabled=True,
        threshold=0.95,
        min_score_margin=0.01,
        max_inject_tokens=4096,
    ),
    store=StoreConfig(path="./stepmem_distill_memory", read_only=True),
    sampling=SamplingConfig(max_new_tokens=32, do_sample=False),
)

# read_only=True → 非写入模式：不切分步骤，每个生成 token 都检索记忆。
with StepMemEngine(serve_cfg) as engine:
    result = engine.chat("9 的平方是多少？")
    print(result.text)
    for item in result.injections:
        print(item.record_id, item.similarity, item.text)
```

### 3.4 流式事件

```python
from stepmem import TokenEvent, RetrievalEvent, InjectionEvent, DoneEvent

for event in engine.stream_chat("介绍一下长期记忆。"):
    if isinstance(event, TokenEvent):
        print(event.text, end="", flush=True)
    elif isinstance(event, RetrievalEvent) and event.accepted:
        print(f"\n[注入记忆 #{event.record_id} sim={event.similarity:.3f}]\n")
    elif isinstance(event, InjectionEvent):
        print(event.text, end="", flush=True)  # 记忆文本是可见回复的一部分
    elif isinstance(event, DoneEvent):
        print("\n", event.result.stats)
```

> **重要**：`InjectionEvent.text` 属于可见输出，流式消费方必须打印/追加它；`DoneEvent.result.text` 已包含完整输出。

### 3.5 CLI

```bash
# 交互式对话（默认持久化到 ~/.stepmem/memory）
stepmem chat --model Qwen2.5-0.5B-Instruct --show-steps

# 单次生成
stepmem generate --model Qwen2.5-0.5B-Instruct \
  --prompt "用一句话解释熵。" --max-new-tokens 64

# 训练记忆库
stepmem train --model Qwen2.5-0.5B-Instruct \
  --dataset examples/data/distill_qa.jsonl \
  --memory ./stepmem_distill_memory

# 只读服务（非写入模式）
stepmem chat --model Qwen2.5-0.5B-Instruct \
  --memory ./stepmem_distill_memory \
  --read-only-memory --attn-implementation sdpa \
  --retrieval-threshold 0.95 --retrieval-margin 0.01

# 阈值校准
stepmem calibrate --model Qwen2.5-0.5B-Instruct \
  --prompt "解释什么是机器学习。" \
  --prompt "写一个 Python 反转链表函数。" \
  --target-step-tokens 8
```

`chat` 内置命令：`/help`、`/reset`、`/memories`、`/save`、`/clear`、`/stats`、`/exit`。

---

## 4. 架构与模块映射

| 机制 | 实现位置 |
|---|---|
| 逐 token 取多头注意力并平均 | `stepmem/stepping.py::reduce_attention` |
| 相邻 token 注意力余弦 + 阈值切分 | `cosine_overlap`、`StepTracker.observe` |
| 首个 token 开启首步骤并触发首次检索 | `StepTracker` 的 `initial` 分支 + `retrieve_on_initial_step` |
| 查询键 = 步骤首 token 最后一层隐藏态 | `stepmem/engine.py::_GenerationSession.events` |
| 存储键 = 同上 | `_close_step` → `VectorStore.add` |
| 余弦 top-1 检索 | `VectorStore.search` |
| 阈值过滤 | `_retrieve_and_inject` |
| 命中原文写入回复与 KV cache | 同一函数：写入回复 + token 化 + 前向追加 KV cache |
| 存储步骤原文，不做摘要 | `StoreConfig.skip_special_tokens` + `VectorStore.add` |
| 运行时流程 | `_GenerationSession.events` |

双模式与扩展能力：

| 能力 | 实现 |
|---|---|
| 写入模式（eager + 步骤切分） | `_GenerationSession._run_write_loop` + `StepTracker` |
| 非写入逐 token 检索 | `_GenerationSession._run_read_only_loop` |
| 首个步骤不入库 | `_close_step` 对 `start_reason=="initial"` 跳过写库 |
| 只读记忆 | `StoreConfig.read_only` / `VectorStore.read_only` |
| 记忆训练 | `stepmem/training.py::MemoryTrainer` |
| 可见记忆注入 | `_retrieve_and_inject` + `_compose_output_text` |
| 歧义门控 | `RetrievalConfig.min_score_margin` |
| 特殊 token 净化 | `utils.sanitize_memory_text` / `engine.clean_memory_text` |

## 5. 工作流程

一次生成中，StepMem 维护：

- `ctx_ids`：完整上下文（prompt + 已生成 token + 已注入 token）
- `next_position`：下一个 token 的绝对位置
- `past_key_values`：HF 动态 KV cache
- `memory_spans`：注入文本及其在可见输出中的插入位置
- `StepTracker`：仅写入模式使用，记录步骤起点、起点隐藏状态、累积 token

### 5.1 写入模式（`store.read_only=False`）

1. 从当前 logits 采样下一个 token；EOS 则结束。
2. token 进入上下文并前向，取注意力分布与最后一层隐藏状态。
3. 与前一 token 的注意力向量求余弦（按绝对位置交集）。
4. 低于阈值且步骤长度足够：
   - 关闭上一段步骤，解码并净化原文；
   - **若该步骤以 `initial` 开始（单次回复的首个步骤），不写入记忆库**；
   - 当前 token 开启新步骤，以其隐藏状态检索 top-1；
   - 通过阈值与歧义门控后，把命中记忆净化、去除重复首 token 前缀；
   - 记忆文本 **直接写入可见回复**，同时 token 化、前向追加进 KV cache；
   - 用记忆最后一个 token 的 logits 继续生成。
5. 未到边界则并入当前步骤。
6. 生成结束按配置保存最后一步（首个步骤同样不入库）。

写入模式必须读取注意力权重，因此会强制 `attn_implementation="eager"`。

### 5.2 非写入模式（`store.read_only=True`）

1. 模型按 `attn_implementation` 加载（默认 auto → SDPA，安装 FlashAttention 时可用 FA2）。
2. **不读取注意力、不切分步骤**；每次前向只返回隐藏状态。
3. 每个生成 token 的最后一层隐藏状态都作为查询键检索记忆。
4. 命中通过阈值与歧义门控后，记忆写入可见回复与 KV cache，然后继续生成。
5. 是否同一条记录只注入一次由 `exclude_injected` 控制（默认 False，允许重复注入）。

非写入模式的特性：

- `result.steps == []`，`stats.steps == 0`；
- 每次检索记录在 `result.retrievals`，`RetrievalInfo.position` 是 token 位置；
- 被拒绝的逐 token 检索只写入 `result.retrievals`，不会在事件流里逐条刷屏；
- `InjectionInfo.step_index` / `RetrievalEvent.step_index` / `TokenEvent.step_index` 为 `-1`；
- `boundary_threshold`、`min_step_tokens`、`retrieve_on_initial_step` 等步骤参数不生效；
- 不能训练记忆（只读）。

### 5.3 注入位置与展示

- 写入模式注入发生在新步骤首 token 之后、下一个 token 之前；非写入模式注入发生在当前 token 之后、下一个 token 之前。
- 默认注入净化后的原始记忆文本；需要分隔/标签时配置 `prefix`/`suffix`，它们会同时出现在回复与上下文中。
- `strip_first_token=True` 会去掉与当前首 token 重复的前缀，避免回复开头重复。
- `max_inject_tokens` 限制单次注入长度。
- 注入文本不进入“模型生成的步骤文本”，因此不会被再次写回记忆库，避免层层复写。
- `exclude_injected` 默认关闭（允许重复注入）；开启后同一条记录在一次生成中最多注入一次，两种模式行为一致。

**首次检索**：第一个生成 token 的隐藏状态要等其前向完成，所以首次检索发生在第一个token 之后、后续内容展开之前。该步骤不会入库。

**多轮对话**：每次 `chat()` 调用都会重新 prefill 完整消息历史；跨轮复用依赖向量库，而不是跨轮 KV cache。

---

## 6. 记忆训练（蒸馏数据）

### 6.1 数据格式

JSONL（每行一个对象）：

```jsonl
{"user":"9 的平方是多少？","assistant":"9 的平方是 81。"}
{"user":"17 加 25 等于多少？","assistant":"17 加 25 等于 42。"}
```

也支持：

- JSON 数组或单个 JSON 对象（`.json`）；
- 字段别名 `prompt` / `response`；
- 可选 `metadata` 对象（会合并到记录 metadata）。

示例数据见 [examples/data/distill_qa.jsonl](examples/data/distill_qa.jsonl)。

### 6.2 训练做了什么

对每条样本：

1. 用 user 字段填充对话 prompt（默认套 chat template）；
2. 模型前向 prompt，采样**第一个 token**；
3. 前向该 token，取最后一层隐藏状态作为**查询键**；
4. 将 assistant 字段原文（去除特殊 token、裁剪首尾空白）作为**记忆内容**写入向量库。

不更新模型参数。每条样本约需一次 prompt prefill + 一次单 token 前向。

### 6.3 API

```python
result = engine.train_memory(
    "data.jsonl",
    sampling={"do_sample": False},  # 默认贪心；建议用于稳定键
    use_chat_template=True,          # False 时直接用 user 原文
    system_prompt=None,              # 覆盖训练时的 system prompt
    skip_eos=False,                  # 首 token 为 EOS 是否跳过
    max_samples=None,
    dedupe=True,                     # 同一次训练内去重
    strict=False,                    # True 时遇错立即抛出
    dry_run=False,
    save=True,                       # 有 store.path 时训练后立即落盘
    progress=None,                   # (done, total, stored)
)

print(result.stats)
```

- 只读记忆库会直接抛出 `StoreError`，不会训练。
- `stats.processed = total - skipped_empty - skipped_duplicate`；非 dry-run 时`processed = stored + skipped_eos + failed`。
- assistant 清洗后为空（例如只含特殊 token）计入 `skipped_empty`；结构非法（空字符串、非字符串、metadata 非对象）计入 `failed`。
- 同一次训练内的重复 `(user, assistant)` 计入 `skipped_duplicate`。
- 训练记录带 `metadata.source="memory_train"`、`train_session`、`dataset_index`、`user_preview`、`first_token_id` 等字段，便于审计。

### 6.4 CLI

```bash
stepmem train --model Qwen2.5-0.5B-Instruct \
  --dataset examples/data/distill_qa.jsonl \
  --memory ./stepmem_distill_memory \
  --progress-every 10

# 只校验数据集，不加载模型
stepmem train --dataset examples/data/distill_qa.jsonl --dry-run
```

### 6.5 关键一致性建议

训练键来自“user prompt 的首个生成 token”。推理时如果使用相同的 chat 模板、system prompt和相近采样策略，键空间最匹配。用于事实问答/算式时建议：

- 训练与推理都用 `do_sample=False`，或都使用较低温度；
- 训练和推理使用相同的 system prompt / chat template；
- 多轮历史会使 prompt 与训练时的单轮 prompt 不同，命中率会下降。

---

## 7. 只读记忆与记忆注入

### 7.1 只读（非写入）模式

```python
store=StoreConfig(path="./stepmem_distill_memory", read_only=True)
```

- 允许加载、检索、注入；
- 禁止 `add()`、`clear()`、`save()`（抛出 `StoreError`）；
- 不切分步骤，**每个生成 token 的隐藏状态都检索**；
- 保留 SDPA / FlashAttention（`attn_implementation="auto"` 或显式指定）；
- `result.steps == []`、`stats.steps == 0`，检索轨迹在 `result.retrievals`；
- 引擎生成过程中不会尝试写库，`stats.stored_steps == 0`；
- `engine.close()` 不会改写磁盘；
- CLI：`--read-only-memory`、`--attn-implementation sdpa`。

适合“训练一次，长期只读服务”的部署方式。该模式不能训练记忆。

### 7.2 记忆注入语义

只要命中满足：

```text
top1_similarity > threshold
top1_similarity - top2_similarity >= min_score_margin（若启用）
```

引擎就会：

1. 净化记忆文本；
2. 去掉与当前 token 输出重复的前缀（`strip_first_token`，写入模式下当前 token 是步骤首 token）；
3. 加上可选 `prefix`/`suffix`；
4. 截断到 `max_inject_tokens`；
5. **写入可见回复**；
6. **前向追加进 KV cache**；
7. 从记忆最后 token 的 logits 继续生成。

因此：

- 回复里出现记忆内容，是因为引擎主动注入了它，而不是模型碰巧复述；
- 记忆内容不需要逐 token 重新生成，节省了这部分解码算力；
- 后续内容仍由模型基于记忆上下文生成，保留泛化/推理能力。

非写入模式每个 token 都会检索；默认允许同一条记忆重复注入，开启 `exclude_injected` 可让同一条记忆在一次生成中只注入一次。

### 7.3 可靠性建议

- 事实问答 / 算式：提高 `threshold`（如 0.9~0.95），启用`min_score_margin`（如 0.01~0.02）；
- 记忆内容较长时提高 `max_inject_tokens`，或训练时只存简洁答案；
- 若回复中记忆与后续生成重复，可检查 `strip_first_token`，或让训练答案与模型首 token对齐。

### 7.4 歧义门控

`min_score_margin > 0` 时检索 top-2，并要求：

```text
top1_similarity - top2_similarity >= min_score_margin
```

否则返回 `reason="ambiguous"`，放弃本轮召回（记忆不会进入回复）。只有 1 条候选时不触发歧义拒绝。它能在记忆库存在近重复/冲突条目时避免“随机选一个”的事实风险。

---

## 8. 特殊 token 净化

记忆写入、训练、注入前都会调用 `engine.clean_memory_text()`：

- 移除 `tokenizer.all_special_tokens`（EOS/BOS/PAD/UNK/chat 模板标记等）；
- 移除 `eos_token`、`bos_token`、`pad_token`、`unk_token`、`sep_token`、`cls_token`、`mask_token`；
- 移除 `additional_special_tokens`；
- 支持 `StoreConfig.extra_special_tokens` 自定义补充；
- 长串优先替换，避免子串残留。

默认不裁剪正文首尾空白（保持注入文本的分隔语义）；记忆训练写入 assistant 字段时会额外 `strip()`。如果清洗后为空则跳过该条记忆。

配置：

```python
store=StoreConfig(
    sanitize_special_tokens=True,
    extra_special_tokens=("<|end|>", "<custom>"),
)
```

关闭：`StoreConfig(sanitize_special_tokens=False)` 或 CLI `--no-sanitize-memory`。旧版记忆库中的脏文本也会在注入前被动清洗。

---

## 9. 配置与默认值

### 9.1 `SteppingConfig`

| 字段 | 默认值 | 说明 |
|---|---|---|
| `attention_layers` | `(-1,)` | `None`=全部层平均；`(-1,)`=最后一层；可 `(0,12,-1)` |
| `head_reduction` | `"mean"` | 多头平均 |
| `boundary_threshold` | `0.75` | 相邻注意力余弦小于该值切分 |
| `min_step_tokens` | `1` | 步骤最短长度 |
| `max_step_tokens` | `0` | 0 不限制；>0 超长强制切分 |

默认只取最后一层：深层注意力更接近下一 token 的生成意图，且与隐藏状态同层，便于查询键与存储键保持一致。**这些参数只在写入模式生效**；非写入模式不切分步骤。

### 9.2 `RetrievalConfig`

| 字段 | 默认值 | 说明 |
|---|---|---|
| `enabled` | `True` | 是否检索 |
| `threshold` | `0.95` | top-1 相似度大于则注入 |
| `min_score_margin` | `0.0` | top1-top2 差距下限；0 关闭 |
| `max_inject_tokens` | `4096` | 单次写入回复/上下文的最大 token 数 |
| `prefix` / `suffix` | `""` | 记忆装饰，会出现在回复与上下文中 |
| `strip_first_token` | `True` | 注入前去掉重复 token 前缀 |
| `exclude_last_written` | `True` | 排除刚写入步骤，防自激（写入模式） |
| `exclude_injected` | `False` | 可选去重：True 时同一条记录单次生成内只注入一次 |
| `exclude_session` | `False` | 是否排除本会话步骤 |
| `retrieve_on_initial_step` | `True` | 首步骤是否立即检索（仅写入模式） |

刚写入的步骤与当前查询键几乎必然同源，top-1 可能命中自身；默认排除，可关闭。

非写入模式没有步骤，每个生成 token 都检索；`retrieve_on_initial_step` 和边界参数不会改变这一点。

### 9.3 `StoreConfig`

| 字段 | 默认值 | 说明 |
|---|---|---|
| `path` | `None` | 非空则持久化 |
| `backend` | `"numpy"` | `numpy` / `faiss` |
| `autosave` / `save_every` | `True` / `32` | 自动落盘 |
| `min_tokens_to_store` | `1` | 过短步骤不写入 |
| `store_final_step` | `True` | 保存最后一步 |
| `skip_special_tokens` | `True` | 解码时跳过特殊 token ID |
| `read_only` | `False` | 非写入模式：逐 token 检索、无步骤、禁止写库 |
| `sanitize_special_tokens` | `True` | 移除特殊 token 字符串 |
| `extra_special_tokens` | `()` | 自定义额外特殊串 |

### 9.4 `SamplingConfig`

| 字段 | 默认值 |
|---|---|
| `max_new_tokens` | `4096` |
| `do_sample` | `True` |
| `temperature` | `0.7` |
| `top_p` | `0.9` |
| `top_k` | `0`（不限制） |
| `min_p` | `0.0` |
| `repetition_penalty` | `1.0` |
| `no_repeat_ngram_size` | `0` |
| `seed` | `None` |
| `stop_strings` | `()` |
| `eos_token_ids` | `()`（自动并入 tokenizer / generation_config） |

`temperature <= 0` 或 `do_sample=False` 退化为贪心。

### 9.5 `StepMemConfig`

| 字段 | 默认值 |
|---|---|
| `model` | `""` |
| `device` | `"auto"` |
| `dtype` | `"auto"` |
| `attn_implementation` | `"auto"`（写入模式强制 eager；非写入模式保留 SDPA/FA） |
| `torch_threads` | `None` |
| `max_context_tokens` | `None` |
| `system_prompt` | `None` |
| `chat_template` | `None` |
| `log_level` | `"WARNING"` |
| `load_kwargs` | `{}` |

配置可 JSON 读写：`StepMemConfig.from_json_file()` / `.to_json_file()`。

`store.read_only` 决定引擎模式，必须在构造 `StepMemEngine` 前确定。

---

## 10. 阈值校准

```bash
stepmem calibrate \
  --model Qwen2.5-0.5B-Instruct \
  --prompt "解释什么是机器学习。" \
  --prompt "写一个 Python 反转链表函数。" \
  --target-step-tokens 8
```

输出：

- 相邻 token 注意力余弦分位数 → 建议 `boundary_threshold = 100/target_step_tokens` 分位数；
- 同 prompt / 跨 prompt 隐藏态成对余弦分位数 → 建议 `retrieval_threshold` 取同 prompt p95且不低于跨 prompt p99；
- 分离度 `within p90 - cross p99`。

校准只跑无记忆生成（`engine.profile`），不写向量库。它需要注意力权重，因此`stepmem calibrate` 会强制写入模式（eager）；非写入模式不切分步骤，不需要边界阈值。

> 隐藏态建议值使用所有 token 的成对相似度，只是量级参考；生产环境应结合命中准确率/召回率手工微调。

---

## 11. 向量库与持久化

```text
stepmem_memory/
├── manifest.json     # format/version/dim/count/model_id/时间戳
├── records.jsonl     # MemoryRecord（id/text/session/step/token_count/metadata）
└── keys.npy          # (N, dim) float32，已 L2 归一化
```

- 写入时键归一化，检索用内积，等价余弦；
- 保存采用临时文件 + `os.replace`，manifest 最后替换作为提交标记；
- 加载校验维度、条数、零向量/NaN，并重新归一化；
- 模型不匹配输出 warning；
- 无删除操作，id 单调递增；`clear()` 清空后需显式 `save()`；
- 只读模式下 `save/clear/add` 均拒绝。

```python
from stepmem import VectorStore, StoreConfig

store = VectorStore(896)
record = store.add(key, "步骤原文", session_id="s1", step_index=3, token_count=12)
hits = store.search(query, top_k=3, exclude_ids={record.id})
store.save("./stepmem_memory")
loaded = VectorStore.load(
    "./stepmem_memory", dim=896, config=StoreConfig(read_only=True)
)
```

---

## 12. 性能设计

- **KV cache**：每步只输入新 token，缓存复用，不重跑前缀。
- **双模式注意力**：写入模式才启用 eager；非写入模式保留 SDPA / FlashAttention。
- **注入只追加**：记忆 token 一次性前向，KV 直接追加；记忆文本不再逐 token 生成。
- **`logits_to_keep=1`**：支持时只保留最后位置 logits，降低大词表 prefill 开销。
- **非写入模式只取隐藏态**：不采集注意力，兼容 SDPA/FA2。
- **训练只取隐藏态**：`capture_hidden_states=True`，不采集注意力，减少训练开销。
- **批量训练不逐条落盘**：训练时关闭逐条 autosave，结束后统一保存。
- **注意力归约**：写入模式 GPU 上完成多头求均值，逐层回传小向量；默认只取最后一层。
- **向量检索**：预分配 NumPy 矩阵 + BLAS；可选 FAISS。
- **`torch.inference_mode()`**：前向与采样无 autograd 开销。
- **低内存加载**：检测到 accelerate 时默认 `low_cpu_mem_usage=True`。
- **单流低延迟**：每个序列有独立状态/KV cache，不做 batch；多请求串行复用模型。

### 注意力探针的代价（仅写入模式）

写入模式读取每层注意力以切分步骤，必须使用 **eager attention**，因此会关闭FlashAttention/SDPA。

非写入模式（`read_only=True`）不读取注意力，模型按 `attn_implementation` 正常加载，保留 SDPA/FlashAttention，仅返回隐藏状态用于逐 token 检索。若只做记忆检索服务，应使用非写入模式。

---

## 13. 工程决策

1. **双模式**：`read_only` 决定写入模式（eager + 步骤）或非写入模式（SDPA/FA + 逐 token）。
2. **层选择**：写入模式默认最后一层注意力；提供 `attention_layers`。
3. **注入时序**：步骤首 token / 当前 token 之后注入（已生成 token 无法回滚）。
4. **可见注入**：所有通过门控的记忆都进入回复与 KV cache；不做静默上下文注入。
5. **首个步骤不入库**：首次检索仍执行，但初始步骤即使写入模式也不写库。
6. **去重前缀**：默认去掉与当前 token 重复的记忆前缀，避免回复重复。
7. **自激防护**：写入模式默认排除刚写入步骤；同记录去重是可选功能（`exclude_injected`，默认关闭，两种模式一致生效）。
8. **步骤最短/最长**：`min_step_tokens` / `max_step_tokens`（仅写入模式）。
9. **最终步骤**：默认保存，`store_final_step=False` 可关闭（首个步骤始终不保存）。
10. **停止串**：触发停止的 token 不进入步骤记忆；只约束模型生成文本，不约束注入文本。
11. **上下文上限**：注入前检查；生成到上限以 `context_limit` 正常结束。
12. **批量**：不支持 batch>1，避免序列状态串扰。
13. **模型切换**：不同模型写出的记忆目录会 warning，不静默混用。
14. **训练不改权重**：只增加检索记忆，模型能力与参数保持原样。
15. **注入不写库**：记忆文本进入回复/上下文，但不作为步骤记忆回写。

---

## 14. 测试

```bash
python -m unittest discover -s tests -v
```

覆盖（共 48 项）：

- 向量库：写入/检索/阈值/排除/归一化/持久化/自动保存/维度校验/只读
- 注意力归约、层解析、绝对位置交集余弦
- 步骤状态机：初始步骤、阈值边界、最短/最长步骤、结束收尾、首步骤不入库
- 采样：贪心、top-k、top-p、min-p、重复惩罚、n-gram、可复现随机
- 配置：嵌套 dict、未知字段拒绝、JSON round-trip、参数校验
- 数据集：JSONL/JSON/别名/错误行
- 特殊 token 净化、首 token 前缀匹配
- 记忆训练：写入、空内容、重复过滤、元数据、只读拒绝
- 非写入模式：保留 SDPA/FA、不切分步骤、逐 token 检索、无步骤事件
- 只读检索：不写库、不保存、磁盘计数不变
- 可见注入：`InjectionEvent`、输出组合、统计
- 歧义门控：top1/top2 同分时拒绝召回
- 离线端到端：tiny random GPT-2 + 字符级 tokenizer（无需联网）

本机验证（CPU，Qwen2.5-0.5B-Instruct，float32）：

- 普通步骤切分、记忆写入、跨会话 top-1 命中（相同 prompt 首 token → 相似度 1.0）与注入正常；
- 用 `{"user":"9 的平方是多少？","assistant":"9 的平方是 81。<|im_end|>"}` 训练，入库文本自动净化为 `9 的平方是 81。`；
- 只读加载后查询同一问题：命中相似度 1.0，回复为 `9 的平方是 81。`，只生成 1 个 token、注入 7 个 token，记忆库条数不变；非写入模式不产生步骤。

---

## 15. 项目结构

```text
stepmem/
├── __init__.py        # 公共 API
├── config.py          # 配置与校验
├── types.py           # 结果、记录、流式事件
├── stepping.py        # 注意力归约、余弦、步骤状态机
├── sampling.py        # 采样与过滤器
├── store.py           # 向量库、只读、持久化、FAISS 可选
├── engine.py          # 模型加载、生成循环、检索/注入
├── training.py        # 蒸馏数据训练、数据集加载
├── calibrate.py       # 阈值校准
├── cli.py             # chat / generate / train / calibrate
├── errors.py
└── utils.py           # 特殊 token 净化、增量解码等
examples/
├── basic_generate.py
├── persistent_memory.py
├── train_memory.py
├── memory_service.py
├── calibrate.py
└── data/distill_qa.jsonl
tests/test_stepmem.py
```

---

## 16. 已知限制

- 仅支持单序列生成（batch_size=1）。
- 写入模式的注意力探针要求模型返回 `attentions` 与 `hidden_states`；不支持的自定义模型会抛 `AttentionUnavailableError`。
- 写入模式必须 eager attention，会关闭 SDPA/FlashAttention；非写入模式没有这个限制。
- 非写入模式每个 token 都检索，存在一定开销，但由于维度固定，检索开销恒定。
- 非写入模式逐 token 检索且 `exclude_injected=False`（默认）时，同一记忆可能在连续 token 上重复注入并占满上下文；需要去重请开启该选项。
- 向量库维度绑定模型隐藏层；换模型必须换记忆目录。

---

## 17. 常见问题

**Q: 写入模式和非写入模式怎么选？**
需要把对话步骤写入记忆库时用写入模式（`read_only=False`）；只做记忆检索服务、不需要写库时用非写入模式（`read_only=True`），后者逐 token 检索并保留 SDPA/FlashAttention。

**Q: 非写入模式为什么没有步骤？**
按设计，步骤切分依赖注意力权重，而非写入模式不读取注意力。因此它不切分步骤，而是把每个生成 token 的隐藏状态都当作查询键；`result.steps` 为空，检索轨迹在`result.retrievals`。

**Q: 首个 token 的步骤为什么没有存进记忆？**
单次回复的首个 token 仍会强制切分并用于首次检索，但它的本身就已经作为开头，无法被更早的 token 召回，因此存进记忆没有意义。

**Q: 记忆会出现在回复里吗？**
会。通过相似度阈值和歧义门控的记忆，会直接写入回复并进入 KV cache。

**Q: 记忆训练会改变模型能力吗？**
不会。它只写向量库，不更新权重。训练数据可以来自更大的蒸馏模型；查询键由当前模型生成。

**Q: 为什么训练只生成第一个 token？**
查询键与存储键都取“步骤首 token 的最后一层隐藏状态”。训练只需要这个键，不需要生成整个答案即可检索；assistant 字段直接作为记忆内容，成本更低、答案更可控。

**Q: 只读模式能检索和注入吗？**
能。只读只禁止 add/clear/save，检索、注入都正常；并且每个 token 都检索。

**Q: 为什么默认注入原文不加提示词包装？**
StepMem 以原始步骤文本形式注入，不做摘要或改写，以保持记忆内容不与上下文割裂。需要包装时通过 `RetrievalConfig.prefix/suffix` 显式开启；它们会同时出现在回复和上下文中。

**Q: 记忆会不会无限增长？**
会。当前没有容量上限；建议按场景拆分目录、定期归档或自行实现 LRU/衰减。
