"""示例：只读（非写入）记忆库服务。

先用 examples/train_memory.py 训练记忆，再运行：

    python examples/memory_service.py --memory ./stepmem_distill_memory \
        --prompt "9 的平方是多少？"

read_only=True 时引擎进入非写入模式：不切分步骤、保留 SDPA/FlashAttention，
每个生成 token 的隐藏状态都会检索记忆。命中的记忆会直接写入可见回复（同时进入
KV cache），服务阶段不会改写记忆库。
"""

from __future__ import annotations

import argparse
import os

from stepmem import (
    RetrievalConfig,
    SamplingConfig,
    StepMemConfig,
    StepMemEngine,
    SteppingConfig,
    StoreConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.environ.get("STEPMEM_MODEL", "Qwen2.5-0.5B-Instruct"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--memory", default="./stepmem_distill_memory")
    parser.add_argument("--prompt", default="9 的平方是多少？")
    parser.add_argument("--retrieval-threshold", type=float, default=0.45)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--max-inject-tokens", type=int, default=128)
    parser.add_argument("--show-steps", action="store_true")
    args = parser.parse_args()

    cfg = StepMemConfig(
        model=args.model,
        device=args.device,
        attn_implementation="auto",  # 非写入模式保留 SDPA / FlashAttention
        stepping=SteppingConfig(attention_layers=(-1,), boundary_threshold=0.75),
        retrieval=RetrievalConfig(
            enabled=True,
            threshold=args.retrieval_threshold,
            min_score_margin=args.margin,
            max_inject_tokens=args.max_inject_tokens,
        ),
        store=StoreConfig(path=args.memory, read_only=True),
        sampling=SamplingConfig(max_new_tokens=32, do_sample=False),
    )

    with StepMemEngine(cfg) as engine:
        print(f"只读记忆库: {len(engine.store)} 条 | {args.memory}")
        result = engine.chat(args.prompt)

        print("\n=== 输出 ===")
        print(result.text)
        print("\n=== 记忆决策 ===")
        if not result.injections:
            print("（没有命中记忆；可降低 --retrieval-threshold 或检查训练数据）")
        for item in result.injections:
            print(
                f"- 注入 #{item.record_id} sim={item.similarity:.4f} "
                f"{item.token_count} tokens: {item.text!r}"
            )
        print("\n=== 统计 ===")
        print(result.stats)
        if args.show_steps:
            for step in result.steps:
                print(f"步骤 {step.index + 1}: {step.text!r} retrieval={step.retrieval}")


if __name__ == "__main__":
    main()
