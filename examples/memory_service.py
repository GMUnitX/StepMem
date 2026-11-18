"""示例：只读记忆库服务。

先用 examples/train_memory.py 训练记忆，再运行：

    python examples/memory_service.py --memory ./stepmem_distill_memory \
        --prompt "9 的平方是多少？"

命中的记忆会直接写入可见回复（同时进入 KV cache）；read_only=True 保证服务阶段
不会改写记忆库。注意：注入内容默认是记忆原文，建议配合 prefix/suffix 或训练时
直接写入适合展示的答案文本。
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
        default=os.environ.get("STEPMEM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
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
