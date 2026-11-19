"""示例：最小可运行的 StepMem 生成，并打印步骤切分结果。

    python examples/basic_generate.py
    python examples/basic_generate.py --model Qwen/Qwen2.5-0.5B-Instruct
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
    parser.add_argument("--prompt", default="用三句话解释什么是光合作用。")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    cfg = StepMemConfig(
        model=args.model,
        device=args.device,
        stepping=SteppingConfig(attention_layers=(-1,), boundary_threshold=0.75),
        retrieval=RetrievalConfig(enabled=False),  # 本示例只看步骤切分
        store=StoreConfig(path=None),
        sampling=SamplingConfig(
            max_new_tokens=args.max_new_tokens, do_sample=False
        ),
    )

    with StepMemEngine(cfg) as engine:
        result = engine.chat(args.prompt)
        print("=== 生成 ===")
        print(result.text)
        print("\n=== 步骤 ===")
        for step in result.steps:
            sim = "-" if step.boundary_similarity is None else f"{step.boundary_similarity:.4f}"
            print(
                f"#{step.index + 1} 位置 {step.start_position}-{step.end_position} "
                f"tokens={len(step.token_ids)} 边界相似度={sim} "
                f"原因={step.boundary_reason} 存储={step.stored}"
            )
            print(f"   {step.text!r}")
        print("\n=== 统计 ===")
        print(result.stats)


if __name__ == "__main__":
    main()
