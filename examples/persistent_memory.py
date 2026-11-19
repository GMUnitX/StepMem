"""示例：跨进程持久化记忆。

第一次运行会写入步骤记忆；第二次运行（甚至换一个进程）会加载同一目录，
在步骤边界检索并注入命中的历史步骤。

    python examples/persistent_memory.py
    python examples/persistent_memory.py --reset
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

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
    parser.add_argument("--memory", default="./stepmem_memory_demo")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument(
        "--prompt", default="请用一句话说明植物如何进行光合作用。"
    )
    args = parser.parse_args()

    if args.reset:
        import shutil

        shutil.rmtree(args.memory, ignore_errors=True)
        print(f"已清空记忆目录 {args.memory}")

    cfg = StepMemConfig(
        model=args.model,
        device=args.device,
        stepping=SteppingConfig(attention_layers=(-1,), boundary_threshold=0.75),
        retrieval=RetrievalConfig(
            enabled=True,
            threshold=0.5,
            max_inject_tokens=128,
            # 默认注入原始步骤文本；需要分隔时可设置 prefix="\n[记忆] "。
        ),
        store=StoreConfig(path=args.memory),
        sampling=SamplingConfig(max_new_tokens=64, do_sample=False),
    )

    with StepMemEngine(cfg) as engine:
        print(f"加载记忆: {len(engine.store)} 条 ({args.memory})")
        result = engine.chat(args.prompt)

        print("\n=== 生成 ===")
        print(result.text)
        print("\n=== 记忆注入 ===")
        if not result.injections:
            print("（本次没有命中记忆）")
        for item in result.injections:
            print(
                f"- 记录 #{item.record_id} 相似度={item.similarity:.4f} "
                f"注入 {item.token_count} tokens"
            )
            print(f"  {item.text!r}")
        print("\n=== 本次写入记忆 ===")
        for step in result.steps:
            if step.stored:
                print(f"- 步骤 #{step.index + 1} → 记录 #{step.record_id}: {step.text!r}")
        print(f"\n记忆总数: {len(engine.store)} | 目录: {Path(args.memory).resolve()}")


if __name__ == "__main__":
    main()
