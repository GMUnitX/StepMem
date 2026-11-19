"""示例：用 {user, assistant} 蒸馏数据训练步骤记忆库。

    python examples/train_memory.py --reset
    python examples/train_memory.py --dataset examples/data/distill_qa.jsonl --limit 3

每个样本只做一次 prompt prefill + 一次首 token 前向，不更新模型权重。
assistant 字段作为记忆内容；首 token 隐藏状态作为查询键。
"""

from __future__ import annotations

import argparse
import os
import shutil
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
    parser.add_argument("--dataset", default="examples/data/distill_qa.jsonl")
    parser.add_argument("--memory", default="./stepmem_distill_memory")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reset", action="store_true", help="先清空记忆目录")
    parser.add_argument("--skip-eos", action="store_true")
    args = parser.parse_args()

    if args.reset:
        shutil.rmtree(args.memory, ignore_errors=True)
        print(f"已清空记忆目录 {args.memory}")

    cfg = StepMemConfig(
        model=args.model,
        device=args.device,
        stepping=SteppingConfig(attention_layers=(-1,), boundary_threshold=0.75),
        retrieval=RetrievalConfig(enabled=False),  # 训练阶段不检索
        store=StoreConfig(path=args.memory),
        sampling=SamplingConfig(do_sample=False, max_new_tokens=1),
    )

    with StepMemEngine(cfg) as engine:
        print(f"训练前记忆: {len(engine.store)} 条 | {args.memory}")
        result = engine.train_memory(
            args.dataset,
            sampling={"do_sample": False},  # 首 token 贪心，训练键更稳定
            skip_eos=args.skip_eos,
            max_samples=args.limit,
            dedupe=True,
            progress=lambda done, total, stored: print(
                f"\r[训练] {done}/{total} | 已写入 {stored}",
                end="",
                flush=True,
            ),
        )
        print()
        stats = result.stats
        print(f"样本总数: {stats.total}")
        print(f"写入记忆: {stats.stored} 条（{stats.tokens_stored} assistant tokens）")
        print(f"跳过空文本: {stats.skipped_empty}")
        print(f"跳过重复: {stats.skipped_duplicate}")
        print(f"跳过 EOS: {stats.skipped_eos}")
        print(f"失败: {stats.failed}")
        print(f"耗时: {stats.elapsed_seconds:.2f}s")
        print(f"训练后记忆: {len(engine.store)} 条 | {Path(args.memory).resolve()}")
        for record in engine.store.records[-3:]:
            print(f"  #{record.id} [{record.metadata.get('user_preview', '')[:30]}] -> {record.text!r}")

    print("\n下一步：python examples/memory_service.py --memory", args.memory)


if __name__ == "__main__":
    main()
