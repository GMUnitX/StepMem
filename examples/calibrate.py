"""示例：阈值校准。

    python examples/calibrate.py --model Qwen2.5-0.5B-Instruct \
        --prompt "解释什么是机器学习。" --target-step-tokens 8
"""

from __future__ import annotations

import argparse
import json
import os

from stepmem import StepMemConfig, StepMemEngine, StoreConfig
from stepmem.calibrate import calibrate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.environ.get("STEPMEM_MODEL", "Qwen2.5-0.5B-Instruct"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--target-step-tokens", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    cfg = StepMemConfig(
        model=args.model,
        device=args.device,
        store=StoreConfig(path=None),
        sampling={"max_new_tokens": args.max_new_tokens, "do_sample": False},
    )
    with StepMemEngine(cfg) as engine:
        result = calibrate(
            engine,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            target_step_tokens=args.target_step_tokens,
        )
    if args.json:
        print(json.dumps(result.report, ensure_ascii=False, indent=2))
    else:
        report = result.report
        print(f"模型: {report['model']}")
        print("相邻注意力相似度分位数:", report["attention_similarity"]["percentiles"])
        print("同 prompt 隐藏态分位数:", report["hidden_within_prompt"]["percentiles"])
        print("跨 prompt 隐藏态分位数:", report["hidden_cross_prompt"]["percentiles"])
        print("建议 boundary_threshold:", report["suggested_boundary_threshold"])
        print("建议 retrieval_threshold:", report["suggested_retrieval_threshold"])


if __name__ == "__main__":
    main()
