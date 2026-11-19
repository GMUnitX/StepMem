"""StepMem 命令行：chat / generate / calibrate。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

from .calibrate import calibrate
from .config import StepMemConfig
from .engine import StepMemEngine
from .errors import StoreError
from .training import load_memory_dataset, validate_memory_record
from .types import (
    DoneEvent,
    InjectionEvent,
    RetrievalEvent,
    StepEndEvent,
    TokenEvent,
)

DEFAULT_MEMORY = str(Path.home() / ".stepmem" / "memory")


# ----------------------------------------------------------------------------
# 参数解析
# ----------------------------------------------------------------------------


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default=os.environ.get("STEPMEM_MODEL"),
        help="Hugging Face 模型名或本地目录（也可用环境变量 STEPMEM_MODEL）",
    )
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0 / mps")
    parser.add_argument(
        "--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16", "float64"]
    )
    parser.add_argument(
        "--attn-implementation",
        default="auto",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        help="非写入模式使用的注意力实现；写入模式会强制 eager",
    )
    parser.add_argument("--threads", type=int, default=None, help="torch CPU 线程数")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-context-tokens", type=int, default=None)
    parser.add_argument("--system", default=None, help="系统提示词（chat）")

    parser.add_argument(
        "--boundary-threshold",
        type=float,
        default=None,
        help="相邻注意力余弦相似度低于该值时切分步骤",
    )
    parser.add_argument(
        "--attention-layers",
        default=None,
        help="参与平均的层，如 '-1' 或 '0,12,-1'，'all' 表示全部层",
    )
    parser.add_argument("--min-step-tokens", type=int, default=None)
    parser.add_argument("--max-step-tokens", type=int, default=None)

    parser.add_argument("--retrieval-threshold", type=float, default=None)
    parser.add_argument(
        "--retrieval-margin",
        type=float,
        default=None,
        help="top1-top2 相似度差下限；低于该值视为歧义并放弃召回",
    )
    parser.add_argument("--no-retrieval", action="store_true")
    parser.add_argument(
        "--max-inject-tokens",
        type=int,
        default=None,
        help="单次写入回复/上下文的最大 token 数",
    )

    parser.add_argument("--memory", default=None, help="记忆目录（设置后跨进程持久化）")
    parser.add_argument("--no-memory", action="store_true", help="强制仅使用内存索引")
    parser.add_argument(
        "--read-only-memory",
        action="store_true",
        help="非写入模式：逐 token 检索，禁止写入/清空/保存",
    )
    parser.add_argument(
        "--no-sanitize-memory",
        action="store_true",
        help="不自动移除记忆文本中的特殊 token（默认会移除）",
    )
    parser.add_argument(
        "--extra-special-token",
        action="append",
        default=None,
        help="额外需要从记忆文本中移除的特殊字符串，可重复",
    )

    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--no-repeat-ngram", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stop", action="append", default=None, help="停止串，可重复")
    parser.add_argument("--verbose", action="store_true")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stepmem",
        description="StepMem · 步骤记忆拓展：内部状态驱动的可扩展步骤记忆推理引擎",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    chat = sub.add_parser("chat", help="交互式多轮对话，支持持久/只读记忆")
    _add_common(chat)
    chat.add_argument("--show-steps", action="store_true", help="显示步骤切分与存储")
    chat.add_argument("--hide-memory", action="store_true", help="不显示记忆注入提示")
    chat.add_argument("--show-stats", action="store_true", help="每轮结束显示统计")

    gen = sub.add_parser("generate", help="单次生成")
    _add_common(gen)
    gen.add_argument("prompt", nargs="?", help="输入提示词（也可用 --prompt）")
    gen.add_argument("--prompt", dest="prompt_option", default=None, help="输入提示词")
    gen.add_argument("--raw", action="store_true", help="不套 chat template，直接补全")
    gen.add_argument("--json", action="store_true", help="输出完整 JSON 结果")
    gen.add_argument("--show-steps", action="store_true")
    gen.add_argument("--show-memory", action="store_true")
    gen.add_argument("--no-context", action="store_true", help="JSON 中省略完整上下文")

    cal = sub.add_parser("calibrate", help="估计边界/检索阈值")
    _add_common(cal)
    cal.add_argument("--prompt", action="append", required=True, help="校准用 prompt，可重复")
    cal.add_argument("--target-step-tokens", type=int, default=8)
    cal.add_argument("--json", action="store_true")

    train = sub.add_parser(
        "train", help="用 {user, assistant} 蒸馏数据训练/扩充记忆库"
    )
    _add_common(train)
    train.add_argument("--dataset", required=True, help="JSONL/JSON 数据集路径")
    train.add_argument("--limit", type=int, default=None, help="只处理前 N 条")
    train.add_argument(
        "--no-chat-template",
        action="store_true",
        help="不套 chat template，直接把 user 文本作为 prompt",
    )
    train.add_argument(
        "--skip-eos", action="store_true", help="首 token 为 EOS 时跳过该样本"
    )
    train.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="同一次训练中允许重复的 (user, assistant)",
    )
    train.add_argument("--dry-run", action="store_true", help="只校验数据集，不加载模型")
    train.add_argument("--json", action="store_true", help="输出 JSON 统计")
    train.add_argument(
        "--progress-every", type=int, default=10, help="每 N 条打印进度；0 关闭"
    )

    return parser


def _parse_layers(value: Optional[str]):
    if value is None:
        return None  # 使用配置默认值
    value = value.strip().lower()
    if value in {"all", "*"}:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--attention-layers 不能为空")
    try:
        return tuple(int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"无法解析层列表: {value}") from exc


def _build_config(args, default_memory: Optional[str]) -> StepMemConfig:
    if not args.model:
        raise SystemExit(
            "错误: 需要 --model 或环境变量 STEPMEM_MODEL，例如 "
            "--model Qwen/Qwen2.5-0.5B-Instruct"
        )
    log_level = "DEBUG" if args.verbose else "WARNING"
    cfg = StepMemConfig(
        model=args.model,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        torch_threads=args.threads,
        max_context_tokens=args.max_context_tokens,
        system_prompt=args.system,
        log_level=log_level,
    )
    if args.boundary_threshold is not None:
        cfg.stepping.boundary_threshold = args.boundary_threshold
    if args.attention_layers is not None:
        cfg.stepping.attention_layers = _parse_layers(args.attention_layers)
    if args.min_step_tokens is not None:
        cfg.stepping.min_step_tokens = args.min_step_tokens
    if args.max_step_tokens is not None:
        cfg.stepping.max_step_tokens = args.max_step_tokens

    if args.retrieval_threshold is not None:
        cfg.retrieval.threshold = args.retrieval_threshold
    if args.retrieval_margin is not None:
        cfg.retrieval.min_score_margin = args.retrieval_margin
    if args.no_retrieval:
        cfg.retrieval.enabled = False
    if args.max_inject_tokens is not None:
        cfg.retrieval.max_inject_tokens = args.max_inject_tokens

    memory = None if args.no_memory else (args.memory or default_memory)
    if memory:
        cfg.store.path = str(memory)
    if args.read_only_memory:
        cfg.store.read_only = True
    if args.no_sanitize_memory:
        cfg.store.sanitize_special_tokens = False
    if args.extra_special_token:
        cfg.store.extra_special_tokens = tuple(args.extra_special_token)

    cfg.validate()
    return cfg


def _sampling_overrides(args) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    simple = {
        "max_new_tokens": "max_new_tokens",
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "min_p": "min_p",
        "repetition_penalty": "repetition_penalty",
        "no_repeat_ngram_size": "no_repeat_ngram",
        "seed": "seed",
    }
    for config_key, attr in simple.items():
        value = getattr(args, attr, None)
        if value is not None:
            overrides[config_key] = value
    if args.stop:
        overrides["stop_strings"] = tuple(args.stop)
    return overrides


# ----------------------------------------------------------------------------
# 输出辅助
# ----------------------------------------------------------------------------


def _print_stats(result) -> None:
    stats = result.stats
    print(
        f"[完成: {stats.finish_reason} | {stats.generated_tokens} tokens | "
        f"{stats.tokens_per_second:.2f} tok/s | 步骤 {stats.steps} | "
        f"边界 {stats.boundaries} | 检索 {stats.retrievals} | "
        f"注入 {stats.injections}次/"
        f"{stats.injected_tokens}tok | 存储 {stats.stored_steps}]"
    )


def _retrieval_label(step) -> str:
    retrieval = step.retrieval
    if retrieval is None:
        return "-"
    similarity = (
        "-" if retrieval.similarity is None else f"{retrieval.similarity:.4f}"
    )
    if retrieval.accepted:
        label = "注入"
    else:
        reason_map = {
            "below_threshold": "低于阈值",
            "no_candidates": "无候选",
            "ambiguous": "歧义",
            "empty_text": "空文本",
            "nothing_to_inject": "无注入内容",
            "context_limit": "上下文满",
            "store_error": "存储错误",
            "encode_error": "编码错误",
        }
        label = reason_map.get(retrieval.reason, retrieval.reason)
    return f"{label}@{similarity}"


def _print_steps(result) -> None:
    if not result.steps:
        print("（本次没有步骤：非写入模式按 token 检索，或尚未产生步骤边界）")
        return
    print("\n步骤明细：")
    print("  #   上下文位置   tokens  边界相似度  边界原因   存储     检索")
    for step in result.steps:
        sim = "-" if step.boundary_similarity is None else f"{step.boundary_similarity:.4f}"
        retrieval = _retrieval_label(step)
        stored = f"#{step.record_id}" if step.stored else "-"
        print(
            f"  {step.index + 1:<3} {step.start_position:>4}-{step.end_position:<8} "
            f"{len(step.token_ids):>5}  {sim:>9}  {step.boundary_reason:<9} "
            f"{stored:<8} {retrieval}"
        )


def _render_stream(events, *, show_memory: bool, show_steps: bool):
    result = None
    for event in events:
        if isinstance(event, TokenEvent):
            if event.text:
                print(event.text, end="", flush=True)
        elif isinstance(event, RetrievalEvent):
            if event.accepted and show_memory:
                position = (
                    f" | pos={event.position}"
                    if event.position is not None
                    else ""
                )
                print(
                    f"\n  [注入记忆 #{event.record_id} | "
                    f"sim={event.similarity:.3f}{position}]\n",
                    end="",
                    flush=True,
                )
        elif isinstance(event, InjectionEvent):
            # 注入文本属于可见回复，必须打印。
            print(event.text, end="", flush=True)
        elif isinstance(event, StepEndEvent):
            if show_steps:
                tag = "最终步骤" if event.is_final else "步骤边界"
                stored = f"已存 #{event.record_id}" if event.stored else "未存储"
                print(
                    f"\n  [{tag} {event.step_index + 1} | "
                    f"{event.token_count} tokens | {stored}]\n",
                    end="",
                    flush=True,
                )
        elif isinstance(event, DoneEvent):
            result = event.result
    print()
    return result


# ----------------------------------------------------------------------------
# 子命令
# ----------------------------------------------------------------------------


def _cmd_chat(args) -> int:
    cfg = _build_config(args, default_memory=DEFAULT_MEMORY)
    engine = StepMemEngine(cfg)
    sampling = _sampling_overrides(args)
    messages: list[dict[str, str]] = []
    last_result = None
    print("StepMem 交互模式。输入 /help 查看命令，/exit 退出。")
    try:
        while True:
            try:
                user = input("\n你: ").strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\n已退出。")
                break
            if not user:
                continue
            if user.startswith("/"):
                command = user.split()[0].lower()
                if command in {"/exit", "/quit"}:
                    break
                if command == "/help":
                    print(
                        "/exit        退出\n"
                        "/reset       清空对话历史（不清除长期记忆）\n"
                        "/memories    查看长期记忆数量与路径\n"
                        "/save        立即保存长期记忆\n"
                        "/clear       清空长期记忆并保存（不可恢复）\n"
                        "/stats       显示上一轮生成统计"
                    )
                    continue
                if command == "/reset":
                    messages.clear()
                    print("对话历史已清空。")
                    continue
                if command == "/memories":
                    mode = (
                        "非写入/逐token检索"
                        if engine.store.read_only
                        else "写入/步骤切分"
                    )
                    print(
                        f"长期记忆: {len(engine.store)} 条 | 模式: {mode} | "
                        f"路径: {cfg.store.path or '(仅内存)'}"
                    )
                    continue
                if command == "/save":
                    if engine.store.read_only:
                        print("当前为只读（非写入）模式，未执行保存。")
                        continue
                    try:
                        engine.save_memory()
                    except StoreError as exc:
                        print(f"保存失败: {exc}")
                    else:
                        print(f"已保存 {len(engine.store)} 条记忆。")
                    continue
                if command == "/clear":
                    try:
                        engine.store.clear()
                        engine.save_memory()
                    except StoreError as exc:
                        print(f"清空失败: {exc}")
                    else:
                        print("长期记忆已清空。")
                    continue
                if command == "/stats":
                    if last_result is None:
                        print("还没有生成记录。")
                    else:
                        _print_stats(last_result)
                        _print_steps(last_result) if args.show_steps else None
                    continue
                print(f"未知命令: {user}")
                continue

            messages.append({"role": "user", "content": user})
            print("StepMem: ", end="", flush=True)
            try:
                result = _render_stream(
                    engine.stream_chat(messages, sampling=sampling),
                    show_memory=not args.hide_memory,
                    show_steps=args.show_steps,
                )
            except KeyboardInterrupt:
                print("\n[已中断当前生成]")
                messages.pop()
                continue
            if result is None:
                messages.pop()
                continue
            messages.append({"role": "assistant", "content": result.text})
            last_result = result
            if args.show_stats:
                _print_stats(result)
    finally:
        engine.close()
        if cfg.store.path:
            mode = (
                "非写入/逐token检索" if cfg.store.read_only else "写入/步骤切分"
            )
            print(
                f"\n长期记忆: {len(engine.store)} 条 | 模式: {mode} | "
                f"路径: {cfg.store.path}"
            )
    return 0


def _cmd_generate(args) -> int:
    prompt = args.prompt_option or args.prompt
    if not prompt:
        raise SystemExit("错误: 需要位置参数 prompt 或 --prompt")
    cfg = _build_config(args, default_memory=None)
    engine = StepMemEngine(cfg)
    sampling = _sampling_overrides(args)
    try:
        if args.json:
            if args.raw:
                result = engine.complete(prompt, sampling=sampling)
            else:
                result = engine.chat(prompt, sampling=sampling)
            print(
                json.dumps(
                    result.to_dict(include_context=not args.no_context),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        if args.raw:
            events = engine.stream_complete(prompt, sampling=sampling)
        else:
            events = engine.stream_chat(prompt, sampling=sampling)
        result = _render_stream(
            events,
            show_memory=args.show_memory,
            show_steps=args.show_steps,
        )
        if result is not None:
            _print_stats(result)
            if args.show_steps:
                _print_steps(result)
        return 0
    finally:
        engine.close()


def _fmt(value) -> str:
    return "-" if value is None else f"{value:.4f}"


def _cmd_calibrate(args) -> int:
    cfg = _build_config(args, default_memory=None)
    # 校准需要注意力权重；强制写入模式（eager），且不使用记忆库。
    cfg.store.path = None
    cfg.store.read_only = False
    engine = StepMemEngine(cfg)
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else 64
    try:
        result = calibrate(
            engine,
            args.prompt,
            max_new_tokens=max_new_tokens,
            target_step_tokens=args.target_step_tokens,
            seed=args.seed,
        )
        report = result.report
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0

        print(f"模型: {report['model']}")
        print(f"目标平均步骤长度: {report['target_step_tokens']} tokens\n")
        print("相邻 token 注意力相似度：")
        attn = report["attention_similarity"]
        print(f"  样本数 {attn['count']}")
        for key, value in attn["percentiles"].items():
            print(f"  {key:>6}: {_fmt(value)}")
        print("\n隐藏状态成对余弦（同 prompt）：")
        within = report["hidden_within_prompt"]
        print(f"  样本数 {within['count']}")
        for key, value in within["percentiles"].items():
            print(f"  {key:>6}: {_fmt(value)}")
        if report["hidden_cross_prompt"]["count"]:
            print("\n隐藏状态成对余弦（跨 prompt）：")
            for key, value in report["hidden_cross_prompt"]["percentiles"].items():
                print(f"  {key:>6}: {_fmt(value)}")
            print(
                f"\n分离度 (within p90 - cross p99): "
                f"{_fmt(report['separation_p90_within_minus_p99_cross'])}"
            )
        print("\n建议参数：")
        print(
            f"  --boundary-threshold {_fmt(report['suggested_boundary_threshold'])}"
        )
        print(
            f"  --retrieval-threshold {_fmt(report['suggested_retrieval_threshold'])}"
        )
        print("\n注意：")
        for note in report["notes"]:
            print(f"  - {note}")
        return 0
    finally:
        engine.close()


def _cmd_train(args) -> int:
    if args.no_memory:
        raise SystemExit("错误: train 需要持久化记忆库，不能使用 --no-memory")
    if args.read_only_memory:
        raise SystemExit("错误: 只读（非写入）模式不能训练；请去掉 --read-only-memory")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("错误: --limit 必须 >= 1")

    records = load_memory_dataset(args.dataset)
    selected = records[: args.limit] if args.limit else records

    if args.dry_run:
        valid = 0
        errors: list[tuple[int, str]] = []
        for index, record in enumerate(selected):
            try:
                validate_memory_record(record, index)
            except Exception as exc:
                errors.append((index, str(exc)))
                continue
            valid += 1
        payload = {
            "dry_run": True,
            "total": len(selected),
            "valid": valid,
            "failed": len(errors),
            "errors": [
                {"index": idx, "message": message} for idx, message in errors[:20]
            ],
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(
                f"数据集校验: 共 {payload['total']} 条 | 有效 {valid} | "
                f"失败 {len(errors)}"
            )
            for idx, message in errors[:10]:
                print(f"  - 第 {idx + 1} 条: {message}")
            print("（dry-run 未加载模型，也未检查特殊 token 清洗结果）")
        return 0 if not errors else 1

    if not selected:
        raise SystemExit("错误: 数据集为空")

    cfg = _build_config(args, default_memory=DEFAULT_MEMORY)
    if cfg.store.read_only:
        raise SystemExit("错误: 只读（非写入）模式不能训练")
    engine = StepMemEngine(cfg)
    sampling = _sampling_overrides(args) or None
    progress_every = int(args.progress_every)

    def progress(done: int, total: int, stored: int) -> None:
        if progress_every > 0 and (done == total or done % progress_every == 0):
            print(
                f"[训练] {done}/{total} | 已写入 {stored}",
                file=sys.stderr,
                flush=True,
            )

    try:
        result = engine.train_memory(
            records,
            sampling=sampling,
            use_chat_template=not args.no_chat_template,
            skip_eos=args.skip_eos,
            max_samples=args.limit,
            dedupe=not args.allow_duplicates,
            progress=progress if progress_every > 0 else None,
            save=True,
        )
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        else:
            stats = result.stats
            print(f"\n记忆训练完成（session {result.session_id}）")
            print(f"  样本总数: {stats.total}")
            print(
                f"  已写入: {stats.stored} 条（assistant {stats.tokens_stored} tokens）"
            )
            print(f"  跳过空文本: {stats.skipped_empty}")
            print(f"  跳过重复: {stats.skipped_duplicate}")
            print(f"  跳过 EOS: {stats.skipped_eos}")
            print(f"  失败: {stats.failed}")
            print(f"  耗时: {stats.elapsed_seconds:.2f}s")
            print(f"  记忆库: {len(engine.store)} 条 | 路径: {cfg.store.path}")
            if result.errors:
                print("  错误（最多显示 10 条）：")
                for err in result.errors[:10]:
                    print(f"    - 第 {err.index + 1} 条: {err.message}")
        return 0 if result.stats.failed == 0 else 1
    finally:
        engine.close()


def main(argv: Optional[list[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "chat":
            return _cmd_chat(args)
        if args.command == "generate":
            return _cmd_generate(args)
        if args.command == "calibrate":
            return _cmd_calibrate(args)
        if args.command == "train":
            return _cmd_train(args)
        parser.error("未知命令")
        return 2
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130
    except SystemExit:
        raise
    except Exception as exc:
        if getattr(args, "verbose", False):
            traceback.print_exc()
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
