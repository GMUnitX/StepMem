"""StepMem 单元测试与离线端到端测试。

运行：``python -m unittest discover -s tests -v``
"""

from __future__ import annotations

import unittest
from typing import Optional

import numpy as np
import torch

from stepmem import (
    AttentionSample,
    DoneEvent,
    InjectionEvent,
    RetrievalEvent,
    RetrievalConfig,
    SamplingConfig,
    StepMemConfig,
    StepMemEngine,
    SteppingConfig,
    StepTracker,
    StoreConfig,
    VectorStore,
    cosine_overlap,
    load_memory_dataset,
    reduce_attention,
    resolve_attention_layers,
    sanitize_memory_text,
    strip_matched_prefix,
    validate_memory_record,
)
from stepmem.errors import DimensionMismatchError, StepMemError, StoreError
from stepmem.sampling import Sampler, make_generator
from stepmem.stepping import ClosedStep
from stepmem.utils import IncrementalDecoder, find_stop_string, to_jsonable


# ----------------------------------------------------------------------------
# 测试替身
# ----------------------------------------------------------------------------


class ToyTokenizer:
    """64 词表的字符级 tokenizer，保证所有 token 解码为非空白字符。"""

    _ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-"

    def __init__(self) -> None:
        assert len(self._ALPHABET) == 64
        self._itos = {i: ch for i, ch in enumerate(self._ALPHABET)}
        self._stoi = {ch: i for i, ch in enumerate(self._ALPHABET)}
        self.eos_token_id = None
        self.pad_token_id = 0
        self.bos_token_id = None
        self.chat_template = "toy"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.unk_token = None
        self.bos_token = None
        self.all_special_tokens = ["<eos>", "<pad>"]
        self.all_special_ids = []

    def encode(self, text, add_special_tokens=True):
        return [self._stoi[ch] for ch in text if ch in self._stoi]

    def decode(self, ids, skip_special_tokens=True, **kwargs):
        return "".join(self._itos[int(i)] for i in ids)

    def __call__(self, text, return_tensors=None, add_special_tokens=True, **kwargs):
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return text


def _tiny_model():
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=64,
        n_positions=128,
        n_embd=32,
        n_layer=2,
        n_head=2,
        bos_token_id=None,
        eos_token_id=None,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = GPT2LMHeadModel(config)
    model.set_attn_implementation("eager")
    model.eval()
    return model


def _cfg(**overrides) -> StepMemConfig:
    base = dict(
        model="toy",
        device="cpu",
        dtype="float32",
        log_level="ERROR",
        stepping=SteppingConfig(
            boundary_threshold=1.0, min_step_tokens=1, max_step_tokens=1
        ),
        retrieval=RetrievalConfig(enabled=False, threshold=0.0, max_inject_tokens=8),
        store=StoreConfig(path=None, min_tokens_to_store=1, store_final_step=True),
        sampling=SamplingConfig(max_new_tokens=6, do_sample=False),
    )
    base.update(overrides)
    return StepMemConfig(**base)


# ----------------------------------------------------------------------------
# 存储
# ----------------------------------------------------------------------------


class TestVectorStore(unittest.TestCase):
    def test_add_and_search(self):
        store = VectorStore(4)
        r1 = store.add([3, 4, 0, 0], "alpha", session_id="s1", step_index=0, token_count=3)
        r2 = store.add([0, 0, 1, 0], "beta", session_id="s1", step_index=1, token_count=2)
        self.assertEqual(len(store), 2)
        self.assertEqual((r1.id, r2.id), (1, 2))

        hits = store.search([6, 8, 0, 0], top_k=2)
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].record.text, "alpha")
        self.assertAlmostEqual(hits[0].similarity, 1.0, places=5)
        self.assertAlmostEqual(hits[1].similarity, 0.0, places=5)

    def test_threshold_and_exclusions(self):
        store = VectorStore(3)
        a = store.add([1, 0, 0], "a", session_id="s1")
        b = store.add([0.9, 0.1, 0], "b", session_id="s2")
        store.add([0, 1, 0], "c", session_id="s2")

        self.assertEqual([h.record.text for h in store.search([1, 0, 0], top_k=1)], ["a"])
        self.assertEqual(
            [h.record.text for h in store.search([1, 0, 0], top_k=1, exclude_ids={a.id})],
            ["b"],
        )
        self.assertEqual(
            [h.record.text for h in store.search([1, 0, 0], top_k=1, exclude_session="s2")],
            ["a"],
        )
        self.assertEqual(
            [h.record.text for h in store.search([1, 0, 0], min_similarity=0.999)],
            ["a"],
        )
        self.assertEqual(store.search([1, 0, 0], exclude_ids={a.id, b.id}, exclude_session="s2"), [])

    def test_invalid_keys_and_text(self):
        store = VectorStore(2)
        with self.assertRaises(DimensionMismatchError):
            store.add([1, 0, 0], "x")
        with self.assertRaises(ValueError):
            store.add([0, 0], "zero")
        with self.assertRaises(ValueError):
            store.add([1, 0], "")
        with self.assertRaises(ValueError):
            store.search([0, 0])

    def test_persistence_roundtrip(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            store = VectorStore(4, StoreConfig(path=tmp))
            store.add([1, 0, 0, 0], "第一条", session_id="s", step_index=4, token_count=7,
                      metadata={"k": "v"})
            store.add([0, 1, 0, 0], "second", session_id="s", step_index=5, token_count=2)
            store.save()
            self.assertTrue((__import__("pathlib").Path(tmp) / "manifest.json").exists())

            loaded = VectorStore.load(tmp, dim=4)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded.records[0].text, "第一条")
            self.assertEqual(loaded.records[0].metadata, {"k": "v"})
            self.assertEqual(loaded.records[1].token_count, 2)
            hits = loaded.search([1, 0, 0, 0], top_k=1)
            self.assertEqual(hits[0].record.id, 1)

            new_record = loaded.add([0, 0, 1, 0], "third")
            self.assertEqual(new_record.id, 3)

            with self.assertRaises(DimensionMismatchError):
                VectorStore.load(tmp, dim=3)

    def test_autosave(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            store = VectorStore(
                2, StoreConfig(path=tmp, autosave=True, save_every=2)
            )
            store.add([1, 0], "one")
            self.assertFalse((Path(tmp) / "manifest.json").exists())
            store.add([0, 1], "two")
            self.assertTrue((Path(tmp) / "manifest.json").exists())

    def test_clear(self):
        store = VectorStore(2)
        store.add([1, 0], "one")
        store.clear()
        self.assertEqual(len(store), 0)
        self.assertEqual(store.search([1, 0]), [])
        with self.assertRaises(ValueError):
            store.search([0, 0])

    def test_read_only_mode(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            writable = VectorStore(2, StoreConfig(path=tmp))
            writable.add([1, 0], "one")
            writable.save()

            readonly = VectorStore.load(
                tmp, dim=2, config=StoreConfig(path=tmp, read_only=True)
            )
            self.assertEqual(len(readonly), 1)
            self.assertEqual(readonly.search([1, 0], top_k=1)[0].record.text, "one")
            with self.assertRaises(StoreError):
                readonly.add([0, 1], "two")
            with self.assertRaises(StoreError):
                readonly.save()
            with self.assertRaises(StoreError):
                readonly.clear()


# ----------------------------------------------------------------------------
# 注意力与步骤状态机
# ----------------------------------------------------------------------------


class TestAttentionAndStepping(unittest.TestCase):
    def test_reduce_attention(self):
        layer0 = torch.tensor(
            [[[[0.2, 0.3, 0.5]], [[0.4, 0.4, 0.2]]]]
        )  # [1, 2 heads, 1, 3]
        layer1 = torch.tensor([[[[0.1, 0.1, 0.8]], [[0.3, 0.3, 0.4]]]])
        vec = reduce_attention((layer0, layer1), layers=(0, 1), head_reduction="mean")
        expected = np.array([0.25, 0.275, 0.475], dtype=np.float32)
        np.testing.assert_allclose(vec, expected, rtol=0, atol=1e-6)

        last = reduce_attention((layer0, layer1), layers=(-1,), head_reduction="mean")
        np.testing.assert_allclose(last, [0.2, 0.2, 0.6], atol=1e-6)

        first_head = reduce_attention((layer0, layer1), layers=(0,), head_reduction="first")
        np.testing.assert_allclose(first_head, [0.2, 0.3, 0.5], atol=1e-6)

    def test_resolve_layers(self):
        self.assertEqual(resolve_attention_layers(None, 3), [0, 1, 2])
        self.assertEqual(resolve_attention_layers((-1,), 3), [2])
        self.assertEqual(resolve_attention_layers((0, -1, 0), 3), [0, 2])
        with self.assertRaises(ValueError):
            resolve_attention_layers((3,), 3)

    def test_cosine_overlap_alignment(self):
        a = AttentionSample(np.array([1, 2, 3, 4], dtype=np.float32), 0)
        b = AttentionSample(np.array([2, 3, 4, 5], dtype=np.float32), 1)
        self.assertAlmostEqual(cosine_overlap(a, b), 1.0, places=6)

        c = AttentionSample(np.array([1, 0], dtype=np.float32), 0)
        d = AttentionSample(np.array([0, 1], dtype=np.float32), 2)
        self.assertIsNone(cosine_overlap(c, d))
        self.assertIsNone(cosine_overlap(None, d))

        zero = AttentionSample(np.zeros(3, dtype=np.float32), 0)
        self.assertIsNone(cosine_overlap(zero, a))

    def test_tracker_boundary_and_finish(self):
        tracker = StepTracker(SteppingConfig(boundary_threshold=0.5, min_step_tokens=1))
        same = AttentionSample(np.array([1, 0, 0], dtype=np.float32), 0)
        orthogonal = AttentionSample(np.array([0, 1, 0], dtype=np.float32), 0)

        d0 = tracker.observe(0, 10, np.ones(4), same)
        self.assertTrue(d0.started)
        self.assertEqual(d0.reason, "initial")

        d1 = tracker.observe(1, 11, np.ones(4), same)
        self.assertFalse(d1.started)
        self.assertIsNone(d1.closed)
        self.assertAlmostEqual(d1.similarity, 1.0)

        d2 = tracker.observe(2, 12, np.ones(4), orthogonal)
        self.assertTrue(d2.started)
        self.assertEqual(d2.reason, "threshold")
        self.assertIsInstance(d2.closed, ClosedStep)
        self.assertEqual(d2.closed.token_ids, [10, 11])
        self.assertEqual((d2.closed.start_position, d2.closed.end_position), (0, 1))
        np.testing.assert_allclose(d2.closed.key, np.ones(4))

        final = tracker.finish()
        self.assertEqual(final.token_ids, [12])
        self.assertEqual((final.start_position, final.end_position), (2, 2))
        self.assertIsNone(tracker.finish())

    def test_min_step_tokens(self):
        tracker = StepTracker(SteppingConfig(boundary_threshold=0.9, min_step_tokens=2))
        high = AttentionSample(np.array([1, 0], dtype=np.float32), 0)
        low = AttentionSample(np.array([0, 1], dtype=np.float32), 0)
        tracker.observe(0, 1, np.ones(3), high)
        d1 = tracker.observe(1, 2, np.ones(3), low)
        self.assertFalse(d1.started)  # 只有 1 个 token，忽略边界
        d2 = tracker.observe(2, 3, np.ones(3), high)
        self.assertTrue(d2.started)
        self.assertEqual(d2.closed.token_ids, [1, 2])

    def test_max_step_tokens(self):
        tracker = StepTracker(
            SteppingConfig(
                boundary_threshold=0.0, min_step_tokens=1, max_step_tokens=2
            )
        )
        same = AttentionSample(np.array([1, 0], dtype=np.float32), 0)
        tracker.observe(0, 1, np.ones(3), same)
        tracker.observe(1, 2, np.ones(3), same)
        decision = tracker.observe(2, 3, np.ones(3), same)
        self.assertTrue(decision.started)
        self.assertEqual(decision.reason, "max_tokens")
        self.assertEqual(decision.closed.token_ids, [1, 2])

    def test_no_attention_means_no_boundary(self):
        tracker = StepTracker(SteppingConfig(boundary_threshold=1.0))
        tracker.observe(0, 1, np.ones(3), None)
        decision = tracker.observe(1, 2, np.ones(3), None)
        self.assertFalse(decision.started)
        self.assertIsNone(decision.similarity)


# ----------------------------------------------------------------------------
# 采样
# ----------------------------------------------------------------------------


class TestSampling(unittest.TestCase):
    def test_greedy(self):
        sampler = Sampler(SamplingConfig(do_sample=False))
        self.assertEqual(sampler.sample(torch.tensor([0.1, 2.0, 0.5]), []), 1)
        sampler = Sampler(SamplingConfig(do_sample=True, temperature=0.0))
        self.assertEqual(sampler.sample(torch.tensor([3.0, 1.0]), []), 0)

    def test_temperature_zero_ignores_sampling(self):
        sampler = Sampler(SamplingConfig(do_sample=True, temperature=0.0, top_k=1))
        self.assertEqual(sampler.sample(torch.tensor([0.0, 5.0, 4.0]), []), 1)

    def test_top_k_one(self):
        sampler = Sampler(
            SamplingConfig(do_sample=True, temperature=1.0, top_k=1, top_p=1.0)
        )
        for seed in range(5):
            generator = make_generator(seed, torch.device("cpu"))
            self.assertEqual(
                sampler.sample(torch.tensor([0.2, 1.0, 0.8]), [], generator), 1
            )

    def test_top_p_filters(self):
        logits = torch.log(torch.tensor([0.7, 0.2, 0.1]))
        sampler = Sampler(
            SamplingConfig(do_sample=True, temperature=1.0, top_k=0, top_p=0.5)
        )
        for seed in range(5):
            generator = make_generator(seed, torch.device("cpu"))
            self.assertEqual(sampler.sample(logits, [], generator), 0)

    def test_repetition_penalty(self):
        sampler = Sampler(
            SamplingConfig(do_sample=False, repetition_penalty=2.0)
        )
        # token0: 3/2 = 1.5, token1: 2 → token1 胜出
        self.assertEqual(sampler.sample(torch.tensor([3.0, 2.0]), [0]), 1)

    def test_no_repeat_ngram(self):
        sampler = Sampler(
            SamplingConfig(do_sample=False, no_repeat_ngram_size=3)
        )
        logits = torch.tensor([1.0, 0.0, 0.0, 5.0])
        # 上下文 [1,2,3,1,2] 的后续 3 总是 3，应被禁止
        self.assertNotEqual(sampler.sample(logits, [1, 2, 3, 1, 2]), 3)

    def test_no_repeat_ngram_one(self):
        sampler = Sampler(
            SamplingConfig(do_sample=False, no_repeat_ngram_size=1)
        )
        logits = torch.tensor([5.0, 4.0, 1.0])
        self.assertEqual(sampler.sample(logits, [0]), 1)

    def test_min_p(self):
        logits = torch.log(torch.tensor([0.8, 0.15, 0.05]))
        sampler = Sampler(
            SamplingConfig(
                do_sample=True, temperature=1.0, top_k=0, top_p=1.0, min_p=0.5
            )
        )
        for seed in range(5):
            generator = make_generator(seed, torch.device("cpu"))
            self.assertEqual(sampler.sample(logits, [], generator), 0)

    def test_seeded_reproducibility(self):
        sampler = Sampler(
            SamplingConfig(do_sample=True, temperature=1.0, top_k=0, top_p=1.0)
        )
        logits = torch.zeros(5)
        first = [
            sampler.sample(logits, [], make_generator(42, torch.device("cpu")))
            for _ in range(10)
        ]
        second = [
            sampler.sample(logits, [], make_generator(42, torch.device("cpu")))
            for _ in range(10)
        ]
        self.assertEqual(first, second)


# ----------------------------------------------------------------------------
# 配置与工具
# ----------------------------------------------------------------------------


class TestConfigAndUtils(unittest.TestCase):
    def test_config_roundtrip(self):
        cfg = StepMemConfig.from_dict(
            {
                "model": "m",
                "stepping": {"attention_layers": [0, -1], "boundary_threshold": 0.7},
                "retrieval": {"threshold": 0.4, "enabled": False},
                "sampling": {"stop_strings": ["</s>"], "max_new_tokens": 5},
                "store": {"path": None},
            }
        )
        self.assertEqual(cfg.stepping.attention_layers, (0, -1))
        self.assertEqual(cfg.sampling.stop_strings, ("</s>",))
        restored = StepMemConfig.from_dict(cfg.to_dict())
        self.assertEqual(restored, cfg)
        with self.assertRaises(ValueError):
            StepMemConfig.from_dict({"model": "m", "unknown": 1})

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            SteppingConfig(boundary_threshold=1.5)
        with self.assertRaises(ValueError):
            SteppingConfig(min_step_tokens=3, max_step_tokens=2)
        with self.assertRaises(ValueError):
            RetrievalConfig(threshold=-0.1)
        with self.assertRaises(ValueError):
            RetrievalConfig(min_score_margin=1.5)
        with self.assertRaises(ValueError):
            RetrievalConfig(max_inject_tokens=0)
        with self.assertRaises(ValueError):
            SamplingConfig(top_p=0.0)
        with self.assertRaises(ValueError):
            StoreConfig(save_every=0)
        cfg = StoreConfig(
            read_only=True,
            sanitize_special_tokens=False,
            extra_special_tokens=["<x>"],
        )
        self.assertTrue(cfg.read_only)
        self.assertFalse(cfg.sanitize_special_tokens)
        self.assertEqual(cfg.extra_special_tokens, ("<x>",))

    def test_sanitize_memory_text(self):
        tokenizer = ToyTokenizer()
        self.assertEqual(
            sanitize_memory_text("  hi<eos> there<pad>  ", tokenizer),
            "hi there",
        )
        self.assertEqual(
            sanitize_memory_text("hi<eos>", tokenizer, ["<eos>"], strip=False),
            "hi",
        )
        self.assertEqual(
            sanitize_memory_text("a<x>b", tokenizer, ["<x>"], strip=False),
            "ab",
        )
        with self.assertRaises(TypeError):
            sanitize_memory_text(None, tokenizer)

    def test_strip_matched_prefix(self):
        self.assertEqual(strip_matched_prefix("hello world", "h"), "ello world")
        self.assertEqual(strip_matched_prefix(" hello", "h"), "ello")
        self.assertEqual(strip_matched_prefix("hello", " h"), "ello")
        self.assertEqual(strip_matched_prefix("hello", "x"), "hello")
        self.assertEqual(strip_matched_prefix("hello", ""), "hello")

    def test_incremental_decoder(self):
        tokenizer = ToyTokenizer()
        decoder = IncrementalDecoder(tokenizer, skip_special_tokens=False)
        self.assertEqual(decoder.update([1, 2]), "bc")
        self.assertEqual(decoder.update([1, 2, 3]), "d")
        self.assertEqual(decoder.update([1]), "")
        self.assertEqual(decoder.text, "b")
        self.assertEqual(decoder.update([1, 2, 3]), "cd")

    def test_find_stop_string(self):
        self.assertEqual(find_stop_string("hello world", ["world", "hello"]), (0, "hello"))
        self.assertIsNone(find_stop_string("hello", ["zzz"]))
        self.assertEqual(find_stop_string("aaabbb", ["bb", "aaa"]), (0, "aaa"))

    def test_to_jsonable(self):
        self.assertEqual(to_jsonable({"a": np.float32(1.5), "b": (1, 2)}), {"a": 1.5, "b": [1, 2]})
        self.assertEqual(to_jsonable(np.array([[1, 2]])), [[1, 2]])


class TestDataset(unittest.TestCase):
    def test_load_jsonl_and_json(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "data.jsonl"
            jsonl.write_text(
                '{"user":"a","assistant":"b"}\n\n{"prompt":"c","response":"d"}\n',
                encoding="utf-8",
            )
            records = load_memory_dataset(str(jsonl))
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["user"], "a")

            array = Path(tmp) / "data.json"
            array.write_text(
                json.dumps([{"user": "x", "assistant": "y"}], ensure_ascii=False),
                encoding="utf-8",
            )
            self.assertEqual(load_memory_dataset(str(array))[0]["assistant"], "y")

            bad = Path(tmp) / "bad.jsonl"
            bad.write_text('{"user":"a","assistant":"b"}\nnot-json\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_memory_dataset(str(bad))

        self.assertEqual(
            load_memory_dataset({"user": "a", "assistant": "b"})[0]["assistant"], "b"
        )
        self.assertEqual(
            len(load_memory_dataset('[{"user":"a","assistant":"b"}]')), 1
        )
        self.assertEqual(
            len(load_memory_dataset(json.dumps({"user": "a", "assistant": "b"}, indent=2))),
            1,
        )
        with self.assertRaises(FileNotFoundError):
            load_memory_dataset("does-not-exist.jsonl")

    def test_validate_memory_record(self):
        user, assistant, metadata = validate_memory_record(
            {"user": "u", "assistant": "a", "metadata": {"k": 1}}
        )
        self.assertEqual((user, assistant), ("u", "a"))
        self.assertEqual(metadata, {"k": 1})
        user, assistant, metadata = validate_memory_record(
            {"prompt": "p", "response": "r"}
        )
        self.assertEqual((user, assistant, metadata), ("p", "r", {}))
        with self.assertRaises(ValueError):
            validate_memory_record({"user": "", "assistant": "a"})
        with self.assertRaises(ValueError):
            validate_memory_record({"user": "u", "assistant": ""})
        with self.assertRaises(ValueError):
            validate_memory_record({"user": "u", "assistant": "a", "metadata": 1})
        with self.assertRaises(ValueError):
            validate_memory_record(["not", "a", "dict"])


# ----------------------------------------------------------------------------
# 离线端到端（tiny random GPT-2）
# ----------------------------------------------------------------------------


class TestEngineTiny(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _tiny_model()
        cls.tokenizer = ToyTokenizer()

    def test_generation_boundaries_and_storage(self):
        engine = StepMemEngine(_cfg(), model=self.model, tokenizer=self.tokenizer)
        result = engine.complete("abc", sampling={"max_new_tokens": 6, "do_sample": False})

        self.assertEqual(result.stats.generated_tokens, 6)
        self.assertEqual(len(result.steps), 6)  # 阈值 1.0 → 每 token 一步
        # 首个步骤（initial）即使写入模式也不入库
        self.assertEqual(result.stats.stored_steps, 5)
        self.assertEqual(len(engine.store), 5)
        self.assertFalse(result.steps[0].stored)
        self.assertTrue(all(step.stored for step in result.steps[1:]))
        self.assertEqual(result.stats.boundaries, 5)  # 不含初始步骤
        self.assertEqual(result.stats.finish_reason, "max_new_tokens")
        for step in result.steps:
            self.assertTrue(step.text)
            self.assertGreaterEqual(step.end_position, step.start_position)

    def test_retrieval_and_injection(self):
        engine1 = StepMemEngine(_cfg(), model=self.model, tokenizer=self.tokenizer)
        engine1.complete("abc", sampling={"max_new_tokens": 5, "do_sample": False})
        self.assertEqual(len(engine1.store), 4)  # 首个步骤不入库

        cfg2 = _cfg(
            retrieval=RetrievalConfig(
                enabled=True,
                threshold=0.0,
                max_inject_tokens=4,
                retrieve_on_initial_step=False,
                strip_first_token=False,
            )
        )
        engine2 = StepMemEngine(
            cfg2, model=self.model, tokenizer=self.tokenizer, store=engine1.store
        )
        result = engine2.complete("abc", sampling={"max_new_tokens": 3, "do_sample": False})

        self.assertGreaterEqual(len(result.injections), 1)
        accepted = [info for info in result.retrievals if info.accepted]
        self.assertTrue(accepted)
        self.assertEqual(accepted[0].reason, "injected")
        self.assertAlmostEqual(accepted[0].similarity, 1.0, places=5)
        self.assertGreater(result.stats.injected_tokens, 0)
        self.assertIn(result.injections[0].text, result.context_text)

    def test_memory_training_and_special_token_cleaning(self):
        engine = StepMemEngine(_cfg(), model=self.model, tokenizer=self.tokenizer)
        result = engine.train_memory(
            [
                {"user": "abc", "assistant": "hello<eos>"},
                {"user": "xyz", "assistant": "world<pad>"},
                {"user": "empty", "assistant": "<eos>"},
            ],
            sampling={"do_sample": False},
        )
        self.assertEqual(result.stats.total, 3)
        self.assertEqual(result.stats.processed, 2)
        self.assertEqual(result.stats.stored, 2)
        self.assertEqual(result.stats.skipped_empty, 1)
        self.assertEqual(result.stats.failed, 0)
        self.assertEqual(
            sorted(record.text for record in engine.store.records),
            ["hello", "world"],
        )
        for record in engine.store.records:
            self.assertEqual(record.metadata["source"], "memory_train")
            self.assertTrue(record.session_id.startswith("train-"))
            self.assertEqual(record.text.find("<eos>"), -1)
        self.assertGreater(result.stats.tokens_stored, 0)
        self.assertEqual(engine.clean_memory_text(" hi<eos> there ", strip=True), "hi there")
        self.assertEqual(engine.clean_memory_text(" hi<eos>"), " hi")

    def test_read_only_store_rejects_training(self):
        store = VectorStore(32, StoreConfig(read_only=True))
        engine = StepMemEngine(
            _cfg(), model=self.model, tokenizer=self.tokenizer, store=store
        )
        with self.assertRaises(StoreError):
            engine.train_memory([{"user": "a", "assistant": "b"}])

    def test_read_only_retrieval_does_not_write(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            engine1 = StepMemEngine(
                _cfg(store=StoreConfig(path=tmp)),
                model=self.model,
                tokenizer=self.tokenizer,
            )
            engine1.complete(
                "abc", sampling={"max_new_tokens": 3, "do_sample": False}
            )
            count = len(engine1.store)
            self.assertGreater(count, 0)
            engine1.close()

            cfg2 = _cfg(
                store=StoreConfig(path=tmp, read_only=True),
                retrieval=RetrievalConfig(
                    enabled=True, threshold=0.0, strip_first_token=False
                ),
            )
            engine2 = StepMemEngine(cfg2, model=self.model, tokenizer=self.tokenizer)
            self.assertEqual(len(engine2.store), count)
            result = engine2.complete(
                "abc", sampling={"max_new_tokens": 2, "do_sample": False}
            )
            self.assertGreaterEqual(len(result.injections), 1)
            self.assertEqual(result.stats.stored_steps, 0)
            self.assertEqual(len(engine2.store), count)
            engine2.close()

            reloaded = VectorStore.load(tmp, dim=32)
            self.assertEqual(len(reloaded), count)

    def test_injection_appears_in_output(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            engine1 = StepMemEngine(
                _cfg(store=StoreConfig(path=tmp)),
                model=self.model,
                tokenizer=self.tokenizer,
            )
            trained = engine1.train_memory(
                [{"user": "abc", "assistant": "helloworld"}],
                sampling={"do_sample": False},
            )
            self.assertEqual(trained.stats.stored, 1)
            engine1.close()

            cfg2 = _cfg(
                store=StoreConfig(path=tmp, read_only=True),
                retrieval=RetrievalConfig(
                    enabled=True,
                    threshold=0.0,
                    max_inject_tokens=16,
                    exclude_injected=True,
                ),
            )
            engine2 = StepMemEngine(cfg2, model=self.model, tokenizer=self.tokenizer)
            events = list(
                engine2.stream_chat(
                    "abc", sampling={"max_new_tokens": 3, "do_sample": False}
                )
            )
            injections = [
                event for event in events if isinstance(event, InjectionEvent)
            ]
            done = [event for event in events if isinstance(event, DoneEvent)]
            self.assertEqual(len(done), 1)
            self.assertEqual(len(injections), 1)
            result = done[0].result
            self.assertIn("helloworld", result.text)
            self.assertIn(injections[0].text, result.text)
            self.assertEqual(result.stats.injections, 1)
            self.assertGreater(result.stats.injected_tokens, 0)
            self.assertEqual(result.stats.stored_steps, 0)
            self.assertEqual(result.stats.steps, 0)
            self.assertEqual(result.steps, [])
            self.assertEqual(len(engine2.store), 1)
            accepted = [info for info in result.retrievals if info.accepted]
            self.assertTrue(accepted)
            self.assertEqual(accepted[0].reason, "injected")
            self.assertFalse(
                any(
                    type(event).__name__ in {"StepStartEvent", "StepEndEvent"}
                    for event in events
                )
            )

    def test_ambiguous_recall_is_rejected(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            engine1 = StepMemEngine(
                _cfg(store=StoreConfig(path=tmp)),
                model=self.model,
                tokenizer=self.tokenizer,
            )
            trained = engine1.train_memory(
                [
                    {"user": "abc", "assistant": "first"},
                    {"user": "abc", "assistant": "second"},
                ],
                sampling={"do_sample": False},
            )
            self.assertEqual(trained.stats.stored, 2)
            engine1.close()

            cfg2 = _cfg(
                store=StoreConfig(path=tmp, read_only=True),
                retrieval=RetrievalConfig(
                    enabled=True, threshold=0.0, min_score_margin=0.5
                ),
            )
            engine2 = StepMemEngine(cfg2, model=self.model, tokenizer=self.tokenizer)
            result = engine2.chat(
                "abc", sampling={"max_new_tokens": 2, "do_sample": False}
            )
            self.assertEqual(result.stats.injections, 0)
            self.assertEqual(len(result.injections), 0)
            self.assertEqual(result.stats.steps, 0)
            self.assertTrue(result.retrievals)
            self.assertFalse(result.retrievals[0].accepted)
            self.assertEqual(result.retrievals[0].reason, "ambiguous")

    def test_calibration_report_offline(self):
        from stepmem.calibrate import calibrate

        engine = StepMemEngine(_cfg(), model=self.model, tokenizer=self.tokenizer)
        result = calibrate(
            engine, ["abc", "xyz"], max_new_tokens=4, target_step_tokens=2
        )
        report = result.report
        self.assertEqual(report["attention_similarity"]["count"], 6)
        self.assertIsNotNone(report["suggested_boundary_threshold"])
        self.assertIsNotNone(report["suggested_retrieval_threshold"])
        self.assertEqual(report["hidden_within_prompt"]["count"], 12)
        self.assertEqual(report["hidden_cross_prompt"]["count"], 16)

    def test_engine_loads_hf_checkpoint_from_disk(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self.model.save_pretrained(tmp)
            engine = StepMemEngine(_cfg(model=tmp), tokenizer=self.tokenizer)
            result = engine.complete(
                "abc", sampling={"max_new_tokens": 2, "do_sample": False}
            )
            self.assertEqual(result.stats.generated_tokens, 2)
            self.assertEqual(engine.model_id, tmp)

    def test_read_only_mode_uses_optimized_attention_and_per_token_retrieval(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            memory_dir = root / "memory"
            self.model.save_pretrained(model_dir)

            # 写入模式：eager + 步骤切分，用于训练记忆
            engine1 = StepMemEngine(
                _cfg(
                    model=str(model_dir),
                    store=StoreConfig(path=str(memory_dir)),
                ),
                tokenizer=self.tokenizer,
            )
            self.assertEqual(engine1.model.config._attn_implementation, "eager")
            trained = engine1.train_memory(
                [{"user": "abc", "assistant": "helloworld"}],
                sampling={"do_sample": False},
            )
            self.assertEqual(trained.stats.stored, 1)
            engine1.close()

            # 非写入模式：保留 SDPA/FlashAttention，不切分步骤，每 token 检索
            cfg2 = _cfg(
                model=str(model_dir),
                store=StoreConfig(path=str(memory_dir), read_only=True),
                retrieval=RetrievalConfig(
                    enabled=True, threshold=0.0, max_inject_tokens=16
                ),
            )
            engine2 = StepMemEngine(cfg2, tokenizer=self.tokenizer)
            self.assertFalse(engine2.write_mode)
            self.assertNotEqual(
                engine2.model.config._attn_implementation, "eager"
            )
            with self.assertRaises(StepMemError):
                engine2.profile(prompt="abc", max_new_tokens=1)

            events = list(
                engine2.stream_chat(
                    "abc", sampling={"max_new_tokens": 3, "do_sample": False}
                )
            )
            result = [e for e in events if isinstance(e, DoneEvent)][0].result

            self.assertEqual(result.stats.steps, 0)
            self.assertEqual(result.steps, [])
            self.assertEqual(result.stats.generated_tokens, 3)
            self.assertEqual(len(result.retrievals), 3)  # 每个生成 token 都检索
            self.assertGreaterEqual(len(result.injections), 1)
            self.assertIn("helloworld", result.text)
            self.assertFalse(
                any(
                    type(event).__name__ in {"StepStartEvent", "StepEndEvent"}
                    for event in events
                )
            )

    def test_repeat_injection_is_configurable(self):
        """exclude_injected 开/关决定同一条记忆能否在一次生成中重复注入。"""
        import tempfile

        from stepmem.engine import _GenerationSession

        with tempfile.TemporaryDirectory() as tmp:
            engine1 = StepMemEngine(
                _cfg(store=StoreConfig(path=tmp)),
                model=self.model,
                tokenizer=self.tokenizer,
            )
            trained = engine1.train_memory(
                [{"user": "abc", "assistant": "helloworld"}],
                sampling={"do_sample": False},
            )
            self.assertEqual(trained.stats.stored, 1)
            key = engine1.store._keys[0].copy()
            engine1.close()

            for exclude_injected, expected in ((False, 2), (True, 1)):
                cfg = _cfg(
                    store=StoreConfig(path=tmp, read_only=True),
                    retrieval=RetrievalConfig(
                        enabled=True,
                        threshold=0.0,
                        max_inject_tokens=16,
                        exclude_injected=exclude_injected,
                    ),
                )
                engine = StepMemEngine(
                    cfg, model=self.model, tokenizer=self.tokenizer
                )
                try:
                    session = _GenerationSession(
                        engine,
                        self.tokenizer.encode("abc"),
                        "abc",
                        SamplingConfig(max_new_tokens=1, do_sample=False),
                    )
                    # 用同一个查询键连续检索两次：等价于逐 token 检索中
                    # 连续两个 token 都命中同一条记忆。
                    events = []
                    events.extend(session._retrieve_and_inject(None, key, 0, 7))
                    events.extend(session._retrieve_and_inject(None, key, 0, 7))
                    injections = [
                        e for e in events if isinstance(e, InjectionEvent)
                    ]
                    self.assertEqual(
                        len(injections),
                        expected,
                        msg=f"exclude_injected={exclude_injected}",
                    )
                    if exclude_injected:
                        rejected = [
                            e
                            for e in events
                            if isinstance(e, RetrievalEvent) and not e.accepted
                        ]
                        self.assertEqual(len(rejected), 1)
                        self.assertEqual(rejected[0].reason, "no_candidates")
                finally:
                    engine.close()

    def test_stop_string_is_not_stored(self):
        from unittest.mock import patch

        engine = StepMemEngine(_cfg(), model=self.model, tokenizer=self.tokenizer)
        scripted = self.tokenizer.encode("abSTOPcd")
        with patch.object(Sampler, "sample", side_effect=scripted):
            result = engine.complete(
                "abc",
                sampling={
                    "max_new_tokens": 10,
                    "do_sample": False,
                    "stop_strings": ["STOP"],
                },
            )
        self.assertEqual(result.text, "ab")
        self.assertEqual(result.stats.finish_reason, "stop_string")
        self.assertEqual(result.stats.generated_tokens, 6)  # a b S T O P
        # b/S/T 成步并存储；首个步骤 a 按规则不入库；最后一步 O 因停止串前缀被裁空。
        self.assertEqual(len(engine.store), 3)
        self.assertFalse(result.steps[0].stored)
        self.assertEqual(result.steps[-1].text, "")

    def test_eos_stops_without_entering_step(self):
        from unittest.mock import patch

        engine = StepMemEngine(_cfg(), model=self.model, tokenizer=self.tokenizer)
        first = self.tokenizer.encode("a")[0]
        with patch.object(Sampler, "sample", side_effect=[first, 5]):
            result = engine.complete(
                "abc",
                sampling={
                    "max_new_tokens": 3,
                    "do_sample": False,
                    "eos_token_ids": (5,),
                },
            )
        self.assertEqual(result.text, "a")
        self.assertEqual(result.stats.finish_reason, "eos")
        self.assertEqual(result.stats.generated_tokens, 1)  # eos 被采样但不计入/不进入上下文
        self.assertEqual(len(engine.store), 0)  # 首个步骤（initial）不入库
        self.assertFalse(result.steps[0].stored)

    def test_chat_template_and_profile(self):
        engine = StepMemEngine(_cfg(), model=self.model, tokenizer=self.tokenizer)
        result = engine.chat("hello", sampling={"max_new_tokens": 2, "do_sample": False})
        self.assertIn("<user>hello", result.prompt_text)
        self.assertIn("<assistant>", result.prompt_text)
        self.assertEqual(result.stats.generated_tokens, 2)

        profile = engine.profile(
            prompt="abc", max_new_tokens=4, sampling={"do_sample": False}
        )
        self.assertEqual(len(profile.token_ids), 4)
        self.assertEqual(len(profile.attention_similarities), 4)
        self.assertIsNone(profile.attention_similarities[0])
        self.assertEqual(profile.hidden_states.shape, (4, 32))
        self.assertTrue(all(s is not None for s in profile.attention_similarities[1:]))

    def test_context_limit(self):
        cfg = _cfg(max_context_tokens=5)
        engine = StepMemEngine(cfg, model=self.model, tokenizer=self.tokenizer)
        result = engine.complete("abc", sampling={"max_new_tokens": 10, "do_sample": False})
        self.assertEqual(result.stats.finish_reason, "context_limit")
        self.assertEqual(result.stats.generated_tokens, 2)

    def test_store_persistence_through_engine(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            engine = StepMemEngine(
                _cfg(store=StoreConfig(path=tmp)), model=self.model, tokenizer=self.tokenizer
            )
            engine.complete("abc", sampling={"max_new_tokens": 3, "do_sample": False})
            engine.close()
            self.assertTrue((Path(tmp) / "manifest.json").exists())

            engine2 = StepMemEngine(
                _cfg(store=StoreConfig(path=tmp)), model=self.model, tokenizer=self.tokenizer
            )
            self.assertEqual(len(engine2.store), 2)  # 3 token 中首个步骤不入库
            engine2.close()


if __name__ == "__main__":
    unittest.main()
