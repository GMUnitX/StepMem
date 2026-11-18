"""向量数据库：余弦相似度检索 + 步骤原文存储 + 磁盘持久化。

实现要点
--------
* 键在写入时做 L2 归一化，检索即内积（等价余弦相似度）。
* 默认使用预分配的 NumPy 矩阵，检索走 BLAS；也支持可选的 FAISS 后端。
* 记录采用 JSONL，键采用 .npy，manifest.json 作为提交标记；保存为原子替换。
* 无删除操作，因此记录 id 单调递增，便于排除“刚写入的记忆”。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from .config import StoreConfig
from .errors import DimensionMismatchError, StoreError
from .types import MemoryRecord, RetrievalHit

LOGGER = logging.getLogger("stepmem")

_FORMAT = "stepmem-vector-store"
_VERSION = 1

try:  # FAISS 是可选依赖
    import faiss as _faiss  # type: ignore
except Exception:  # pragma: no cover - 取决于环境
    _faiss = None


def _normalize_key(key, dim: int, *, name: str = "key") -> np.ndarray:
    arr = np.asarray(key, dtype=np.float32).reshape(-1)
    if arr.size != dim:
        raise DimensionMismatchError(
            f"{name} 维度为 {arr.size}，索引维度为 {dim}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 包含 NaN/Inf")
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        raise ValueError(f"{name} 的 L2 范数为 0，无法用于余弦检索")
    return (arr / norm).astype(np.float32, copy=False)


class VectorStore:
    """步骤记忆的向量索引。"""

    def __init__(
        self,
        dim: int,
        config: Optional[StoreConfig] = None,
        *,
        model_id: Optional[str] = None,
    ) -> None:
        if int(dim) <= 0:
            raise ValueError("dim 必须 > 0")
        self.dim = int(dim)
        self.config = config or StoreConfig()
        self.model_id = model_id
        self._lock = threading.RLock()
        self._records: list[MemoryRecord] = []
        self._capacity = 64
        self._keys = np.empty((self._capacity, self.dim), dtype=np.float32)
        self._count = 0
        self._next_id = 1
        self._id_to_idx: dict[int, int] = {}
        self._created_at = time.time()
        self._updated_at = self._created_at
        self._adds_since_save = 0
        self._autosave_broken = False
        self._backend = self._resolve_backend()
        self._faiss_index = None
        if self._backend == "faiss":
            self._init_faiss()

    # ------------------------------------------------------------------ 基础信息

    def __len__(self) -> int:
        return self._count

    @property
    def records(self) -> tuple[MemoryRecord, ...]:
        with self._lock:
            return tuple(self._records)

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def read_only(self) -> bool:
        """只读模式：允许加载/检索，禁止写入、清空和保存。"""
        return bool(self.config.read_only)

    @property
    def path(self) -> Optional[str]:
        return self.config.path

    def _resolve_backend(self) -> str:
        backend = (self.config.backend or "numpy").lower()
        if backend == "faiss":
            if _faiss is None:
                raise ImportError(
                    "store.backend='faiss' 需要安装 faiss-cpu 或 faiss-gpu；"
                    "也可以使用默认的 'numpy' 后端"
                )
            return "faiss"
        return "numpy"

    def _init_faiss(self) -> None:
        assert _faiss is not None
        self._faiss_index = _faiss.IndexFlatIP(self.dim)

    # ------------------------------------------------------------------ 写入

    def add(
        self,
        key,
        text: str,
        *,
        session_id: str = "",
        step_index: int = 0,
        token_count: int = 0,
        metadata: Optional[dict[str, Any]] = None,
        autosave: bool = True,
    ) -> MemoryRecord:
        """写入一条（存储键，步骤原文）记忆并返回记录。

        ``autosave=False`` 用于批量训练：避免每 N 条重写整个索引，由调用方在
        批量结束后显式 ``save()``。
        """
        if self.read_only:
            raise StoreError("记忆库处于只读模式，禁止写入；请设置 store.read_only=False")
        if not isinstance(text, str) or not text:
            raise ValueError("记忆文本不能为空")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise TypeError("metadata 必须是 dict")
        try:
            json.dumps(metadata, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise TypeError("metadata 必须可 JSON 序列化") from exc
        normalized = _normalize_key(key, self.dim)

        with self._lock:
            if self._count == self._capacity:
                self._grow(max(64, self._capacity * 2))
            idx = self._count
            self._keys[idx] = normalized
            record = MemoryRecord(
                id=self._next_id,
                text=text,
                session_id=str(session_id),
                step_index=int(step_index),
                token_count=int(token_count),
                created_at=time.time(),
                metadata=dict(metadata),
            )
            self._records.append(record)
            self._id_to_idx[record.id] = idx
            self._next_id += 1
            self._count += 1
            self._updated_at = record.created_at
            if self._backend == "faiss":
                assert self._faiss_index is not None
                self._faiss_index.add(normalized.reshape(1, -1))
            self._adds_since_save += 1
            if autosave:
                self._maybe_autosave()
            return record

    def _grow(self, new_capacity: int) -> None:
        new_keys = np.empty((new_capacity, self.dim), dtype=np.float32)
        new_keys[: self._count] = self._keys[: self._count]
        self._keys = new_keys
        self._capacity = new_capacity

    def _maybe_autosave(self) -> None:
        cfg = self.config
        if self.read_only or not cfg.autosave or not cfg.path or self._autosave_broken:
            return
        if self._adds_since_save < cfg.save_every:
            return
        try:
            self.save()
        except Exception as exc:  # 自动保存失败不应中断生成
            self._autosave_broken = True
            LOGGER.error("StepMem 自动保存失败，已暂停自动保存: %s", exc)

    # ------------------------------------------------------------------ 检索

    def search(
        self,
        query,
        top_k: int = 1,
        min_similarity: float = 0.0,
        *,
        exclude_ids: Optional[Iterable[int]] = None,
        exclude_session: Optional[str] = None,
    ) -> list[RetrievalHit]:
        """检索 top_k 条记忆，按余弦相似度降序。

        ``min_similarity`` 为包含式下界；严格大于阈值的判定由引擎层执行。
        ``exclude_ids`` / ``exclude_session`` 在排序前生效。
        """
        top_k = int(top_k)
        if top_k < 1:
            raise ValueError("top_k 必须 >= 1")
        q = _normalize_key(query, self.dim, name="query")
        excluded_ids = {int(i) for i in exclude_ids} if exclude_ids else set()

        with self._lock:
            n = self._count
            if n == 0:
                return []

            excluded_idx: set[int] = set()
            for rid in excluded_ids:
                idx = self._id_to_idx.get(rid)
                if idx is not None:
                    excluded_idx.add(idx)
            if exclude_session is not None:
                for idx, rec in enumerate(self._records):
                    if rec.session_id == exclude_session:
                        excluded_idx.add(idx)

            if self._backend == "faiss":
                scores, ids = self._search_faiss(q, top_k, excluded_idx, float(min_similarity))
                return [
                    RetrievalHit(self._records[int(i)], float(s))
                    for s, i in zip(scores, ids)
                ]

            scores = self._keys[:n] @ q  # (n,)
            mask = np.ones(n, dtype=bool)
            if excluded_idx:
                mask[np.fromiter(excluded_idx, dtype=np.int64)] = False
            valid = np.flatnonzero(mask)
            if valid.size == 0:
                return []
            valid_scores = scores[valid]
            k = min(top_k, valid.size)
            if k == valid.size:
                order_local = np.argsort(-valid_scores)
            else:
                part = np.argpartition(-valid_scores, k - 1)[:k]
                order_local = part[np.argsort(-valid_scores[part])]
            hits: list[RetrievalHit] = []
            for local in order_local:
                sim = float(valid_scores[local])
                if sim < min_similarity:
                    break  # 已按降序排列
                hits.append(RetrievalHit(self._records[int(valid[local])], sim))
            return hits

    def _search_faiss(
        self,
        q: np.ndarray,
        top_k: int,
        excluded_idx: set[int],
        min_similarity: float,
    ) -> tuple[list[float], list[int]]:
        assert self._faiss_index is not None
        n = self._count
        fetch = min(n, max(top_k * 4, top_k + len(excluded_idx) + 8))
        while True:
            sims, ids = self._faiss_index.search(q.reshape(1, -1), fetch)
            kept_scores: list[float] = []
            kept_ids: list[int] = []
            for sim, idx in zip(sims[0], ids[0]):
                if idx < 0:
                    continue
                if int(idx) in excluded_idx:
                    continue
                if float(sim) < min_similarity:
                    continue
                kept_scores.append(float(sim))
                kept_ids.append(int(idx))
                if len(kept_ids) >= top_k:
                    return kept_scores, kept_ids
            if fetch >= n:
                return kept_scores, kept_ids
            fetch = min(n, fetch * 2)

    def get(self, record_id: int) -> Optional[MemoryRecord]:
        with self._lock:
            idx = self._id_to_idx.get(int(record_id))
            return None if idx is None else self._records[idx]

    # ------------------------------------------------------------------ 生命周期

    def clear(self) -> None:
        """清空内存中的索引（不自动落盘）。"""
        if self.read_only:
            raise StoreError("记忆库处于只读模式，禁止清空")
        with self._lock:
            self._records.clear()
            self._id_to_idx.clear()
            self._keys = np.empty((64, self.dim), dtype=np.float32)
            self._capacity = 64
            self._count = 0
            self._next_id = 1
            self._adds_since_save = 0
            self._autosave_broken = False
            self._updated_at = time.time()
            if self._backend == "faiss":
                self._init_faiss()

    def save(self, path: Optional[str] = None) -> Path:
        """原子地保存到目录：records.jsonl + keys.npy + manifest.json。"""
        if self.read_only:
            raise StoreError("记忆库处于只读模式，禁止保存")
        with self._lock:
            target = Path(path or self.config.path or "")
            if not str(target):
                raise StoreError("未配置 store.path，无法保存")
            target.mkdir(parents=True, exist_ok=True)

            keys = self._keys[: self._count].copy()
            records_tmp = target / "records.jsonl.tmp"
            keys_tmp = target / "keys.npy.tmp"
            manifest_tmp = target / "manifest.json.tmp"

            with open(records_tmp, "w", encoding="utf-8", newline="\n") as fh:
                for rec in self._records:
                    json.dump(
                        {
                            "id": rec.id,
                            "text": rec.text,
                            "session_id": rec.session_id,
                            "step_index": rec.step_index,
                            "token_count": rec.token_count,
                            "created_at": rec.created_at,
                            "metadata": rec.metadata,
                        },
                        fh,
                        ensure_ascii=False,
                    )
                    fh.write("\n")

            with open(keys_tmp, "wb") as fh:
                np.save(fh, keys, allow_pickle=False)

            manifest = {
                "format": _FORMAT,
                "version": _VERSION,
                "dim": self.dim,
                "count": self._count,
                "model_id": self.model_id,
                "backend": self._backend,
                "dtype": "float32",
                "created_at": self._created_at,
                "updated_at": time.time(),
            }
            with open(manifest_tmp, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, ensure_ascii=False, indent=2)

            os.replace(records_tmp, target / "records.jsonl")
            os.replace(keys_tmp, target / "keys.npy")
            os.replace(manifest_tmp, target / "manifest.json")

            self.config.path = str(target)
            self._adds_since_save = 0
            self._autosave_broken = False
            self._updated_at = manifest["updated_at"]
            LOGGER.debug("StepMem 记忆已保存: %s (%d 条)", target, self._count)
            return target

    @classmethod
    def open(
        cls,
        path: str,
        dim: int,
        config: Optional[StoreConfig] = None,
        *,
        model_id: Optional[str] = None,
    ) -> "VectorStore":
        """目录存在则加载，否则创建空索引（不立即写盘）。"""
        p = Path(path)
        if (p / "manifest.json").exists():
            return cls.load(p, dim=dim, config=config, model_id=model_id)
        cfg = replace(config or StoreConfig(), path=str(p))
        return cls(dim, cfg, model_id=model_id)

    @classmethod
    def load(
        cls,
        path: str,
        dim: Optional[int] = None,
        config: Optional[StoreConfig] = None,
        *,
        model_id: Optional[str] = None,
    ) -> "VectorStore":
        p = Path(path)
        manifest_path = p / "manifest.json"
        records_path = p / "records.jsonl"
        keys_path = p / "keys.npy"
        if not manifest_path.exists():
            raise StoreError(f"目录中没有 manifest.json: {p}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise StoreError(f"manifest.json 读取失败: {exc}") from exc
        if manifest.get("format") != _FORMAT:
            raise StoreError(f"不是 StepMem 记忆目录: {p}")
        if int(manifest.get("version", 0)) != _VERSION:
            raise StoreError(
                f"记忆格式版本不兼容: {manifest.get('version')} != {_VERSION}"
            )
        file_dim = int(manifest.get("dim", 0))
        if file_dim <= 0:
            raise StoreError("manifest.json 缺少有效 dim")
        if dim is not None and int(dim) != file_dim:
            raise DimensionMismatchError(
                f"当前模型隐藏维度 {dim} 与记忆索引维度 {file_dim} 不一致；"
                "请使用同一模型或更换 store.path"
            )
        if not records_path.exists() or not keys_path.exists():
            raise StoreError(f"记忆目录缺少 records.jsonl 或 keys.npy: {p}")

        cfg = replace(config or StoreConfig(), path=str(p))
        store = cls(file_dim, cfg, model_id=model_id or manifest.get("model_id"))
        saved_model = manifest.get("model_id")
        if model_id and saved_model and str(model_id) != str(saved_model):
            LOGGER.warning(
                "记忆索引由模型 %r 写入，当前模型为 %r；键空间可能不兼容",
                saved_model,
                model_id,
            )

        records: list[MemoryRecord] = []
        line_no = 0
        try:
            with open(records_path, "r", encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    raw = json.loads(line)
                    records.append(
                        MemoryRecord(
                            id=int(raw["id"]),
                            text=str(raw["text"]),
                            session_id=str(raw.get("session_id", "")),
                            step_index=int(raw.get("step_index", 0)),
                            token_count=int(raw.get("token_count", 0)),
                            created_at=float(raw.get("created_at", 0.0)),
                            metadata=dict(raw.get("metadata") or {}),
                        )
                    )
        except Exception as exc:
            raise StoreError(f"records.jsonl 读取失败（第 {line_no} 行）：{exc}") from exc

        try:
            keys = np.load(keys_path, allow_pickle=False)
        except Exception as exc:
            raise StoreError(f"keys.npy 读取失败: {exc}") from exc
        if keys.ndim != 2 or keys.shape[1] != file_dim:
            raise StoreError(
                f"keys.npy 形状 {keys.shape} 与 dim={file_dim} 不一致"
            )
        if keys.shape[0] != len(records):
            raise StoreError(
                f"keys.npy 条数 {keys.shape[0]} 与 records 条数 {len(records)} 不一致"
            )
        if len(records) > 0:
            keys = keys.astype(np.float32, copy=False)
            norms = np.linalg.norm(keys, axis=1)
            if not np.all(np.isfinite(norms)) or np.any(norms < 1e-6):
                raise StoreError("keys.npy 包含非法向量（NaN/Inf 或零向量）")
            # 防御性重新归一化，保证旧文件/手工文件也按余弦检索。
            keys = keys / norms[:, None]

        count = len(records)
        capacity = max(64, 1 << max(0, (count - 1).bit_length()))
        store._records = records
        store._count = count
        store._capacity = capacity
        store._keys = np.empty((capacity, file_dim), dtype=np.float32)
        if count:
            store._keys[:count] = keys
        store._next_id = max((r.id for r in records), default=0) + 1
        store._id_to_idx = {rec.id: idx for idx, rec in enumerate(records)}
        store._created_at = float(manifest.get("created_at", time.time()))
        store._updated_at = float(manifest.get("updated_at", store._created_at))
        if store._backend == "faiss" and count:
            assert store._faiss_index is not None
            store._faiss_index.add(store._keys[:count])
        return store

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"VectorStore(dim={self.dim}, count={self._count}, "
            f"backend={self._backend!r}, read_only={self.read_only}, "
            f"path={self.config.path!r})"
        )
