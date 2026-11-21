"""VectorStore interface and a local JSON-file adapter.

pgvector and Chroma adapters plug in behind the same interface later.

Filters are dicts of metadata key -> value, where value may be a plain value
(equality), {"$ne": v}, {"$in": [..]} or {"$nin": [..]}.
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.kb.embeddings import cosine

Filter = dict[str, Any]


@dataclass
class VectorRecord:
    id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchHit:
    record: VectorRecord
    score: float


def matches(metadata: dict[str, Any], flt: Filter | None) -> bool:
    for key, cond in (flt or {}).items():
        val = metadata.get(key)
        if isinstance(cond, dict):
            if "$ne" in cond and val == cond["$ne"]:
                return False
            if "$in" in cond and val not in cond["$in"]:
                return False
            if "$nin" in cond and val in cond["$nin"]:
                return False
        elif val != cond:
            return False
    return True


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, records: list[VectorRecord]) -> None: ...

    @abstractmethod
    def search(self, query_embedding: list[float], flt: Filter | None, top_k: int) -> list[SearchHit]: ...

    @abstractmethod
    def list(self, flt: Filter | None = None) -> list[VectorRecord]: ...

    @abstractmethod
    def delete(self, flt: Filter) -> int: ...

    def delete_by_version(self, entity_id: str, keep_version: str) -> int:
        """Remove chunks of an entity that belong to any other KB version."""
        return self.delete({"entity_id": entity_id, "kb_version": {"$ne": keep_version}})


class LocalVectorStore(VectorStore):
    """Brute-force cosine search over records kept in memory and persisted to one JSON file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._records: dict[str, VectorRecord] = {}
        if path.exists():
            for raw in json.loads(path.read_text(encoding="utf-8")):
                rec = VectorRecord(**raw)
                self._records[rec.id] = rec

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps([asdict(r) for r in self._records.values()], ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def upsert(self, records: list[VectorRecord]) -> None:
        with self._lock:
            for r in records:
                self._records[r.id] = r
            self._save()

    def search(self, query_embedding: list[float], flt: Filter | None, top_k: int) -> list[SearchHit]:
        hits = [
            SearchHit(r, cosine(query_embedding, r.embedding))
            for r in self._records.values()
            if matches(r.metadata, flt)
        ]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def list(self, flt: Filter | None = None) -> list[VectorRecord]:
        return [r for r in self._records.values() if matches(r.metadata, flt)]

    def delete(self, flt: Filter) -> int:
        with self._lock:
            ids = [rid for rid, r in self._records.items() if matches(r.metadata, flt)]
            for rid in ids:
                del self._records[rid]
            if ids:
                self._save()
            return len(ids)

    def __len__(self) -> int:
        return len(self._records)


def make_vector_store(settings) -> VectorStore:
    kind = settings.vector_store
    if kind in ("local", ""):
        return LocalVectorStore(settings.data_path / "vectors.json")  # runtime data, not in git
    if kind == "chroma":
        from app.kb.vectorstore_chroma import ChromaVectorStore, collection_name

        return ChromaVectorStore.connect(
            api_key=settings.chroma_api_key,
            tenant=settings.chroma_tenant,
            database=settings.chroma_database,
            collection=collection_name(settings.chroma_collection, settings.embedding_model),
            host=settings.chroma_host,
        )
    raise ValueError(f"Vector store '{kind}' is not supported (use local or chroma)")
