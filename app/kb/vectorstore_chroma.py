"""Chroma (Cloud or self-hosted) adapter for the VectorStore interface.

We supply embeddings ourselves (embedding_function=None) and use cosine distance, so
score = 1 - distance. One collection per embedding model: changing EMBEDDING_MODEL
writes to a fresh collection instead of mixing vector sizes.

`list()` results are cached for a short time because the KM tools read the small catalog
on every request; writes from this process clear the cache.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any

from app.kb.vectorstore import Filter, SearchHit, VectorRecord, VectorStore

PAGE = 300  # Chroma Cloud's per-request record limit


def to_where(flt: Filter | None) -> dict[str, Any] | None:
    """Translate our filter dict to a Chroma `where` clause."""
    clauses: list[dict[str, Any]] = []
    for key, cond in (flt or {}).items():
        if isinstance(cond, dict):
            for op in ("$ne", "$in", "$nin"):
                if op in cond:
                    clauses.append({key: {op: cond[op]}})
        else:
            clauses.append({key: {"$eq": cond}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def clean_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Chroma metadata values must be str, int, float or bool (no None, no lists)."""
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        out[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
    return out


def collection_name(base: str, embedding_model: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", embedding_model).strip("-").lower()
    name = f"{base}__{slug}"[:63].strip("-_")
    return name


class ChromaVectorStore(VectorStore):
    def __init__(self, collection: Any, cache_ttl_seconds: float = 60.0) -> None:
        self.collection = collection
        self.cache_ttl = cache_ttl_seconds
        self._cache: dict[str, tuple[float, list[VectorRecord]]] = {}
        self._lock = threading.Lock()

    @classmethod
    def connect(
        cls,
        *,
        api_key: str,
        tenant: str,
        database: str,
        collection: str,
        host: str = "",
    ) -> "ChromaVectorStore":
        import chromadb

        if host:
            client = chromadb.HttpClient(host=host, tenant=tenant, database=database,
                                         headers={"x-chroma-token": api_key} if api_key else None)
        else:
            if not (api_key and tenant and database):
                raise ValueError("CHROMA_API_KEY, CHROMA_TENANT and CHROMA_DATABASE are required for VECTOR_STORE=chroma")
            client = chromadb.CloudClient(tenant=tenant, database=database, api_key=api_key)
        col = client.get_or_create_collection(
            collection, embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
        )
        return cls(col)

    # ---- writes ----------------------------------------------------------
    def _invalidate(self) -> None:
        with self._lock:
            self._cache.clear()

    def upsert(self, records: list[VectorRecord]) -> None:
        for i in range(0, len(records), PAGE):
            batch = records[i:i + PAGE]
            self.collection.upsert(
                ids=[r.id for r in batch],
                embeddings=[r.embedding for r in batch],
                documents=[r.text for r in batch],
                metadatas=[clean_metadata(r.metadata) for r in batch],
            )
        self._invalidate()

    def delete(self, flt: Filter) -> int:
        ids = [r.id for r in self._fetch(flt, with_embeddings=False)]
        for i in range(0, len(ids), PAGE):
            self.collection.delete(ids=ids[i:i + PAGE])
        if ids:
            self._invalidate()
        return len(ids)

    # ---- reads -----------------------------------------------------------
    def search(self, query_embedding: list[float], flt: Filter | None, top_k: int) -> list[SearchHit]:
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=to_where(flt),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for rid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append(SearchHit(VectorRecord(id=rid, text=doc or "", embedding=[], metadata=dict(meta or {})), 1.0 - dist))
        return hits

    def list(self, flt: Filter | None = None) -> list[VectorRecord]:
        key = repr(sorted((flt or {}).items(), key=lambda kv: kv[0]))
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(key)
            if cached and now - cached[0] < self.cache_ttl:
                return cached[1]
        records = self._fetch(flt, with_embeddings=True)
        with self._lock:
            self._cache[key] = (now, records)
        return records

    def _fetch(self, flt: Filter | None, with_embeddings: bool) -> list[VectorRecord]:
        include = ["documents", "metadatas"] + (["embeddings"] if with_embeddings else [])
        out: list[VectorRecord] = []
        offset = 0
        while True:
            res = self.collection.get(where=to_where(flt), include=include, limit=PAGE, offset=offset)
            ids = res["ids"]
            embs = res.get("embeddings") if with_embeddings else None
            for i, rid in enumerate(ids):
                out.append(
                    VectorRecord(
                        id=rid,
                        text=(res["documents"][i] or "") if res.get("documents") else "",
                        embedding=[float(x) for x in embs[i]] if embs is not None else [],
                        metadata=dict(res["metadatas"][i] or {}),
                    )
                )
            if len(ids) < PAGE:
                return out
            offset += PAGE
