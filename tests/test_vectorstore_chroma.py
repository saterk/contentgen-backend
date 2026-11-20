"""Chroma adapter against an in-memory fake collection (no network)."""

from __future__ import annotations

import math

from app.kb.vectorstore import VectorRecord, matches
from app.kb.vectorstore_chroma import ChromaVectorStore, clean_metadata, collection_name, to_where


def _eval_where(meta: dict, where: dict | None) -> bool:
    if not where:
        return True
    if "$and" in where:
        return all(_eval_where(meta, w) for w in where["$and"])
    ((key, cond),) = where.items()
    (op, val), = cond.items()
    v = meta.get(key)
    return {"$eq": v == val, "$ne": v != val, "$in": v in val, "$nin": v not in val}[op]


class FakeCollection:
    """Implements the subset of chromadb.Collection the adapter uses."""

    name = "fake"

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self.get_calls = 0

    def upsert(self, ids, embeddings, documents, metadatas):
        for rid, e, d, m in zip(ids, embeddings, documents, metadatas):
            assert all(isinstance(v, (str, int, float, bool)) for v in m.values()), m
            self.rows[rid] = {"e": e, "d": d, "m": m}

    def get(self, where=None, include=(), limit=None, offset=0):
        self.get_calls += 1
        ids = [rid for rid, r in self.rows.items() if _eval_where(r["m"], where)][offset:offset + limit]
        return {
            "ids": ids,
            "documents": [self.rows[i]["d"] for i in ids],
            "metadatas": [self.rows[i]["m"] for i in ids],
            "embeddings": [self.rows[i]["e"] for i in ids] if "embeddings" in include else None,
        }

    def query(self, query_embeddings, n_results, where=None, include=()):
        q = query_embeddings[0]

        def dist(e):
            dot = sum(a * b for a, b in zip(q, e))
            return 1 - dot / (math.hypot(*q) * math.hypot(*e))

        ranked = sorted((rid for rid, r in self.rows.items() if _eval_where(r["m"], where)), key=lambda i: dist(self.rows[i]["e"]))
        ids = ranked[:n_results]
        return {
            "ids": [ids],
            "documents": [[self.rows[i]["d"] for i in ids]],
            "metadatas": [[self.rows[i]["m"] for i in ids]],
            "distances": [[dist(self.rows[i]["e"]) for i in ids]],
        }

    def delete(self, ids):
        for i in ids:
            self.rows.pop(i, None)


def rec(rid, emb, **meta):
    return VectorRecord(id=rid, text=f"text {rid}", embedding=emb, metadata=meta)


def test_to_where():
    assert to_where(None) is None and to_where({}) is None
    assert to_where({"type": "segment"}) == {"type": {"$eq": "segment"}}
    assert to_where({"entity_id": "x", "kb_version": {"$ne": "v2"}}) == {
        "$and": [{"entity_id": {"$eq": "x"}}, {"kb_version": {"$ne": "v2"}}]
    }
    assert to_where({"entity_id": {"$nin": ["a", "b"]}}) == {"entity_id": {"$nin": ["a", "b"]}}


def test_where_translation_agrees_with_local_matching():
    metas = [{"t": "a", "v": "1"}, {"t": "b", "v": "2"}, {"t": "a", "v": "2"}]
    for flt in ({"t": "a"}, {"t": "a", "v": {"$ne": "1"}}, {"t": {"$in": ["a", "b"]}}, {"v": {"$nin": ["2"]}}):
        assert [_eval_where(m, to_where(flt)) for m in metas] == [matches(m, flt) for m in metas]


def test_clean_metadata_drops_none_and_stringifies():
    assert clean_metadata({"a": None, "b": 1, "c": "x", "d": ["l"]}) == {"b": 1, "c": "x", "d": "['l']"}


def test_collection_name_includes_embedding_model():
    assert collection_name("content_creator_kb", "openai/text-embedding-3-small") == "content_creator_kb__openai-text-embedding-3-small"
    assert len(collection_name("x" * 60, "model")) <= 63


def test_upsert_search_list_delete_roundtrip():
    col = FakeCollection()
    store = ChromaVectorStore(col)
    store.upsert([
        rec("s1", [1.0, 0.0], type="segment", entity_id="a", kb_version="v1", fact_id=None),
        rec("s2", [0.0, 1.0], type="segment", entity_id="a", kb_version="v2"),
        rec("p1", [0.7, 0.7], type="product", entity_id="b", kb_version="v2"),
    ])
    assert "fact_id" not in col.rows["s1"]["m"], "None metadata is dropped"

    hits = store.search([1.0, 0.0], {"type": "segment"}, top_k=2)
    assert [h.record.id for h in hits] == ["s1", "s2"]
    assert round(hits[0].score, 6) == 1.0 and round(hits[1].score, 6) == 0.0

    seg = store.list({"type": "segment"})
    assert sorted(r.id for r in seg) == ["s1", "s2"] and seg[0].embedding

    assert store.delete_by_version("a", keep_version="v2") == 1
    assert sorted(col.rows) == ["p1", "s2"]


def test_list_is_cached_and_writes_invalidate():
    col = FakeCollection()
    store = ChromaVectorStore(col, cache_ttl_seconds=60)
    store.upsert([rec("x", [1.0, 0.0], type="segment", entity_id="a", kb_version="v")])
    store.list({"type": "segment"})
    store.list({"type": "segment"})
    assert col.get_calls == 1
    store.upsert([rec("y", [0.0, 1.0], type="segment", entity_id="a", kb_version="v")])
    assert len(store.list({"type": "segment"})) == 2 and col.get_calls == 2


def test_list_paginates(monkeypatch):
    import app.kb.vectorstore_chroma as mod

    monkeypatch.setattr(mod, "PAGE", 2)
    col = FakeCollection()
    store = ChromaVectorStore(col)
    store.upsert([rec(f"r{i}", [1.0, float(i)], type="segment", entity_id="a", kb_version="v") for i in range(5)])
    assert len(store.list()) == 5 and col.get_calls == 3


def test_km_tools_work_on_chroma_store(settings):
    """The whole KB ingests into the Chroma adapter and the KM tools resolve and retrieve from it."""
    from app.kb.embeddings import make_embedding_provider
    from app.kb.files import KBFiles
    from app.kb.ingest import ingest
    from app.pipeline.tools.km_tools import KMTools

    store = ChromaVectorStore(FakeCollection())
    embedder = make_embedding_provider(settings)
    ingest(settings.kb_source_dir, "c1", store, embedder, settings.kb_dir)
    km = KMTools(store=store, embedder=embedder, kb=KBFiles.load(settings.kb_dir), settings=settings)
    assert km.resolve_entities("young families", "CI").segment.entity_id == "young_families"
    assert km.resolve_entities("", "ประกันบำนาญ").product.entity_id == "annuity"
    assert {c.entity_id for c in km.get_segment_knowledge("first_jobber")} == {"first_jobber"}
    assert km.known_fact_ids("health") == {f"HLT-F0{i}" for i in range(1, 6)}
