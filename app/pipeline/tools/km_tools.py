"""KM tools: map chat text to known segments/products and load their knowledge (spec 5.2)."""

from __future__ import annotations

from typing import Literal

from app.config import Settings
from app.kb.embeddings import EmbeddingProvider, cosine
from app.kb.files import KBFiles, contains_term, norm_text
from app.kb.vectorstore import SearchHit, VectorRecord, VectorStore
from app.schemas.kb import Candidate, EntityMatch, KBChunk, ResolvedEntities

EntityType = Literal["segment", "product"]

ALIAS_MENTION_SCORE = 0.95  # the text names a known alias outright


def _chunk(rec: VectorRecord, score: float | None) -> KBChunk:
    m = rec.metadata
    return KBChunk(
        chunk_id=rec.id,
        type=m["type"],
        entity_id=m["entity_id"],
        fact_id=m.get("fact_id"),
        section=m.get("section"),
        text=rec.text,
        source_doc=m.get("source_doc", ""),
        kb_version=m.get("kb_version", ""),
        score=None if score is None else round(score, 4),
    )


class KMTools:
    def __init__(self, store: VectorStore, embedder: EmbeddingProvider, kb: KBFiles, settings: Settings) -> None:
        self.store = store
        self.embedder = embedder
        self.kb = kb
        self.settings = settings

    # ---- catalog -------------------------------------------------------
    def catalog(self, entity_type: EntityType) -> dict[str, str]:
        """entity_id -> display name."""
        return {
            r.metadata["entity_id"]: r.metadata["name"]
            for r in self.store.list({"collection": "catalog", "type": entity_type})
        }

    def known_fact_ids(self, product_id: str) -> set[str]:
        return {
            r.metadata["fact_id"]
            for r in self.store.list({"collection": "products", "product_id": product_id})
            if r.metadata.get("fact_id")
        }

    # ---- resolve_entities ----------------------------------------------
    def resolve_entities(self, segment_text: str, product_text: str) -> ResolvedEntities:
        return ResolvedEntities(
            segment=self._resolve_one(segment_text, "segment"),
            product=self._resolve_one(product_text, "product"),
        )

    def _resolve_one(self, text: str, entity_type: EntityType) -> EntityMatch:
        text = (text or "").strip()
        entries = self.store.list({"collection": "catalog", "type": entity_type})
        if not text or not entries:
            return EntityMatch(text=text, entity_id=None)
        qv = self.embedder.embed([text])[0]
        best: dict[str, Candidate] = {}
        for e in entries:
            alias = e.metadata.get("alias", e.text)
            if norm_text(alias) == norm_text(text):
                score = 1.0
            else:
                score = cosine(qv, e.embedding)
                if contains_term(text, alias):
                    score = max(score, ALIAS_MENTION_SCORE)
            eid = e.metadata["entity_id"]
            if eid not in best or score > best[eid].score:
                best[eid] = Candidate(entity_id=eid, name=e.metadata["name"], score=round(score, 4))
        ranked = sorted(best.values(), key=lambda c: c.score, reverse=True)
        top = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None
        confident = top.score >= self.settings.km_match_threshold and (
            runner_up is None or top.score - runner_up.score >= self.settings.km_ambiguity_margin
        )
        return EntityMatch(
            text=text,
            entity_id=top.entity_id if confident else None,
            name=top.name,
            score=top.score,
            confident=confident,
            alternatives=ranked[1:4],
        )

    # ---- knowledge -----------------------------------------------------
    def _search(self, query: str, flt: dict, top_k: int) -> list[SearchHit]:
        qv = self.embedder.embed([query])[0]
        return self.store.search(qv, flt, top_k)

    def get_segment_knowledge(self, segment_id: str, query: str | None = None, top_k: int = 6) -> list[KBChunk]:
        hits = self._search(query or segment_id, {"collection": "segments", "segment_id": segment_id}, top_k)
        return [_chunk(h.record, h.score) for h in hits]

    def get_product_knowledge(self, product_id: str, query: str | None = None, top_k: int = 8) -> list[KBChunk]:
        """Ranked facts and product sections, plus compliance rules and disclaimers (always included)."""
        flt = {"collection": "products", "product_id": product_id}
        always = [
            _chunk(r, None)
            for r in self.store.list(flt)
            if "compliance" in (r.metadata.get("section") or "")
        ]
        always_ids = {c.chunk_id for c in always}
        ranked = [
            _chunk(h.record, h.score)
            for h in self._search(query or product_id, flt, top_k + len(always_ids))
            if h.record.id not in always_ids
        ][:top_k]
        disclaimers = [
            KBChunk(
                chunk_id=f"disclaimer:{d.id}",
                type="disclaimer",
                entity_id=product_id,
                fact_id=d.id,
                section="disclaimer",
                text=d.text,
                source_doc="compliance/disclaimers.json",
                kb_version=self.kb.kb_version,
            )
            for d in self.kb.disclaimers_for(product_id)
        ]
        return ranked + always + disclaimers
