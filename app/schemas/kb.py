"""Knowledge-base schemas used by the KM tools (spec section 5.2)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class KBChunk(BaseModel):
    chunk_id: str
    type: Literal["segment", "product", "disclaimer"]
    entity_id: str
    fact_id: str | None = None  # product facts and disclaimers
    section: str | None = None
    text: str
    source_doc: str
    kb_version: str
    score: float | None = None  # None for chunks that are always included (not ranked)


class Candidate(BaseModel):
    entity_id: str
    name: str
    score: float


class EntityMatch(BaseModel):
    text: str
    entity_id: str | None  # set only when the match is confident
    name: str | None = None
    score: float = 0.0
    confident: bool = False
    alternatives: list[Candidate] = Field(default_factory=list)  # up to 3, best first


class ResolvedEntities(BaseModel):
    segment: EntityMatch
    product: EntityMatch
