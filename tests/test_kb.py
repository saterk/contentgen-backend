from __future__ import annotations

import json

import pytest

from app.kb.ingest import build_chunks, chunk_text, ingest_with_settings, parse_doc
from app.kb.vectorstore import LocalVectorStore


def test_ingest_counts_and_manifest(settings):
    manifest = json.loads((settings.kb_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["kb_version"] == "test.1"
    assert manifest["embedding_model"] == "mock-embed-1"
    assert manifest["entities"]["critical_illness"]["facts"] == 5
    assert set(manifest["entities"]) == {"critical_illness", "endowment", "annuity", "health", "young_families", "happy_retirement", "first_jobber"}


def test_fact_lines_become_single_chunks_with_fact_ids(settings):
    doc = parse_doc(settings.kb_source_dir / "products" / "critical_illness.md", "product")
    assert doc.entity_id == "critical_illness" and "ci" in doc.aliases and "โรคร้ายแรง" in doc.aliases
    chunks = build_chunks(doc, "v1")
    facts = [c for c in chunks if c.metadata.get("fact_id")]
    assert [c.metadata["fact_id"] for c in facts] == [f"CI-F0{i}" for i in range(1, 6)]
    assert all(not c.text.startswith("[") for c in facts)
    assert all(c.metadata["product_id"] == "critical_illness" and c.metadata["kb_version"] == "v1" for c in chunks)


def test_chunk_text_splits_long_text_with_overlap():
    text = "\n\n".join(f"Paragraph {i} " + "x" * 400 for i in range(10))
    chunks = chunk_text(text, size=1000, overlap=100)
    assert len(chunks) > 3
    assert all(len(c) <= 1000 for c in chunks)


def test_reingest_removes_older_versions(settings):
    ingest_with_settings(settings, kb_version="test.2")
    store = LocalVectorStore(settings.data_path / "vectors.json")
    versions = {r.metadata["kb_version"] for r in store.list()}
    assert versions == {"test.2"}


def test_segment_knowledge_is_filtered_to_the_segment(ctx):
    chunks = ctx.km.get_segment_knowledge("young_families", query="story warm")
    assert 0 < len(chunks) <= 6
    assert {c.entity_id for c in chunks} == {"young_families"}
    assert all(c.type == "segment" and c.score is not None for c in chunks)


def test_product_knowledge_includes_facts_compliance_and_disclaimers(ctx):
    chunks = ctx.km.get_product_knowledge("critical_illness", query="story warm")
    assert any(c.fact_id and c.fact_id.startswith("CI-F") for c in chunks)
    always = [c for c in chunks if c.score is None]
    assert {c.type for c in always} == {"product", "disclaimer"}
    assert any(c.section and "compliance" in c.section for c in always)
    assert {c.fact_id for c in chunks if c.type == "disclaimer"} == {"DISC-CI-01", "DISC-CI-02"}
    assert {c.entity_id for c in chunks} == {"critical_illness"}


@pytest.mark.parametrize(
    ("text", "segment_id"),
    [
        ("first jobber", "first_jobber"),
        ("เด็กจบใหม่", "first_jobber"),
        ("happy retirement", "happy_retirement"),
        ("วัยเกษียณ", "happy_retirement"),
        ("young families", "young_families"),
        ("ครอบครัวรุ่นใหม่", "young_families"),
    ],
)
def test_resolve_segment(ctx, text, segment_id):
    m = ctx.km.resolve_entities(text, "").segment
    assert m.confident and m.entity_id == segment_id and m.score >= ctx.settings.km_match_threshold


@pytest.mark.parametrize("text", ["CI", "critical illness", "โรคร้ายแรง", "ประกันโรคร้ายแรง", "critical illness insurance plan"])
def test_resolve_product(ctx, text):
    m = ctx.km.resolve_entities("", text).product
    assert m.confident and m.entity_id == "critical_illness"


@pytest.mark.parametrize("text", ["pet owners", "motorbike"])
def test_unknown_phrases_are_not_guessed(ctx, text):
    r = ctx.km.resolve_entities(text, text)
    assert r.segment.entity_id is None and r.product.entity_id is None
    assert not r.segment.confident and not r.product.confident


def test_close_candidates_are_ambiguous(ctx):
    # Both segments named outright → two candidates at the same score → ask, don't guess.
    m = ctx.km.resolve_entities("young families or first jobbers", "").segment
    assert not m.confident and m.entity_id is None
    assert {m.name, *(a.name for a in m.alternatives)} >= {"Young Families", "First Jobber"}


@pytest.mark.parametrize(
    ("field", "raw", "expected"),
    [
        ("tone", "warm", "warm"),
        ("tone", "friendly and warm", "warm"),
        ("tone", "อบอุ่น", "warm"),
        ("style", "storytelling", "story"),
        ("style", "Story", "story"),
        ("style", "stroy", "story"),
        ("format", "single image with CTA", "single_image_cta"),
        ("format", "video", None),
        ("tone", "angry", None),
    ],
)
def test_taxonomy_normalise(ctx, field, raw, expected):
    assert ctx.kb.taxonomy.normalise(field, raw) == expected


@pytest.mark.parametrize(
    ("text", "product_id"),
    [
        ("endowment", "endowment"),
        ("ประกันสะสมทรัพย์", "endowment"),
        ("annuity", "annuity"),
        ("ประกันบำนาญ", "annuity"),
        ("health insurance", "health"),
        ("ประกันสุขภาพ", "health"),
    ],
)
def test_resolve_new_products(ctx, text, product_id):
    m = ctx.km.resolve_entities("", text).product
    assert m.confident and m.entity_id == product_id


def test_removed_source_docs_leave_the_kb(settings):
    (settings.kb_source_dir / "segments" / "first_jobber.md").unlink()
    ingest_with_settings(settings, kb_version="test.3")
    store = LocalVectorStore(settings.data_path / "vectors.json")
    assert "first_jobber" not in {r.metadata["entity_id"] for r in store.list()}
