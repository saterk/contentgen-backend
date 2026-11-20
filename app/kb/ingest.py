"""Ingestion CLI: chunk, embed and upsert segment/product documents (spec section 6.3).

    python -m app.kb.ingest --source ../data/kb_source --kb-version 2026-11-20.1

Source documents are Markdown with a YAML front-matter header:

    ---
    entity_id: critical_illness
    name: Critical illness
    aliases: [CI, โรคร้ายแรง]
    description: One-line description
    ---

Product facts written as `[CI-F01] ...` become one chunk each and keep their fact_id.
DOCX and PDF sources are not supported yet.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from app.config import Settings, get_settings
from app.kb.embeddings import EmbeddingProvider, make_embedding_provider
from app.kb.vectorstore import VectorRecord, VectorStore, make_vector_store

log = logging.getLogger("ingest")

FACT_RE = re.compile(r"^\s*[-*]?\s*\[([A-Z][A-Z0-9]*-F\d+)\]\s*(.+?)\s*$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
CHUNK_CHARS = 1500  # ~300–500 tokens for mixed Thai/English text
OVERLAP_CHARS = 150


@dataclass
class SourceDoc:
    path: Path
    type: str  # segment | product
    entity_id: str
    name: str
    aliases: list[str]
    description: str
    body: str


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict


def parse_doc(path: Path, doc_type: str) -> SourceDoc:
    raw = path.read_text(encoding="utf-8")
    meta: dict = {}
    body = raw
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", raw, re.S)
    if m:
        meta = yaml.safe_load(m.group(1)) or {}
        body = m.group(2)
    entity_id = meta.get("entity_id") or path.stem
    return SourceDoc(
        path=path,
        type=doc_type,
        entity_id=str(entity_id),
        name=str(meta.get("name") or entity_id),
        aliases=[str(a) for a in (meta.get("aliases") or [])],
        description=str(meta.get("description") or ""),
        body=body,
    )


def split_sections(body: str) -> list[tuple[str, str]]:
    """Split by headings into (section title, text). Text before the first heading → 'intro'."""
    sections: list[tuple[str, list[str]]] = [("intro", [])]
    for line in body.splitlines():
        h = HEADING_RE.match(line)
        if h:
            sections.append((h.group(2).strip(), []))
        else:
            sections[-1][1].append(line)
    return [(title, "\n".join(lines).strip()) for title, lines in sections if "\n".join(lines).strip()]


def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 > size:
            chunks.append(cur)
            cur = cur[-overlap:] + "\n\n" + p if overlap else p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
        while len(cur) > size:  # a single very long paragraph
            chunks.append(cur[:size])
            cur = cur[size - overlap:]
    if cur:
        chunks.append(cur)
    return chunks


def section_slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or "section"


def build_chunks(doc: SourceDoc, kb_version: str) -> list[Chunk]:
    base = {
        "type": doc.type,
        "entity_id": doc.entity_id,
        f"{doc.type}_id": doc.entity_id,
        "kb_version": kb_version,
        "source_doc": doc.path.name,
    }
    chunks: list[Chunk] = []
    for title, text in split_sections(doc.body):
        section = section_slug(title)
        prose: list[str] = []
        for line in text.splitlines():
            fm = FACT_RE.match(line)
            if fm and doc.type == "product":
                fact_id = fm.group(1)
                chunks.append(
                    Chunk(
                        id=f"{doc.type}:{doc.entity_id}:{fact_id}:{kb_version}",
                        text=fm.group(2),
                        metadata={**base, "collection": "products", "section": section, "fact_id": fact_id},
                    )
                )
            else:
                prose.append(line)
        for i, piece in enumerate(chunk_text("\n".join(prose))):
            chunks.append(
                Chunk(
                    id=f"{doc.type}:{doc.entity_id}:{section}:{i}:{kb_version}",
                    text=f"{title}\n{piece}",
                    metadata={**base, "collection": f"{doc.type}s", "section": section},
                )
            )
    return chunks


def catalog_chunks(doc: SourceDoc, kb_version: str) -> list[Chunk]:
    """One catalog entry per name and alias, so short user phrases match short entries."""
    names = [doc.name, *doc.aliases]
    seen: set[str] = set()
    out = []
    for i, n in enumerate(names):
        if n.lower() in seen:
            continue
        seen.add(n.lower())
        out.append(
            Chunk(
                id=f"catalog:{doc.type}:{doc.entity_id}:{i}:{kb_version}",
                text=n,
                metadata={
                    "collection": "catalog",
                    "type": doc.type,
                    "entity_id": doc.entity_id,
                    "name": doc.name,
                    "alias": n,
                    "description": doc.description,
                    "kb_version": kb_version,
                    "source_doc": doc.path.name,
                },
            )
        )
    return out


def ingest(
    source: Path,
    kb_version: str,
    store: VectorStore,
    embedder: EmbeddingProvider,
    kb_dir: Path,
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for doc_type, folder in (("segment", "segments"), ("product", "products")):
        for path in sorted((source / folder).glob("*.md")):
            doc = parse_doc(path, doc_type)
            chunks = catalog_chunks(doc, kb_version) + build_chunks(doc, kb_version)
            vectors = embedder.embed([c.text for c in chunks])
            for c in chunks:
                c.metadata["embedding_model"] = embedder.model
            store.upsert(
                [VectorRecord(id=c.id, text=c.text, embedding=v, metadata=c.metadata) for c, v in zip(chunks, vectors)]
            )
            removed = store.delete_by_version(doc.entity_id, kb_version)
            n_facts = sum(1 for c in chunks if c.metadata.get("fact_id"))
            n_catalog = sum(1 for c in chunks if c.metadata["collection"] == "catalog")
            counts[doc.entity_id] = {
                "chunks": len(chunks) - n_catalog,
                "facts": n_facts,
                "catalog_entries": n_catalog,
                "removed_old": removed,
            }
            log.info("%s %s: %s", doc_type, doc.entity_id, counts[doc.entity_id])

    # Entities whose source document was removed disappear from the KB too.
    stale = {r.metadata["entity_id"] for r in store.list({"entity_id": {"$nin": list(counts)}})}
    if stale:
        removed = store.delete({"entity_id": {"$nin": list(counts)}})
        log.info("removed %d chunks of entities no longer in the source: %s", removed, sorted(stale))

    manifest_path = kb_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.update(
        {
            "kb_version": kb_version,
            "embedding_provider": embedder.provider_name,
            "embedding_model": embedder.model,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "entities": counts,
        }
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return counts


def ingest_with_settings(settings: Settings, source: Path | None = None, kb_version: str | None = None):
    store = make_vector_store(settings)
    embedder = make_embedding_provider(settings)
    version = kb_version or datetime.now().strftime("%Y-%m-%d.1")
    return ingest(source or settings.kb_source_dir, version, store, embedder, settings.kb_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ingest segment and product knowledge into the vector store.")
    parser.add_argument("--source", type=Path, default=None, help="kb_source folder (default: DATA_DIR/kb_source)")
    parser.add_argument("--kb-version", default=None, help="e.g. 2026-11-20.1 (default: today's date .1)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    counts = ingest_with_settings(get_settings(), args.source, args.kb_version)
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
