"""Everything the pipeline needs, built once per app from settings."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import Engine

from app.config import Settings
from app.db import make_engine
from app.kb.embeddings import EmbeddingProvider, make_embedding_provider
from app.kb.files import KBFiles
from app.kb.vectorstore import VectorStore, make_vector_store
from app.pipeline.tools.km_tools import KMTools
from app.providers import make_image_provider, make_llm
from app.providers.image_base import ImageProvider
from app.providers.llm_base import LLMProvider
from app.storage import ImageStore, make_image_store

log = logging.getLogger("content_creator")


class Prompts:
    """Versioned prompt templates in backend/prompts/<name>.<version>.md."""

    def __init__(self, folder: Path, version: str) -> None:
        self.folder = folder
        self.version = version
        self._cache: dict[str, str] = {}

    def get(self, name: str) -> str:
        if name not in self._cache:
            path = self.folder / f"{name}.{self.version}.md"
            self._cache[name] = path.read_text(encoding="utf-8")
        return self._cache[name]


@dataclass
class AppContext:
    settings: Settings
    engine: Engine
    llm: LLMProvider
    judge_llm: LLMProvider
    images: ImageProvider
    embedder: EmbeddingProvider
    store: VectorStore
    kb: KBFiles
    km: KMTools
    prompts: Prompts
    storage: ImageStore
    warnings: list[str] = field(default_factory=list)
    _sem: asyncio.Semaphore | None = field(default=None, init=False, repr=False)
    _sem_loop: asyncio.AbstractEventLoop | None = field(default=None, init=False, repr=False)

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """Shared across all chat requests: at most MAX_CONCURRENT_JOBS pieces run at once."""
        loop = asyncio.get_running_loop()
        if self._sem is None or self._sem_loop is not loop:
            self._sem = asyncio.Semaphore(self.settings.max_concurrent_jobs)
            self._sem_loop = loop
        return self._sem

    @classmethod
    def build(cls, settings: Settings) -> "AppContext":
        settings.images_dir.mkdir(parents=True, exist_ok=True)
        kb = KBFiles.load(settings.kb_dir)
        store = make_vector_store(settings)
        embedder = make_embedding_provider(settings)
        ctx = cls(
            settings=settings,
            engine=make_engine(settings.db_url),
            llm=make_llm(settings),
            judge_llm=make_llm(settings, model=settings.judge_model_name),
            images=make_image_provider(settings),
            embedder=embedder,
            store=store,
            kb=kb,
            km=KMTools(store=store, embedder=embedder, kb=kb, settings=settings),
            prompts=Prompts(settings.prompts_dir, settings.prompt_version),
            storage=make_image_store(settings),
        )
        ctx.check_kb()
        return ctx

    def check_kb(self) -> None:
        self.warnings.clear()
        if not self.kb.manifest:
            self.warnings.append("KB not ingested yet: run `python -m app.kb.ingest`")
        elif self.kb.manifest.get("embedding_model") != self.embedder.model:
            self.warnings.append(
                f"KB was embedded with '{self.kb.manifest.get('embedding_model')}' but EMBEDDING_MODEL is "
                f"'{self.embedder.model}': re-run ingestion"
            )
        for w in self.warnings:
            log.warning(w)
