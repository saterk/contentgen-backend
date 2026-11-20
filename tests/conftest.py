from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import ROOT_DIR, Settings
from app.kb.ingest import ingest_with_settings
from app.main import create_app
from app.pipeline.context import AppContext
from app.pipeline.events import Event
from app.pipeline.runner import run_chat

FULL_REQUEST = (
    "Product: critical illness. Segment: young families. Style: story. "
    "Tone: warm. Format: single image with CTA. Make {n} pieces."
)
API_KEY = "test-key"


def make_settings(data_dir: Path, **overrides) -> Settings:
    base = dict(
        data_dir=data_dir,
        app_api_key=API_KEY,
        llm_provider="mock",
        llm_model="mock-llm-1",
        judge_model="",
        image_provider="mock",
        image_model="mock-image-1",
        embedding_provider="mock",
        embedding_model="mock-embed-1",
        vector_store="local",
        database_url="",
        image_size="216x270",  # small images keep tests fast
        image_variants=1,
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    shutil.copytree(ROOT_DIR / "data" / "kb", d / "kb", ignore=shutil.ignore_patterns("vectors.json", "manifest.json"))
    shutil.copytree(ROOT_DIR / "data" / "kb_source", d / "kb_source")
    return d


@pytest.fixture
def settings(data_dir: Path) -> Settings:
    s = make_settings(data_dir)
    ingest_with_settings(s, kb_version="test.1")
    return s


@pytest.fixture
def ctx(settings: Settings) -> AppContext:
    return AppContext.build(settings)


@pytest.fixture
def client(settings: Settings):
    with TestClient(create_app(settings)) as c:
        c.headers["Authorization"] = f"Bearer {API_KEY}"
        yield c


def run(ctx: AppContext, messages: str | list[dict], mode: str = "content-creator", conversation_id: str = "conv_test") -> list[Event]:
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    async def collect() -> list[Event]:
        return [e async for e in run_chat(ctx, messages, conversation_id, mode)]

    return asyncio.run(collect())


def of_type(events: list[Event], t: str) -> list[Event]:
    return [e for e in events if e.type == t]
