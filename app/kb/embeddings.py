"""Embedding provider interface and a deterministic offline mock."""

from __future__ import annotations

import math
import zlib
from abc import ABC, abstractmethod

from app.kb.files import norm_text


class EmbeddingProvider(ABC):
    provider_name: str
    model: str

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class MockEmbedding(EmbeddingProvider):
    """Hashed character n-grams (1–3) plus word tokens. Works for Thai and English.

    Deterministic across processes (crc32, not Python's salted hash). Good enough to
    exercise retrieval and entity resolution offline; not a semantic model.
    """

    provider_name = "mock"

    def __init__(self, model: str = "mock-embed-1", dims: int = 512) -> None:
        self.model = model
        self.dims = dims

    def _features(self, text: str) -> list[str]:
        t = norm_text(text)
        feats = [f"w:{w}" for w in t.split()]
        padded = f" {t} "
        for n in (2, 3):
            feats += [f"c{n}:{padded[i:i + n]}" for i in range(len(padded) - n + 1)]
        return feats

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dims
            for f in self._features(text):
                h = zlib.crc32(f.encode("utf-8"))
                vec[h % self.dims] += 1.0 if (h >> 16) & 1 else -1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class OpenAICompatEmbedding(EmbeddingProvider):
    """Embeddings through an OpenAI-compatible /embeddings endpoint (OpenRouter, OpenAI)."""

    BATCH = 64

    def __init__(
        self, *, provider_name: str, model: str, api_key: str, base_url: str | None, http_client=None
    ) -> None:
        if not api_key:
            raise ValueError(f"EMBEDDING_API_KEY (or LLM_API_KEY) is required for EMBEDDING_PROVIDER={provider_name}")
        from openai import OpenAI

        self.provider_name = provider_name
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=30, max_retries=2, http_client=http_client)
        self._cache: dict[str, list[float]] = {}

    def embed(self, texts: list[str]) -> list[list[float]]:
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        for i in range(0, len(todo), self.BATCH):
            batch = todo[i:i + self.BATCH]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            for t, d in zip(batch, sorted(resp.data, key=lambda d: d.index)):
                self._cache[t] = list(d.embedding)
        return [self._cache[t] for t in texts]


def make_embedding_provider(settings) -> EmbeddingProvider:
    provider, model = settings.embedding_provider, settings.embedding_model
    if provider in ("mock", ""):
        return MockEmbedding(model=model or "mock-embed-1")
    if provider in ("openrouter", "openai"):
        default = {"openrouter": "https://openrouter.ai/api/v1", "openai": None}[provider]
        same_as_llm = settings.llm_provider == provider
        return OpenAICompatEmbedding(
            provider_name=provider,
            model=model,
            api_key=settings.embedding_api_key or settings.llm_api_key,
            base_url=settings.embedding_base_url or (settings.llm_base_url if same_as_llm else "") or default,
        )
    raise ValueError(f"Embedding provider '{provider}' is not supported (use mock, openrouter or openai)")
