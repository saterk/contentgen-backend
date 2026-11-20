"""Pluggable model providers, chosen by config."""

from __future__ import annotations

from app.config import Settings
from app.providers.image_base import ImageProvider
from app.providers.llm_base import LLMProvider


def make_llm(settings: Settings, model: str | None = None) -> LLMProvider:
    if settings.llm_provider in ("mock", ""):
        from app.providers.llm_mock import MockLLM

        return MockLLM(model=model or settings.llm_model or "mock-llm-1")
    if settings.llm_provider in ("openrouter", "openai"):
        from app.providers.llm_openai_compat import DEFAULT_BASE_URLS, OpenAICompatLLM

        return OpenAICompatLLM(
            provider_name=settings.llm_provider,
            model=model or settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url or DEFAULT_BASE_URLS[settings.llm_provider],
            reasoning=settings.llm_reasoning,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    raise ValueError(f"LLM provider '{settings.llm_provider}' is not supported (use mock, openrouter or openai)")


def make_image_provider(settings: Settings) -> ImageProvider:
    if settings.image_provider in ("mock", ""):
        from app.providers.image_mock import MockImageProvider

        return MockImageProvider(model=settings.image_model or "mock-image-1")
    if settings.image_provider == "openrouter":
        from app.providers.image_openrouter import OpenRouterImageProvider

        return OpenRouterImageProvider(
            model=settings.image_model,
            api_key=settings.image_api_key or settings.llm_api_key,
            base_url=settings.image_base_url or "https://openrouter.ai/api/v1",
        )
    raise ValueError(f"Image provider '{settings.image_provider}' is not supported (use mock or openrouter)")
