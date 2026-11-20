"""LLM adapter for any OpenAI-compatible Chat Completions API (OpenRouter, OpenAI, vLLM, ...)."""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from app.providers.llm_base import LLMProvider, LLMResponse

DEFAULT_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": None,  # SDK default
}


def _image_part(path: Path) -> dict[str, Any]:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


class OpenAICompatLLM(LLMProvider):
    def __init__(
        self,
        *,
        provider_name: str,
        model: str,
        api_key: str,
        base_url: str | None,
        reasoning: str = "off",  # off | low | medium | high (OpenRouter only)
        timeout_seconds: float = 60,
        default_temperature: float = 0.7,
        http_client: Any = None,  # httpx.AsyncClient, for tests
    ) -> None:
        if not api_key:
            raise ValueError(f"LLM_API_KEY is required for LLM_PROVIDER={provider_name}")
        if not model:
            raise ValueError(f"LLM_MODEL is required for LLM_PROVIDER={provider_name}")
        self.provider_name = provider_name
        self.model = model
        self.reasoning = reasoning
        self.default_temperature = default_temperature
        self.client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout_seconds, max_retries=2, http_client=http_client
        )

    def _extra_body(self) -> dict[str, Any]:
        if self.provider_name != "openrouter":
            return {}
        extra: dict[str, Any] = {"usage": {"include": True}}  # returns cost in usage
        if self.reasoning == "off":
            extra["reasoning"] = {"enabled": False}
        elif self.reasoning in ("low", "medium", "high"):
            extra["reasoning"] = {"effort": self.reasoning}
        return extra

    async def complete(
        self,
        *,
        task: str,
        system: str,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any] | None = None,
        images: list[Path] | None = None,
        context: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        sys_prompt = system
        if json_schema is not None:
            sys_prompt += (
                "\n\nRespond with one JSON object only (no markdown, no commentary) that validates "
                "against this JSON Schema:\n" + json.dumps(json_schema, ensure_ascii=False)
            )
        msgs: list[dict[str, Any]] = [{"role": "system", "content": sys_prompt}]
        msgs += [dict(m) for m in messages]
        if images:
            last_user = next(i for i in range(len(msgs) - 1, -1, -1) if msgs[i]["role"] == "user")
            msgs[last_user]["content"] = [
                {"type": "text", "text": msgs[last_user]["content"]},
                *(_image_part(p) for p in images),
            ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "temperature": self.default_temperature if temperature is None else temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if json_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
        extra = self._extra_body()
        if extra:
            kwargs["extra_body"] = extra

        resp = await self.client.chat.completions.create(**kwargs)
        text = (resp.choices[0].message.content or "") if resp.choices else ""
        usage = resp.usage
        cost = float(((usage.model_extra or {}).get("cost") or 0.0)) if usage else 0.0
        return LLMResponse(
            text=text,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cost_usd=cost,
        )
