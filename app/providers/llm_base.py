"""LLM provider interface and a structured-output helper with validation retries."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class LLMProvider(ABC):
    provider_name: str
    model: str

    @abstractmethod
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
        """Run one model call.

        `task` names the job (extract_requirement, plan_angles, design_brief,
        copy_judge, image_judge). Real adapters use `system`, `messages`, `json_schema`
        and `images`; the mock uses `context` (the same inputs, unrendered).
        """


class StructuredOutputError(Exception):
    pass


@dataclass
class StructuredResult(Generic[T]):
    value: T
    attempts: int
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    errors: list[str] = field(default_factory=list)


def _extract_json(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t.split("\n", 1)[1] if "\n" in t else t
    start = min((i for i in (t.find("{"), t.find("[")) if i >= 0), default=0)
    return t[start:]


async def structured_call(
    llm: LLMProvider,
    schema: type[T],
    *,
    task: str,
    system: str,
    user: str,
    context: dict[str, Any] | None = None,
    images: list[Path] | None = None,
    validation_context: dict[str, Any] | None = None,
    retries: int = 2,
    temperature: float | None = None,
) -> StructuredResult[T]:
    """Call the LLM and validate its JSON against `schema`.

    On a validation error, retry up to `retries` times with the error added to the prompt.
    """
    messages = [{"role": "user", "content": user}]
    json_schema = schema.model_json_schema()
    errors: list[str] = []
    totals = {"in": 0, "out": 0, "cost": 0.0}
    started = time.perf_counter()
    for attempt in range(1, retries + 2):
        resp = await llm.complete(
            task=task,
            system=system,
            messages=messages,
            json_schema=json_schema,
            images=images,
            context={**(context or {}), "attempt": attempt, "errors": errors},
            temperature=temperature,
        )
        totals["in"] += resp.input_tokens
        totals["out"] += resp.output_tokens
        totals["cost"] += resp.cost_usd
        try:
            value = schema.model_validate(json.loads(_extract_json(resp.text)), context=validation_context)
            return StructuredResult(
                value=value,
                attempts=attempt,
                input_tokens=totals["in"],
                output_tokens=totals["out"],
                cost_usd=totals["cost"],
                latency_ms=int((time.perf_counter() - started) * 1000),
                errors=errors,
            )
        except (ValidationError, json.JSONDecodeError) as e:
            err = str(e)
            errors.append(err)
            messages = messages + [
                {"role": "assistant", "content": resp.text},
                {
                    "role": "user",
                    "content": f"Your output did not match the required JSON schema:\n{err}\nReturn corrected JSON only.",
                },
            ]
    raise StructuredOutputError(f"{task}: output failed validation after {retries + 1} attempts: {errors[-1]}")


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)
