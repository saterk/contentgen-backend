"""Structured run logs: one run_log row per pipeline step, plus per-request totals."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator

from sqlalchemy import Engine

from app.db import session_scope
from app.models import RunLog, new_id

log = logging.getLogger("content_creator")


@dataclass
class StepRecord:
    """Mutable record a step fills in while it runs."""

    provider: str | None = None
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, input_tokens: int = 0, output_tokens: int = 0, cost_usd: float = 0.0) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cost_usd += cost_usd


@dataclass
class Totals:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    steps: int = 0
    by_item: dict[str, float] = field(default_factory=dict)


class RunLogger:
    def __init__(self, engine: Engine, prompt_version: str, kb_version: str) -> None:
        self.engine = engine
        self.prompt_version = prompt_version
        self.kb_version = kb_version
        self.totals = Totals()

    @asynccontextmanager
    async def step(
        self,
        step: str,
        *,
        request_id: str | None = None,
        item_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> AsyncIterator[StepRecord]:
        rec = StepRecord(provider=provider, model=model)
        started = time.perf_counter()
        error: str | None = None
        try:
            yield rec
        except BaseException as e:  # log cancellations and timeouts too, then re-raise
            error = f"{type(e).__name__}: {e}"[:2000]
            raise
        finally:
            self.write(
                step,
                request_id=request_id,
                item_id=item_id,
                rec=rec,
                latency_ms=int((time.perf_counter() - started) * 1000),
                error=error,
            )

    def write(
        self,
        step: str,
        *,
        request_id: str | None,
        item_id: str | None,
        rec: StepRecord,
        latency_ms: int,
        error: str | None = None,
    ) -> None:
        self.totals.input_tokens += rec.input_tokens
        self.totals.output_tokens += rec.output_tokens
        self.totals.cost_usd += rec.cost_usd
        self.totals.steps += 1
        if item_id:
            self.totals.by_item[item_id] = self.totals.by_item.get(item_id, 0.0) + rec.cost_usd
        with session_scope(self.engine) as s:
            s.add(
                RunLog(
                    id=new_id("log"),
                    request_id=request_id,
                    item_id=item_id,
                    step=step,
                    provider=rec.provider,
                    model=rec.model,
                    prompt_version=self.prompt_version,
                    kb_version=self.kb_version,
                    input_tokens=rec.input_tokens,
                    output_tokens=rec.output_tokens,
                    cost_usd=rec.cost_usd,
                    latency_ms=latency_ms,
                    error=error,
                )
            )
        log.info(
            "step=%s request=%s item=%s latency_ms=%s tokens=%s/%s error=%s",
            step, request_id, item_id, latency_ms, rec.input_tokens, rec.output_tokens, error,
        )
