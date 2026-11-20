"""Runs one chat request end to end and yields events (spec 5.0).

understand → plan N angles → pieces in parallel (≤ MAX_CONCURRENT_JOBS, shared) →
variety check (one regeneration round) → summary.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from app.kb.embeddings import cosine
from app.models import ContentItem, ContentRequestRow, ItemStatus, RequestStatus
from app.pipeline import events as ev
from app.pipeline.context import AppContext
from app.pipeline.orchestrator import NeedsInput, Orchestrator, Understood, new_request_id
from app.pipeline.piece import MODE_BRIEF, MODE_FULL, process_piece
from app.pipeline.store import (
    InvalidTransition,
    create_items,
    create_request,
    get_item,
    set_status,
    update_request,
)
from app.runlog import RunLogger
from app.schemas.content import ContentBrief

log = logging.getLogger("content_creator")

MODES = (MODE_FULL, MODE_BRIEF)
_SENTINEL = object()


async def run_chat(
    ctx: AppContext,
    messages: list[dict[str, Any]],
    conversation_id: str,
    mode: str = MODE_FULL,
) -> AsyncIterator[ev.Event]:
    started = time.monotonic()
    deadline = started + ctx.settings.request_timeout_seconds
    request_id = new_request_id()
    rl = RunLogger(ctx.engine, ctx.settings.prompt_version, ctx.kb.kb_version)
    orch = Orchestrator(ctx, rl, request_id)

    yield ev.text("Reading your request and searching the knowledge base…\n")
    outcome = await orch.understand(messages, conversation_id)
    if isinstance(outcome, NeedsInput):
        yield ev.Event(
            type=ev.NEEDS_INPUT,
            text=outcome.question,
            data={"missing": outcome.missing, "conversation_id": conversation_id, **outcome.data},
        )
        return

    u: Understood = outcome
    req = u.request
    create_request(ctx.engine, ContentRequestRow(
        id=req.id, conversation_id=conversation_id, mode=mode,
        request_json=req.model_dump_json(), kb_version=ctx.kb.kb_version,
    ))
    yield ev.Event(type=ev.REQUIREMENT, text=u.summary(), data={"request": req.model_dump(mode="json")})

    angles = await orch.plan_angles(u, req.count)
    tags = {
        "product": req.profile.product, "style": req.profile.style, "tone": req.profile.tone,
        "format": req.profile.format, "visual_style": req.visual_style, "segment": req.segment_id, "journey_stage": req.journey_stage,
        "cell_id": req.cell_id, "prompt_version": ctx.settings.prompt_version, "kb_version": ctx.kb.kb_version,
        "llm_model": ctx.llm.model, "judge_model": ctx.judge_llm.model,
        "image_model": ctx.images.model if mode == MODE_FULL else None,
    }
    items = create_items(ctx.engine, req.id, conversation_id, angles, tags)
    update_request(ctx.engine, req.id, angles_json=json.dumps(angles, ensure_ascii=False))
    plan_text = "Plan:\n" + "\n".join(f"{i}. {a}" for i, a in enumerate(angles, start=1)) + "\n\n"
    yield ev.Event(
        type=ev.PLAN, text=plan_text,
        data={"count": len(items), "pieces": [{"piece": it.piece_no, "item_id": it.id, "angle": it.angle} for it in items]},
    )

    jobs = [(it.id, it.piece_no, it.angle, 1) for it in items]
    async for e in _run_pieces(ctx, rl, u, jobs, mode, deadline):
        yield e

    # ---- variety check: regenerate near-duplicates once ----------------------
    dupes = await asyncio.to_thread(_find_near_duplicates, ctx, [get_item(ctx.engine, it.id) for it in items])
    if dupes and time.monotonic() < deadline:
        current_angles = [get_item(ctx.engine, it.id).angle for it in items]  # type: ignore[union-attr]
        new_angles = await orch.plan_angles(u, len(dupes), avoid=current_angles)
        regen = []
        for (item, other, score), angle in zip(dupes, new_angles):
            yield ev.text(
                f"Pieces {other.piece_no} and {item.piece_no} are too similar (similarity {score:.2f}). "
                f"Regenerating piece {item.piece_no} with a new angle: {angle}\n"
            )
            set_status(ctx.engine, item.id, ItemStatus.pending_review, angle=angle, version=item.version + 1)
            regen.append((item.id, item.piece_no, angle, item.version + 1))
        async for e in _run_pieces(ctx, rl, u, regen, mode, deadline):
            yield e

    # ---- summary -------------------------------------------------------------
    final = [get_item(ctx.engine, it.id) for it in items]
    ready = [i for i in final if i and i.status == ItemStatus.pending_review.value]
    attention = [i for i in final if i and i.status != ItemStatus.pending_review.value]
    update_request(ctx.engine, req.id, status=RequestStatus.done.value)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    t = rl.totals
    text_lines = [f"Done: {len(final)} piece(s) in {elapsed_ms / 1000:.1f}s. {len(ready)} passed the checks and are ready for review."]
    if attention:
        text_lines.append("Needs attention: " + ", ".join(f"piece {i.piece_no} ({i.error or i.status})" for i in attention))
    yield ev.Event(
        type=ev.DONE,
        text="\n" + "\n".join(text_lines) + "\n",
        data={
            "request_id": req.id,
            "conversation_id": conversation_id,
            "item_ids": [i.id for i in final if i],
            "summary": {"total": len(final), "pending_review": len(ready), "needs_attention": len(attention)},
            "total_time_ms": elapsed_ms,
            "cost_usd": round(t.cost_usd, 6),
            "usage": {"prompt_tokens": t.input_tokens, "completion_tokens": t.output_tokens},
        },
    )


async def _run_pieces(
    ctx: AppContext,
    rl: RunLogger,
    u: Understood,
    jobs: list[tuple[str, int, str, int]],
    mode: str,
    deadline: float,
) -> AsyncIterator[ev.Event]:
    """Run pieces concurrently and yield their events as they arrive."""
    queue: asyncio.Queue = asyncio.Queue()

    async def emit(e: ev.Event) -> None:
        await queue.put(e)

    async def run_one(item_id: str, piece: int, angle: str, version: int) -> None:
        try:
            async with ctx.semaphore:
                await asyncio.wait_for(
                    process_piece(ctx, rl, u, item_id, piece, angle, version, mode, emit),
                    timeout=ctx.settings.item_timeout_seconds,
                )
        except asyncio.TimeoutError:
            await _fail(ctx, item_id, piece, emit, f"timed out after {ctx.settings.item_timeout_seconds:.0f}s")
        except asyncio.CancelledError:
            raise
        except Exception as e:  # one failing piece must not stop the others
            log.exception("piece %s failed", piece)
            await _fail(ctx, item_id, piece, emit, f"error: {type(e).__name__}: {e}")
        finally:
            await queue.put(_SENTINEL)

    tasks = [asyncio.create_task(run_one(*job)) for job in jobs]
    remaining = len(tasks)
    try:
        while remaining:
            timeout = deadline - time.monotonic()
            try:
                item = await asyncio.wait_for(queue.get(), timeout=max(0.0, timeout))
            except asyncio.TimeoutError:
                yield ev.text("The request reached its time limit; stopping the remaining pieces.\n")
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                for item_id, piece, _, _ in jobs:
                    status = await _fail_quiet(ctx, item_id, "request timed out")
                    if status:
                        yield ev.Event(type=ev.ITEM_READY, text=f"[Piece {piece}] Needs attention: request timed out.\n",
                                       piece=piece, item_id=item_id,
                                       data={"status": ItemStatus.needs_attention.value, "reason": "request timed out"})
                return
            if item is _SENTINEL:
                remaining -= 1
            else:
                yield item
    finally:
        for t in tasks:  # client disconnected or generator closed early
            if not t.done():
                t.cancel()


async def _fail(ctx: AppContext, item_id: str, piece: int, emit, reason: str) -> None:
    if await _fail_quiet(ctx, item_id, reason):
        await emit(ev.Event(
            type=ev.ITEM_READY, text=f"[Piece {piece}] Needs attention: {reason}.\n",
            piece=piece, item_id=item_id, data={"status": ItemStatus.needs_attention.value, "reason": reason},
        ))


async def _fail_quiet(ctx: AppContext, item_id: str, reason: str) -> bool:
    try:
        set_status(ctx.engine, item_id, ItemStatus.needs_attention, error=reason)
        return True
    except InvalidTransition:  # already finished
        return False


def similarity_pairs(ctx: AppContext, items: list[ContentItem]) -> list[tuple[ContentItem, ContentItem, float]]:
    """(later item, earlier item, score) for every pair; score = mean of headline and image-prompt similarity."""
    briefs = [(i, ContentBrief.model_validate_json(i.brief_json)) for i in items if i.brief_json]
    if len(briefs) < 2:
        return []
    heads = ctx.embedder.embed([b.headline for _, b in briefs])
    prompts = ctx.embedder.embed([b.image_prompt for _, b in briefs])
    pairs = []
    for a in range(len(briefs)):
        for b in range(a + 1, len(briefs)):
            score = (cosine(heads[a], heads[b]) + cosine(prompts[a], prompts[b])) / 2
            pairs.append((briefs[b][0], briefs[a][0], score))
    return pairs


def _find_near_duplicates(ctx: AppContext, items: list[ContentItem | None]) -> list[tuple[ContentItem, ContentItem, float]]:
    ready = [i for i in items if i and i.status == ItemStatus.pending_review.value]
    chosen: dict[str, tuple[ContentItem, ContentItem, float]] = {}
    for later, earlier, score in sorted(similarity_pairs(ctx, ready), key=lambda p: -p[2]):
        if score <= ctx.settings.variety_threshold:
            break
        if later.id in chosen or earlier.id in chosen:
            continue  # regenerate one piece per similar pair
        chosen[later.id] = (later, earlier, score)
    return list(chosen.values())
