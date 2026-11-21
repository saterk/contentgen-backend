"""Read-only REST endpoints for the prototype (spec 7.2)."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from sqlmodel import Session, col, select

from app.api.deps import get_ctx, require_app_token
from app.api.errors import APIError
from app.models import CheckResult, ContentItem, ContentRequestRow, ImageAsset, RunLog
from app.pipeline.context import AppContext

router = APIRouter(prefix="/api", dependencies=[Depends(require_app_token)])


def _item_summary(item: ContentItem) -> dict:
    return {
        "id": item.id,
        "request_id": item.request_id,
        "conversation_id": item.conversation_id,
        "piece": item.piece_no,
        "content_id": item.content_id,
        "angle": item.angle,
        "status": item.status,
        "version": item.version,
        "error": item.error,
        "tags": json.loads(item.tags_json or "{}"),
        "brief": json.loads(item.brief_json) if item.brief_json else None,
        "created_at": item.created_at.isoformat(),
        "updated_at": item.updated_at.isoformat(),
    }


def _image(img: ImageAsset, ctx: AppContext) -> dict:
    return {
        "id": img.id,
        "url": ctx.storage.url(img.path),
        "item_version": img.item_version,
        "variant_no": img.variant_no,
        "attempt": img.attempt,
        "width": img.width,
        "height": img.height,
        "prompt": img.prompt,
        "negative_prompt": img.negative_prompt,
        "provider": img.provider,
        "model": img.model,
        "seed": img.seed,
        "passed": img.passed,
        "check": json.loads(img.check_json or "{}"),
    }


@router.get("/items/{item_id}")
def get_item(item_id: str, ctx: AppContext = Depends(get_ctx)) -> dict:
    with Session(ctx.engine) as s:
        item = s.get(ContentItem, item_id)
        if item is None:
            raise APIError(404, f"Item '{item_id}' not found", "invalid_request_error", "not_found")
        req = s.get(ContentRequestRow, item.request_id)
        images = s.exec(select(ImageAsset).where(ImageAsset.item_id == item_id).order_by(col(ImageAsset.created_at))).all()
        checks = s.exec(select(CheckResult).where(CheckResult.item_id == item_id).order_by(col(CheckResult.at))).all()
        logs = s.exec(select(RunLog).where(RunLog.item_id == item_id).order_by(col(RunLog.at))).all()
        disclaimer = None
        out = _item_summary(item)
        if out["brief"]:
            d = ctx.kb.disclaimer(out["brief"].get("disclaimer_id", ""))
            disclaimer = {"id": d.id, "text": d.text} if d else None
        current = [i for i in images if i.item_version == item.version]
        return {
            **out,
            "request": json.loads(req.request_json) if req else None,
            "disclaimer": disclaimer,
            "images": [_image(i, ctx) for i in current],
            "chosen_image_id": next((i.id for i in current if i.passed), None),
            "image_history": [_image(i, ctx) for i in images if i.item_version != item.version],
            "checks": [
                {
                    "stage": c.stage,
                    "item_version": c.item_version,
                    "attempt": c.attempt,
                    "passed": c.passed,
                    "rule_hits": json.loads(c.rule_hits_json or "[]"),
                    "judge_score": c.judge_score,
                    "judge_notes": json.loads(c.judge_notes) if c.judge_notes and c.judge_notes.startswith("{") else c.judge_notes,
                    "at": c.at.isoformat(),
                }
                for c in checks
            ],
            "run_logs": [
                {
                    "step": r.step, "provider": r.provider, "model": r.model,
                    "prompt_version": r.prompt_version, "kb_version": r.kb_version,
                    "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                    "cost_usd": r.cost_usd, "latency_ms": r.latency_ms, "error": r.error,
                    "at": r.at.isoformat(),
                }
                for r in logs
            ],
            "totals": {
                "cost_usd": round(sum(r.cost_usd for r in logs), 6),
                "latency_ms": sum(r.latency_ms for r in logs),
                "input_tokens": sum(r.input_tokens for r in logs),
                "output_tokens": sum(r.output_tokens for r in logs),
            },
        }


@router.get("/conversations/{conversation_id}/items")
def conversation_items(conversation_id: str, ctx: AppContext = Depends(get_ctx)) -> dict:
    with Session(ctx.engine) as s:
        items = s.exec(
            select(ContentItem)
            .where(ContentItem.conversation_id == conversation_id)
            .order_by(col(ContentItem.created_at), col(ContentItem.piece_no))
        ).all()
        return {"conversation_id": conversation_id, "items": [_item_summary(i) for i in items]}
