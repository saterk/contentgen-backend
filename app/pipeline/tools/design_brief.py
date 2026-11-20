"""The designer tool: one structured LLM call that writes a ContentBrief (spec 5.3)."""

from __future__ import annotations

import json
from typing import Any

from app.pipeline.context import AppContext
from app.providers.llm_base import StructuredResult, structured_call
from app.schemas.content import ContentBrief, ContentRequest
from app.schemas.kb import KBChunk


def knowledge_payload(segment_knowledge: list[KBChunk], product_knowledge: list[KBChunk]) -> dict[str, Any]:
    """Split knowledge into what the brief may cite (facts, disclaimers) and background notes."""
    return {
        "segment_knowledge": [{"section": c.section, "text": c.text} for c in segment_knowledge],
        "product_facts": [
            {"fact_id": c.fact_id, "text": c.text}
            for c in product_knowledge
            if c.type == "product" and c.fact_id
        ],
        "product_notes": [
            {"section": c.section, "text": c.text}
            for c in product_knowledge
            if c.type == "product" and not c.fact_id
        ],
        "disclaimers": [{"disclaimer_id": c.fact_id, "text": c.text} for c in product_knowledge if c.type == "disclaimer"],
    }


async def design_brief(
    ctx: AppContext,
    request: ContentRequest,
    angle: str,
    segment_knowledge: list[KBChunk],
    product_knowledge: list[KBChunk],
    brand_guide: str,
    feedback: list[str] | None = None,
) -> StructuredResult[ContentBrief]:
    s = ctx.settings
    payload = {
        "request": {
            "profile": request.profile.model_dump(),
            "segment_id": request.segment_id,
            "objective": request.objective,
            "audience_note": request.audience_note,
            "journey_stage": request.journey_stage,
            "channel": request.channel,
            "language": request.language,
            "visual_style": request.visual_style,
        },
        "angle": angle,
        **knowledge_payload(segment_knowledge, product_knowledge),
        "length_limits_characters": {
            "headline": s.headline_max_chars,
            "body": s.body_max_chars,
            "cta_text": s.cta_max_chars,
        },
        "banned_terms": ctx.kb.banned_terms,
        "feedback_from_failed_check": feedback or [],
    }
    user = (
        "## Brand guide\n\n"
        + brand_guide
        + "\n\n## Image rules\n\n"
        + ctx.kb.image_rules
        + "\n\n## Input\n\n```json\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n```"
    )
    return await structured_call(
        ctx.llm,
        ContentBrief,
        task="design_brief",
        system=ctx.prompts.get("design_brief"),
        user=user,
        context=payload,
        retries=2,
        temperature=0.7,
    )
