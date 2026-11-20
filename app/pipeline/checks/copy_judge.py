"""LLM judge for copy (spec 5.4.2)."""

from __future__ import annotations

import json

from app.pipeline.checks.copy_rules import RuleHit
from app.pipeline.context import AppContext
from app.providers.llm_base import StructuredResult, structured_call
from app.schemas.content import ContentBrief, ContentRequest, dump
from app.schemas.kb import KBChunk
from app.schemas.llm import CopyJudgement


async def judge_copy(
    ctx: AppContext,
    request: ContentRequest,
    brief: ContentBrief,
    cited_facts: list[KBChunk],
    rule_hits: list[RuleHit],
) -> StructuredResult[CopyJudgement]:
    payload = {
        "profile": dump(request.profile),
        "segment_id": request.segment_id,
        "language": request.language,
        "brief": dump(brief),
        "cited_facts": [{"fact_id": c.fact_id, "text": c.text} for c in cited_facts],
        "banned_hits": [h.detail.strip("'") for h in rule_hits if h.rule == "banned_term"],
    }
    return await structured_call(
        ctx.judge_llm,
        CopyJudgement,
        task="copy_judge",
        system=ctx.prompts.get("copy_judge"),
        user="```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```",
        context=payload,
    )
