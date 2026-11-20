"""Image checks: size rule plus a vision LLM judge against the VisualBrief (spec 5.6)."""

from __future__ import annotations

import json
from pathlib import Path

from app.pipeline.context import AppContext
from app.providers.llm_base import StructuredResult, structured_call
from app.schemas.content import VisualBrief, dump
from app.schemas.llm import ImageJudgement


def check_image_rules(width: int, height: int, target: tuple[int, int]) -> list[str]:
    return [] if (width, height) == target else [f"size {width}x{height} != target {target[0]}x{target[1]}"]


async def judge_image(ctx: AppContext, image_path: Path, visual: VisualBrief) -> StructuredResult[ImageJudgement]:
    payload = {"visual_brief": dump(visual), "image_rules": ctx.kb.image_rules}
    return await structured_call(
        ctx.judge_llm,
        ImageJudgement,
        task="image_judge",
        system=ctx.prompts.get("image_judge"),
        user="Check the attached image.\n```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```",
        context=payload,
        images=[image_path],
    )
