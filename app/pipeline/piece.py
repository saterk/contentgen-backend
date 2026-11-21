"""One content piece through steps 5.3–5.6: brief → copy checks → images → image checks."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

from app.models import ImageAsset, ItemStatus, new_id
from app.pipeline import events as ev
from app.pipeline.checks.copy_judge import judge_copy
from app.pipeline.checks.copy_rules import check_copy_rules
from app.pipeline.checks.image_judge import check_image_rules, judge_image
from app.pipeline.context import AppContext
from app.pipeline.i18n import t as tr
from app.pipeline.image_agent import generate_images
from app.pipeline.orchestrator import Understood
from app.pipeline.store import save_check, save_image, set_status
from app.pipeline.tools import ToolBudget
from app.pipeline.tools.design_brief import design_brief
from app.runlog import RunLogger
from app.schemas.content import ContentBrief

Emit = Callable[[ev.Event], Awaitable[None]]

MODE_FULL = "content-creator"
MODE_BRIEF = "content-creator-brief"

JUDGE_OFF_NOTE = "judge disabled (JUDGE_ENABLED=false): rules only"

IMAGE_RETRY_ADJUSTMENT = (
    "Strictly no text, letters, numbers, logos or watermarks anywhere; natural hands and faces; "
    "calm everyday setting with no clinical elements"
)


def brief_markdown(piece: int, brief: ContentBrief, disclaimer_text: str | None, lang: str = "en") -> str:
    lines = [f"**{tr(lang, 'piece', n=piece)} — {brief.headline}**", brief.body]
    lines += [f"• {line} `{fid}`" for line, fid in zip(brief.benefit_lines, brief.benefit_fact_ids)]
    lines.append(f"**CTA:** {brief.cta_text}")
    if disclaimer_text:
        lines.append(f"_{disclaimer_text}_")
    return "\n".join(lines) + "\n\n"


async def process_piece(
    ctx: AppContext,
    rl: RunLogger,
    u: Understood,
    item_id: str,
    piece: int,
    angle: str,
    version: int,
    mode: str,
    emit: Emit,
) -> ItemStatus:
    req = u.request
    s = ctx.settings
    budget = ToolBudget(s.piece_tool_call_limit, f"piece {piece}")
    known_facts = ctx.km.known_fact_ids(req.product_id or "")
    facts_by_id = {c.fact_id: c for c in u.product_knowledge if c.fact_id}

    L = u.lang
    tag = f"[{tr(L, 'piece', n=piece)}]"

    async def step(name: str, label_key: str) -> None:
        await emit(ev.Event(type=ev.STEP, text=f"{tag} {tr(L, label_key)}\n", piece=piece, item_id=item_id, data={"step": name}))

    # ---- copy: brief + checks, one automatic retry --------------------------
    feedback: list[str] = []
    brief: ContentBrief | None = None
    for attempt in (1, 2):
        set_status(ctx.engine, item_id, ItemStatus.generating_copy)
        await step("design_brief", "step_drafting" if attempt == 1 else "step_redrafting")
        budget.use("design_brief")
        async with rl.step("design_brief", request_id=req.id, item_id=item_id,
                           provider=ctx.llm.provider_name, model=ctx.llm.model) as rec:
            res = await design_brief(ctx, req, angle, u.segment_knowledge, u.product_knowledge, ctx.kb.brand_guide, feedback)
            rec.add(res.input_tokens, res.output_tokens, res.cost_usd)
        brief = res.value

        await step("copy_check", "step_copy_check")
        judgement = None
        async with rl.step("copy_check", request_id=req.id, item_id=item_id,
                           provider=ctx.judge_llm.provider_name if s.judge_enabled else "rules",
                           model=ctx.judge_llm.model if s.judge_enabled else None) as rec:
            hits = check_copy_rules(brief, product_id=req.product_id or "", known_fact_ids=known_facts, kb=ctx.kb, settings=s)
            if s.judge_enabled:
                cited = [facts_by_id[f] for f in brief.benefit_fact_ids if f in facts_by_id]
                jres = await judge_copy(ctx, req, brief, cited, hits)
                rec.add(jres.input_tokens, jres.output_tokens, jres.cost_usd)
                judgement = jres.value
        passed = not hits and (judgement is None or judgement.passed)
        save_check(
            ctx.engine, item_id=item_id, item_version=version, stage="copy", attempt=attempt,
            rule_hits=[h.as_dict() for h in hits], judge_score=judgement.score if judgement else None,
            judge_notes=json.dumps(judgement.model_dump(), ensure_ascii=False) if judgement else JUDGE_OFF_NOTE,
            passed=passed,
        )
        if passed:
            break
        feedback = [str(h) for h in hits]
        if judgement:
            feedback += [f"risky claim: {c}" for c in judgement.risky_claims]
            if judgement.score < 3:
                feedback.append(f"judge scores too low: {judgement.model_dump(exclude={'risky_claims'})}")
        set_status(ctx.engine, item_id, ItemStatus.copy_check_failed, brief_json=brief.model_dump_json())
        await emit(ev.Event(
            type=ev.CHECK_FAILED, text=f"{tag} {tr(L, 'copy_failed', findings='; '.join(feedback))}\n",
            piece=piece, item_id=item_id, data={"stage": "copy", "attempt": attempt, "findings": feedback},
        ))
    else:
        return await _needs_attention(ctx, item_id, piece, emit, tr(L, "reason_copy_twice"), L)

    assert brief is not None
    disclaimer = ctx.kb.disclaimer(brief.disclaimer_id)
    set_status(ctx.engine, item_id, ItemStatus.generating_copy, brief_json=brief.model_dump_json())
    await emit(ev.Event(
        type=ev.BRIEF_READY, text=brief_markdown(piece, brief, disclaimer.text if disclaimer else None, L),
        piece=piece, item_id=item_id,
    ))

    if mode == MODE_BRIEF:
        set_status(ctx.engine, item_id, ItemStatus.pending_review)
        await emit(ev.Event(type=ev.ITEM_READY, piece=piece, item_id=item_id, data={"status": ItemStatus.pending_review.value}))
        return ItemStatus.pending_review

    # ---- images + checks, one automatic retry -------------------------------
    any_passed = False
    for attempt in (1, 2):
        set_status(ctx.engine, item_id, ItemStatus.generating_images)
        await step("generate_images", "step_images" if attempt == 1 else "step_reimages")
        async with rl.step("generate_images", request_id=req.id, item_id=item_id,
                           provider=ctx.images.provider_name, model=ctx.images.model) as rec:
            saved = await generate_images(ctx, item_id, version, brief, attempt, visual_style=req.visual_style,
                                          adjustment=IMAGE_RETRY_ADJUSTMENT if attempt == 2 else None)
            rec.add(cost_usd=sum(x.cost_usd for x in saved))

        await step("image_check", "step_image_check")
        findings: list[str] = []
        for img in saved:
            j = None
            async with rl.step("image_check", request_id=req.id, item_id=item_id,
                               provider=ctx.judge_llm.provider_name if s.judge_enabled else "rules",
                               model=ctx.judge_llm.model if s.judge_enabled else None) as rec:
                rule_hits = check_image_rules(img.width, img.height, s.image_wh)
                if s.judge_enabled:
                    jres = await judge_image(ctx, ctx.storage.local_path(img.file_name), brief.visual)
                    rec.add(jres.input_tokens, jres.output_tokens, jres.cost_usd)
                    j = jres.value
            ok = not rule_hits and (j is None or j.passed)
            any_passed = any_passed or ok
            if not ok:
                findings.append(f"variant {img.variant_no}: " + "; ".join(rule_hits + (_image_findings(j) if j else [])))
            save_image(ctx.engine, ImageAsset(
                id=new_id("img"), item_id=item_id, item_version=version, path=img.file_name, prompt=img.prompt,
                negative_prompt=img.negative_prompt, provider=ctx.images.provider_name, model=ctx.images.model,
                seed=img.seed, variant_no=img.variant_no, attempt=attempt, width=img.width, height=img.height,
                check_json=json.dumps({"rules": rule_hits, "judge": j.model_dump() if j else JUDGE_OFF_NOTE},
                                      ensure_ascii=False), passed=ok,
            ))
        save_check(
            ctx.engine, item_id=item_id, item_version=version, stage="image", attempt=attempt,
            rule_hits=[{"detail": f} for f in findings], judge_score=None,
            judge_notes=None if any_passed else "no variant passed", passed=any_passed,
        )
        if any_passed:
            break
        set_status(ctx.engine, item_id, ItemStatus.image_check_failed)
        await emit(ev.Event(
            type=ev.CHECK_FAILED, text=f"{tag} {tr(L, 'image_failed', findings=' | '.join(findings))}\n",
            piece=piece, item_id=item_id, data={"stage": "image", "attempt": attempt, "findings": findings},
        ))
    else:
        return await _needs_attention(ctx, item_id, piece, emit, tr(L, "reason_images"), L)

    await emit(ev.Event(type=ev.IMAGES_READY, piece=piece, item_id=item_id, data={"count": len(saved)}))
    set_status(ctx.engine, item_id, ItemStatus.pending_review)
    await emit(ev.Event(
        type=ev.ITEM_READY, text=f"{tag} {tr(L, 'ready')}\n",
        piece=piece, item_id=item_id, data={"status": ItemStatus.pending_review.value},
    ))
    return ItemStatus.pending_review


def _image_findings(j) -> list[str]:
    out = []
    if not j.matches_subject:
        out.append("does not match subject")
    if j.must_avoid_present:
        out.append(f"contains {', '.join(j.must_avoid_present)}")
    if j.has_text_or_logo:
        out.append("visible text or logo")
    if j.distortions:
        out.append(f"distortions: {', '.join(j.distortions)}")
    if not j.suitable_for_insurance_ad:
        out.append("not suitable for an insurance ad")
    return out


async def _needs_attention(
    ctx: AppContext, item_id: str, piece: int, emit: Emit, reason: str, lang: str = "en"
) -> ItemStatus:
    set_status(ctx.engine, item_id, ItemStatus.needs_attention, error=reason)
    await emit(ev.Event(
        type=ev.ITEM_READY, text=f"[{tr(lang, 'piece', n=piece)}] {tr(lang, 'needs_attention', reason=reason)}\n",
        piece=piece, item_id=item_id, data={"status": ItemStatus.needs_attention.value, "reason": reason},
    ))
    return ItemStatus.needs_attention
