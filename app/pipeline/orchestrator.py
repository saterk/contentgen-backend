"""Orchestrator (spec 5.0–5.2): understand the chat, resolve entities, load knowledge, plan angles.

Prototype note: the step order is fixed in code and the LLM is used for the
judgement calls (extraction, angle planning, briefs). Tool calls are counted
against the spec's limits with ToolBudget. A free tool-choosing agent
(Strands) can replace `understand()` later behind the same return types.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from app.kb.files import TAXONOMY_FIELDS
from app.models import new_id
from app.pipeline.context import AppContext
from app.pipeline.tools import ToolBudget
from app.providers.llm_base import structured_call
from app.runlog import RunLogger
from app.schemas.content import ContentProfile, ContentRequest
from app.schemas.kb import EntityMatch, KBChunk
from app.schemas.llm import AnglePlan, RequirementDraft

CHANNELS = {"line": "line", "email": "email", "e-mail": "email", "social": "social", "facebook": "social",
            "instagram": "social", "app": "app", "ไลน์": "line"}
LANGUAGES = {"th": "th", "thai": "th", "ไทย": "th", "ภาษาไทย": "th", "en": "en", "english": "en", "อังกฤษ": "en"}
SUGGEST_FLOOR = 0.5  # alternatives below this score aren't worth suggesting


@dataclass
class NeedsInput:
    question: str
    missing: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Understood:
    request: ContentRequest
    segment_name: str
    product_name: str
    segment_knowledge: list[KBChunk]
    product_knowledge: list[KBChunk]
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        r = self.request
        p = r.profile
        lines = [
            "Here's what I understood:",
            f"- **Product:** {self.product_name} (`{r.product_id}`)",
            f"- **Segment:** {self.segment_name} (`{r.segment_id}`)",
            f"- **Style / tone / format:** {p.style} / {p.tone} / {p.format} · **Visual style:** {r.visual_style}",
            f"- **Pieces:** {r.count} · **Channel:** {r.channel} · **Language:** {r.language}",
            f"- **Objective:** {r.objective}",
        ]
        lines += [f"- _{n}_" for n in self.notes]
        return "\n".join(lines) + "\n\n"


class Orchestrator:
    def __init__(self, ctx: AppContext, rl: RunLogger, request_id: str) -> None:
        self.ctx = ctx
        self.rl = rl
        self.request_id = request_id
        self.budget = ToolBudget(ctx.settings.request_tool_call_limit, "request phase")

    # ---- 1. extract ----------------------------------------------------
    async def extract(self, messages: list[dict[str, Any]]) -> RequirementDraft:
        convo = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] in ("user", "assistant")]
        tax = self.ctx.kb.taxonomy
        allowed = {
            f"{fld}s": {value: aliases for value, aliases in tax.values[fld].items()} for fld in (*TAXONOMY_FIELDS, "visual_style")
        }
        known = {
            "products": sorted(self.ctx.km.catalog("product").values()),
            "segments": sorted(self.ctx.km.catalog("segment").values()),
        }
        async with self.rl.step(
            "extract_requirement", request_id=self.request_id,
            provider=self.ctx.llm.provider_name, model=self.ctx.llm.model,
        ) as rec:
            res = await structured_call(
                self.ctx.llm,
                RequirementDraft,
                task="extract_requirement",
                system=self.ctx.prompts.get("orchestrator"),
                user="Task: extract_requirement\n\n"
                "Allowed style, tone and format values (key = value to return, list = example wordings):\n"
                "```json\n" + json.dumps(allowed, ensure_ascii=False) + "\n```\n\n"
                "Products and segments we have knowledge for (the user may name them loosely, "
                "e.g. \"about health\" names the product Health):\n"
                "```json\n" + json.dumps(known, ensure_ascii=False) + "\n```\n\n"
                "Conversation:\n```json\n"
                + json.dumps(convo, ensure_ascii=False, indent=2)
                + "\n```",
                context={"messages": convo},
                temperature=0,
            )
            rec.add(res.input_tokens, res.output_tokens, res.cost_usd)
        return res.value

    # ---- 2–4. validate, resolve, load knowledge -------------------------
    async def understand(self, messages: list[dict[str, Any]], conversation_id: str) -> NeedsInput | Understood:
        s = self.ctx.settings
        tax = self.ctx.kb.taxonomy
        draft = await self.extract(messages)
        notes: list[str] = []
        questions: list[str] = []
        missing: list[str] = []

        if not draft.product_text:
            missing.append("product")
            questions.append("Which product is this for?")
        if not draft.segment_text:
            missing.append("segment")
            questions.append("Which customer segment?")

        # Visual style (how the image looks). Optional; defaults to photo.
        visual_style = tax.normalise("visual_style", draft.visual_style)
        if draft.visual_style and not visual_style:
            notes.append(f'visual style "{draft.visual_style}" isn\'t available '
                         f'({", ".join(tax.allowed("visual_style"))}); using {s.default_visual_style}')

        # Content profile. Only an unrecognised value is asked about; a missing one gets a default.
        defaults = {"style": s.default_style, "tone": s.default_tone, "format": tax.allowed("format")[0]}
        profile_vals: dict[str, str] = {}
        for fld in TAXONOMY_FIELDS:
            raw = getattr(draft, fld)
            allowed = tax.allowed(fld)
            value = tax.normalise(fld, raw)
            if value:
                if raw and raw.strip().lower() != value:
                    notes.append(f'{fld} "{raw}" → {value}')
            elif raw and (vs := tax.normalise("visual_style", raw)):
                # e.g. "การ์ตูน" given as the style: it describes the image, not the content style
                visual_style = visual_style or vs
                notes.append(f'"{raw}" is a visual style → images in {vs} style')
                value = defaults[fld]
                notes.append(f"{fld}: {value} (default; tell me if you want another)")
            elif raw and len(allowed) == 1:
                value = allowed[0]  # e.g. "2 ภาพ" as format: there is only one format
                notes.append(f'format: {value} (the only format available)')
            elif raw:
                missing.append(fld)
                questions.append(f'I don\'t recognise the {fld} "{raw}". Pick one of: {", ".join(allowed)}.')
                continue
            else:
                value = defaults[fld]
                if len(allowed) > 1:
                    notes.append(f"{fld}: {value} (default; tell me if you want another)")
            profile_vals[fld] = value
        visual_style = visual_style or s.default_visual_style

        count = draft.count
        if count is None or count < 1:
            missing.append("count")
            questions.append(f"How many pieces would you like (1–{s.max_pieces_per_request})?")
        elif count > s.max_pieces_per_request:
            notes.append(f"You asked for {count} pieces; the prototype makes up to {s.max_pieces_per_request}, so I'll make {s.max_pieces_per_request}.")
            count = s.max_pieces_per_request

        if questions:
            hint = "You can answer like: Product: … Segment: … Style: … Tone: … Make 3 pieces."
            return NeedsInput(
                question="I need a bit more before I start. " + " ".join(questions) + "\n\n" + hint,
                missing=missing,
                data={"extracted": draft.model_dump(exclude_none=True)},
            )

        # Map segment and product to known ids (KM tool).
        self.budget.use("resolve_entities")
        async with self.rl.step("resolve_entities", request_id=self.request_id, provider="km", model=self.ctx.embedder.model):
            resolved = await asyncio.to_thread(
                self.ctx.km.resolve_entities, draft.segment_text or "", draft.product_text or ""
            )
        questions = [q for q in (self._entity_question(resolved.segment, "segment"),
                                 self._entity_question(resolved.product, "product")) if q]
        if questions:
            return NeedsInput(
                question=" ".join(questions),
                missing=[k for k, m in (("segment", resolved.segment), ("product", resolved.product)) if not m.confident],
                data={"resolved": resolved.model_dump(mode="json")},
            )
        segment_id, product_id = resolved.segment.entity_id, resolved.product.entity_id
        assert segment_id and product_id

        channel = CHANNELS.get((draft.channel or "").strip().lower(), None)
        if draft.channel and not channel:
            notes.append(f'channel "{draft.channel}" is not supported yet; using {s.default_channel}')
        language = LANGUAGES.get((draft.language or "").strip().lower(), None)
        if draft.language and not language:
            notes.append(f'language "{draft.language}" is not supported; using {s.default_language}')

        profile = ContentProfile.model_validate(
            {"product": product_id, **profile_vals},
            context={"taxonomy": tax, "product_ids": set(self.ctx.km.catalog("product"))},
        )
        request = ContentRequest(
            id=self.request_id,
            conversation_id=conversation_id,
            segment_text=draft.segment_text or "",
            product_text=draft.product_text or "",
            segment_id=segment_id,
            product_id=product_id,
            journey_stage=draft.journey_stage or s.default_journey_stage,
            profile=profile,
            objective=draft.objective or s.default_objective,
            audience_note=draft.audience_note,
            channel=channel or s.default_channel,
            language=language or s.default_language,
            count=count,
            variants=s.image_variants,
            visual_style=visual_style,
        )

        # Load knowledge once per request; shared by all pieces (KM tools).
        query = " ".join([profile.style, profile.tone, request.objective])
        self.budget.use("get_segment_knowledge")
        async with self.rl.step("get_segment_knowledge", request_id=self.request_id, provider="km", model=self.ctx.embedder.model):
            seg_k = await asyncio.to_thread(self.ctx.km.get_segment_knowledge, segment_id, query)
        self.budget.use("get_product_knowledge")
        async with self.rl.step("get_product_knowledge", request_id=self.request_id, provider="km", model=self.ctx.embedder.model):
            prod_k = await asyncio.to_thread(self.ctx.km.get_product_knowledge, product_id, query)

        return Understood(
            request=request,
            segment_name=resolved.segment.name or segment_id,
            product_name=resolved.product.name or product_id,
            segment_knowledge=seg_k,
            product_knowledge=prod_k,
            notes=notes,
        )

    def _entity_question(self, m: EntityMatch, kind: str) -> str | None:
        if m.confident:
            return None
        names = [m.name] if m.name and m.score >= SUGGEST_FLOOR else []
        names += [a.name for a in m.alternatives if a.score >= SUGGEST_FLOOR]
        names = list(dict.fromkeys(names))[:3]
        if names:
            return f'For the {kind} "{m.text}", did you mean ' + " or ".join(f"*{n}*" for n in names) + "?"
        known = ", ".join(self.ctx.km.catalog(kind).values()) or "none loaded"
        return f'I couldn\'t match the {kind} "{m.text}" to a known {kind}. Known {kind}s: {known}. Which one?'

    # ---- 5. plan angles ------------------------------------------------
    async def plan_angles(self, u: Understood, count: int, avoid: list[str] | None = None) -> list[str]:
        r = u.request
        payload = {
            "count": count,
            "profile": r.profile.model_dump(),
            "segment_id": r.segment_id,
            "objective": r.objective,
            "segment_knowledge": [c.text for c in u.segment_knowledge],
            "avoid": avoid or [],
        }
        async with self.rl.step(
            "plan_angles", request_id=self.request_id, provider=self.ctx.llm.provider_name, model=self.ctx.llm.model
        ) as rec:
            res = await structured_call(
                self.ctx.llm,
                AnglePlan,
                task="plan_angles",
                system=self.ctx.prompts.get("orchestrator"),
                user="Task: plan_angles\n\n```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```",
                context=payload,
                temperature=0.9,
            )
            rec.add(res.input_tokens, res.output_tokens, res.cost_usd)
        angles = [a.strip() for a in res.value.angles if a.strip()][:count]
        while len(angles) < count:  # the model returned too few; pad with marked variations
            angles.append(f"{angles[len(angles) % max(1, len(angles))] if angles else 'General angle'} (variation {len(angles) + 1})")
        return angles


def new_request_id() -> str:
    return new_id("req")
