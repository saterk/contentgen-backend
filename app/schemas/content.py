"""Content profile, request and brief schemas (spec sections 4.1–4.3).

Allowed style / tone / format values are not hardcoded: they come from
data/kb/taxonomy.json and the KB catalog. Pass them in the validation context:

    ContentProfile.model_validate(data, context={"taxonomy": tax, "product_ids": {...}})

Without a context the fields are only checked to be non-empty strings.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator


def _allowed(info: ValidationInfo, key: str) -> set[str] | None:
    ctx = info.context or {}
    if key == "product":
        ids = ctx.get("product_ids")
        return set(ids) if ids is not None else None
    tax = ctx.get("taxonomy")
    if tax is None:
        return None
    return set(tax.allowed(key))


class ContentProfile(BaseModel):
    product: str = Field(min_length=1, description="product_id from the KB catalog")
    style: str = Field(min_length=1)
    tone: str = Field(min_length=1)
    format: str = Field(min_length=1)

    @field_validator("product", "style", "tone", "format")
    @classmethod
    def _check_allowed(cls, v: str, info: ValidationInfo) -> str:
        allowed = _allowed(info, info.field_name or "")
        if allowed is not None and v not in allowed:
            raise ValueError(f"'{v}' is not an allowed {info.field_name}: {sorted(allowed)}")
        return v


class ContentRequest(BaseModel):
    id: str
    source: Literal["chat"] = "chat"  # "batch" is added in a later phase
    conversation_id: str
    batch_id: str | None = None
    cell_id: str | None = None
    segment_text: str
    product_text: str
    segment_id: str | None
    product_id: str | None
    journey_stage: str
    propensity_band: Literal["high", "mid", "low"] | None = None
    profile: ContentProfile
    objective: str
    audience_note: str | None = None
    channel: Literal["line", "email", "social", "app"]
    language: Literal["th", "en"] = "th"
    visual_style: str = "photo"  # from taxonomy.json visual_styles
    count: int = Field(default=1, ge=1)  # capped by MAX_PIECES_PER_REQUEST in the orchestrator
    variants: int = Field(default=1, ge=1, le=4)
    requested_by: str = "chat_user"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class VisualBrief(BaseModel):
    subject: str = Field(min_length=1)
    mood_story: str = Field(min_length=1)
    light_color: str = Field(min_length=1)
    composition: str = Field(min_length=1)
    must_avoid: list[str] = Field(default_factory=list)


class ContentBrief(BaseModel):
    headline: str = Field(min_length=1)
    body: str = Field(min_length=1)
    benefit_lines: list[str] = Field(min_length=1, max_length=3)
    benefit_fact_ids: list[str] = Field(min_length=1, max_length=3)
    cta_text: str = Field(min_length=1)
    disclaimer_id: str = Field(min_length=1)
    visual: VisualBrief
    image_prompt: str = Field(min_length=1)
    negative_prompt: str | None = None
    language: Literal["th", "en"]
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _each_benefit_cites_a_fact(self) -> "ContentBrief":
        if len(self.benefit_lines) != len(self.benefit_fact_ids):
            raise ValueError(
                "benefit_fact_ids must have one fact id per benefit line "
                f"({len(self.benefit_lines)} lines, {len(self.benefit_fact_ids)} ids)"
            )
        return self


def dump(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")
