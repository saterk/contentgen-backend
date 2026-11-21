"""Structured outputs the LLM must return for each orchestrator / check task."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RequirementDraft(BaseModel):
    """What the orchestrator extracted from the chat. Free text; normalised afterwards."""

    # request: asks for content (new, or referring back to an earlier one)
    # answer:  answers the assistant's clarifying question
    # chat:    anything else (greeting, "who are you", a question) -> reply only, nothing generated
    intent: Literal["request", "answer", "chat"] = "request"
    reply: str | None = None  # for intent=chat: the assistant's short answer, in the user's language
    product_text: str | None = None
    segment_text: str | None = None
    style: str | None = None
    tone: str | None = None
    format: str | None = None
    visual_style: str | None = None  # look of the image: photo, cartoon, 3d …
    count: int | None = None
    channel: str | None = None
    objective: str | None = None
    journey_stage: str | None = None
    language: str | None = None
    audience_note: str | None = None


class AnglePlan(BaseModel):
    angles: list[str] = Field(min_length=1)


class CopyJudgement(BaseModel):
    profile_adherence: int = Field(ge=1, le=5)
    factual_support: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    risky_claims: list[str] = Field(default_factory=list)
    notes: str = ""

    @property
    def passed(self) -> bool:
        return not self.risky_claims and min(
            self.profile_adherence, self.factual_support, self.clarity
        ) >= 3

    @property
    def score(self) -> float:
        return round((self.profile_adherence + self.factual_support + self.clarity) / 3, 2)


class ImageJudgement(BaseModel):
    matches_subject: bool
    must_avoid_present: list[str] = Field(default_factory=list)
    has_text_or_logo: bool
    distortions: list[str] = Field(default_factory=list)
    suitable_for_insurance_ad: bool
    score: int = Field(ge=1, le=5)
    notes: str = ""

    @property
    def passed(self) -> bool:
        return (
            self.matches_subject
            and not self.must_avoid_present
            and not self.has_text_or_logo
            and not self.distortions
            and self.suitable_for_insurance_ad
        )
