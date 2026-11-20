"""Rule-based copy checks (spec 5.4.1). Exact matches, no LLM."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.kb.files import KBFiles, contains_term
from app.schemas.content import ContentBrief


@dataclass
class RuleHit:
    rule: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"rule": self.rule, "detail": self.detail}

    def __str__(self) -> str:
        return f"{self.rule}: {self.detail}"


def copy_text(brief: ContentBrief) -> str:
    return "\n".join([brief.headline, brief.body, *brief.benefit_lines, brief.cta_text])


def banned_terms_in(text: str, banned: list[str]) -> list[str]:
    return [t for t in banned if contains_term(text, t)]


def check_copy_rules(
    brief: ContentBrief,
    *,
    product_id: str,
    known_fact_ids: set[str],
    kb: KBFiles,
    settings: Settings,
) -> list[RuleHit]:
    hits: list[RuleHit] = []

    for fid in brief.benefit_fact_ids:
        if fid not in known_fact_ids:
            hits.append(RuleHit("unknown_fact_id", f"{fid} is not an approved fact for {product_id}"))

    disc = kb.disclaimer(brief.disclaimer_id)
    if disc is None:
        hits.append(RuleHit("unknown_disclaimer", f"{brief.disclaimer_id} does not exist in the KB"))
    elif disc.product_id != product_id:
        hits.append(
            RuleHit("disclaimer_product_mismatch", f"{brief.disclaimer_id} belongs to {disc.product_id}, not {product_id}")
        )

    for term in banned_terms_in(copy_text(brief), kb.banned_terms):
        hits.append(RuleHit("banned_term", f"'{term}'"))

    for field, value, limit in (
        ("headline", brief.headline, settings.headline_max_chars),
        ("body", brief.body, settings.body_max_chars),
        ("cta_text", brief.cta_text, settings.cta_max_chars),
    ):
        if len(value) > limit:
            hits.append(RuleHit("too_long", f"{field} is {len(value)} characters (max {limit})"))

    return hits
