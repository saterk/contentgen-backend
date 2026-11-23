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


def _script(ch: str) -> str | None:
    o = ord(ch)
    if 0x0E00 <= o <= 0x0E7F:
        return "thai"
    if 0x3040 <= o <= 0x30FF or 0x3400 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF or 0xAC00 <= o <= 0xD7AF:
        return "cjk"  # Chinese, Japanese kana, Korean: never expected in our copy
    if ch.isascii() and ch.isalpha():
        return "latin"
    return None


def language_problem(text: str, language: str) -> str | None:
    """Why the copy is not in the requested language, or None.

    LLMs sometimes drift into another language (e.g. Chinese) mid-request. Thai copy may still
    contain English names ("AIA", "Endowment"), so it only needs to be mostly Thai.
    """
    counts = {"thai": 0, "cjk": 0, "latin": 0}
    for ch in text:
        if (sc := _script(ch)) is not None:
            counts[sc] += 1
    if counts["cjk"]:
        return f"contains Chinese/Japanese/Korean characters; write only in {'Thai' if language == 'th' else 'English'}"
    letters = counts["thai"] + counts["latin"]
    if not letters:
        return None
    if language == "th" and counts["thai"] / letters < 0.5:
        return "copy must be in Thai (English only for names such as AIA or product names)"
    if language == "en" and counts["thai"] / letters > 0.1:
        return "copy must be in English"
    return None


def check_copy_rules(
    brief: ContentBrief,
    *,
    product_id: str,
    known_fact_ids: set[str],
    kb: KBFiles,
    settings: Settings,
    language: str | None = None,
) -> list[RuleHit]:
    hits: list[RuleHit] = []

    if language and (problem := language_problem(copy_text(brief), language)):
        hits.append(RuleHit("wrong_language", problem))

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
