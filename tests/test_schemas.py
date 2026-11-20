from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.status import ItemStatus as S
from app.models.status import can_transition
from app.pipeline.tools import ToolBudget, ToolBudgetExceeded
from app.schemas.content import ContentBrief, ContentProfile, ContentRequest


def brief_data(**overrides) -> dict:
    data = {
        "headline": "Plan ahead",
        "body": "Get ready today.",
        "benefit_lines": ["Lump sum on diagnosis"],
        "benefit_fact_ids": ["CI-F01"],
        "cta_text": "Talk to an advisor",
        "disclaimer_id": "DISC-CI-01",
        "visual": {
            "subject": "A couple at home",
            "mood_story": "Cozy",
            "light_color": "Warm light",
            "composition": "4:5",
            "must_avoid": ["text in image"],
        },
        "image_prompt": "A couple at home, warm light",
        "negative_prompt": None,
        "language": "en",
        "rationale": "story / warm",
    }
    data.update(overrides)
    return data


def test_profile_checks_values_against_taxonomy(ctx):
    vctx = {"taxonomy": ctx.kb.taxonomy, "product_ids": {"critical_illness"}}
    ok = ContentProfile.model_validate(
        {"product": "critical_illness", "style": "story", "tone": "warm", "format": "single_image_cta"}, context=vctx
    )
    assert ok.style == "story"
    with pytest.raises(ValidationError, match="not an allowed tone"):
        ContentProfile.model_validate(
            {"product": "critical_illness", "style": "story", "tone": "angry", "format": "single_image_cta"}, context=vctx
        )
    with pytest.raises(ValidationError, match="not an allowed product"):
        ContentProfile.model_validate(
            {"product": "savings", "style": "story", "tone": "warm", "format": "single_image_cta"}, context=vctx
        )


def test_profile_without_context_only_needs_strings():
    p = ContentProfile(product="anything", style="x", tone="y", format="z")
    assert p.product == "anything"
    with pytest.raises(ValidationError):
        ContentProfile(product="", style="x", tone="y", format="z")


def test_brief_requires_one_fact_id_per_benefit_line():
    ContentBrief.model_validate(brief_data())
    with pytest.raises(ValidationError, match="one fact id per benefit line"):
        ContentBrief.model_validate(brief_data(benefit_lines=["a", "b"], benefit_fact_ids=["CI-F01"]))


def test_brief_limits_benefit_lines_to_three():
    with pytest.raises(ValidationError):
        ContentBrief.model_validate(brief_data(benefit_lines=["a"] * 4, benefit_fact_ids=["CI-F01"] * 4))
    with pytest.raises(ValidationError):
        ContentBrief.model_validate(brief_data(benefit_lines=[], benefit_fact_ids=[]))


def test_brief_language_is_th_or_en():
    with pytest.raises(ValidationError):
        ContentBrief.model_validate(brief_data(language="fr"))


def test_content_request_is_chat_only_and_count_positive():
    base = dict(
        id="req_1", conversation_id="c", segment_text="s", product_text="p", segment_id="s", product_id="p",
        journey_stage="awareness", profile=ContentProfile(product="p", style="story", tone="warm", format="f"),
        objective="o", channel="line",
    )
    r = ContentRequest(**base)
    assert r.source == "chat" and r.batch_id is None and r.count == 1
    with pytest.raises(ValidationError):
        ContentRequest(**base, source="batch")
    with pytest.raises(ValidationError):
        ContentRequest(**base, count=0)


@pytest.mark.parametrize(
    ("cur", "new", "ok"),
    [
        (S.queued, S.generating_copy, True),
        (S.generating_copy, S.copy_check_failed, True),
        (S.copy_check_failed, S.generating_copy, True),
        (S.generating_copy, S.generating_images, True),
        (S.generating_copy, S.pending_review, True),  # brief-only mode
        (S.generating_images, S.image_check_failed, True),
        (S.image_check_failed, S.generating_images, True),
        (S.generating_images, S.pending_review, True),
        (S.pending_review, S.generating_copy, True),  # variety regeneration
        (S.image_check_failed, S.needs_attention, True),
        (S.queued, S.pending_review, False),
        (S.pending_review, S.needs_attention, False),
        (S.needs_attention, S.generating_copy, False),
        (S.copy_check_failed, S.generating_images, False),
    ],
)
def test_status_transitions(cur, new, ok):
    assert can_transition(cur, new) is ok


def test_tool_budget():
    b = ToolBudget(2, "piece 1")
    b.use("design_brief")
    b.use("get_product_knowledge")
    assert b.remaining == 0
    with pytest.raises(ToolBudgetExceeded, match="limit of 2"):
        b.use("design_brief")
