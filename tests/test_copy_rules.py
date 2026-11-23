from __future__ import annotations

from app.pipeline.checks.copy_rules import check_copy_rules
from app.schemas.content import ContentBrief
from tests.test_schemas import brief_data


def rules(ctx, **overrides):
    brief = ContentBrief.model_validate(brief_data(**overrides))
    return check_copy_rules(
        brief,
        product_id="critical_illness",
        known_fact_ids=ctx.km.known_fact_ids("critical_illness"),
        kb=ctx.kb,
        settings=ctx.settings,
    )


def test_clean_brief_passes(ctx):
    assert rules(ctx) == []


def test_unknown_fact_id(ctx):
    hits = rules(ctx, benefit_fact_ids=["CI-F99"])
    assert [h.rule for h in hits] == ["unknown_fact_id"]


def test_missing_disclaimer(ctx):
    hits = rules(ctx, disclaimer_id="DISC-NOPE")
    assert [h.rule for h in hits] == ["unknown_disclaimer"]


def test_disclaimer_for_another_product(ctx):
    ctx.kb.disclaimers.append(type(ctx.kb.disclaimers[0])(id="DISC-SAV-01", product_id="savings", text="x"))
    hits = rules(ctx, disclaimer_id="DISC-SAV-01")
    assert [h.rule for h in hits] == ["disclaimer_product_mismatch"]


def test_banned_terms_english_and_thai(ctx):
    hits = rules(ctx, headline="Guaranteed payout", body="คุ้มครอง 100% การันตี")
    assert {h.detail for h in hits if h.rule == "banned_term"} == {"'guaranteed'", "'100%'", "'การันตี'"}


def test_banned_terms_match_whole_words_only(ctx):
    # "guarantee" is banned, but must not fire inside unrelated words
    assert rules(ctx, body="Unguaranteeable is not a word we use.") == []


def test_length_limits(ctx):
    hits = rules(ctx, headline="x" * (ctx.settings.headline_max_chars + 1), cta_text="y" * (ctx.settings.cta_max_chars + 1))
    assert sorted(h.detail.split()[0] for h in hits if h.rule == "too_long") == ["cta_text", "headline"]


def test_language_problem():
    from app.pipeline.checks.copy_rules import language_problem

    assert language_problem("ออมเงินอย่างมีวินัย กับ AIA Endowment", "th") is None
    assert "Chinese" in language_problem("守护孩子的未来，从每一顿温馨的晚餐开始", "th")
    assert "Chinese" in language_problem("วางแผนการศึกษา 养成定期储蓄习惯", "th")
    assert "Thai" in language_problem("Save with life cover and a maturity payout", "th")
    assert language_problem("Save with life cover and a maturity payout", "en") is None
    assert "English" in language_problem("ออมเงินอย่างมีวินัย พร้อมความคุ้มครอง", "en")
