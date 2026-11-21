from __future__ import annotations

import asyncio
import json

from sqlmodel import Session, select

from app.models import CheckResult, ContentItem, ImageAsset, RunLog
from app.pipeline.context import AppContext
from app.pipeline.runner import similarity_pairs
from app.providers.llm_mock import MockLLM
from tests.conftest import FULL_REQUEST, make_settings, of_type, run


def items_of(ctx: AppContext, events) -> list[ContentItem]:
    ids = of_type(events, "done")[0].data["item_ids"]
    with Session(ctx.engine) as s:
        return [s.get(ContentItem, i) for i in ids]


# ---- happy path -------------------------------------------------------------


def test_three_pieces_end_to_end(ctx):
    events = run(ctx, FULL_REQUEST.format(n=3))
    types = [e.type for e in events if e.type]
    assert types[0] == "requirement" and types[1] == "plan" and types[-1] == "done"

    req = of_type(events, "requirement")[0].data["request"]
    assert req["segment_id"] == "young_families" and req["product_id"] == "critical_illness"
    assert req["profile"] == {"product": "critical_illness", "style": "story", "tone": "warm", "format": "single_image_cta"}
    assert req["count"] == 3 and req["channel"] == "line" and req["language"] == "th"

    plan = of_type(events, "plan")[0].data
    assert plan["count"] == 3 and len({p["angle"] for p in plan["pieces"]}) == 3

    ready = of_type(events, "item_ready")
    assert sorted(e.piece for e in ready) == [1, 2, 3]
    assert all(e.data["status"] == "pending_review" for e in ready)
    assert len(of_type(events, "brief_ready")) == 3 and len(of_type(events, "images_ready")) == 3

    done = of_type(events, "done")[0].data
    assert done["summary"] == {"total": 3, "pending_review": 3, "needs_attention": 0}
    assert done["usage"]["prompt_tokens"] > 0

    items = items_of(ctx, events)
    headlines = {json.loads(i.brief_json)["headline"] for i in items}
    assert len(headlines) == 3, "pieces should differ"
    tags = json.loads(items[0].tags_json)
    assert tags["style"] == "story" and tags["segment"] == "young_families" and tags["content_id"].startswith("V-")
    with Session(ctx.engine) as s:
        for it in items:
            imgs = s.exec(select(ImageAsset).where(ImageAsset.item_id == it.id)).all()
            assert imgs and all(img.passed for img in imgs)
            assert all((ctx.settings.images_dir / img.path).exists() for img in imgs)
            assert {(img.width, img.height) for img in imgs} == {ctx.settings.image_wh}
            stages = {c.stage for c in s.exec(select(CheckResult).where(CheckResult.item_id == it.id)).all()}
            assert stages == {"copy", "image"}
        steps = {r.step for r in s.exec(select(RunLog)).all()}
        assert {"extract_requirement", "resolve_entities", "get_segment_knowledge", "get_product_knowledge",
                "plan_angles", "design_brief", "copy_check", "generate_images", "image_check"} <= steps


def test_brief_only_mode_makes_no_images(ctx):
    events = run(ctx, FULL_REQUEST.format(n=2), mode="content-creator-brief")
    assert [e.data["status"] for e in of_type(events, "item_ready")] == ["pending_review"] * 2
    assert not of_type(events, "images_ready")
    with Session(ctx.engine) as s:
        assert s.exec(select(ImageAsset)).all() == []


def test_count_above_cap_is_clamped(ctx):
    events = run(ctx, FULL_REQUEST.format(n=15))
    assert of_type(events, "requirement")[0].data["request"]["count"] == 10
    assert "up to 10" in of_type(events, "requirement")[0].text
    assert len(of_type(events, "item_ready")) == 10
    assert len({i.angle for i in items_of(ctx, events)}) == 10


def test_close_tone_is_mapped_and_reported(ctx):
    events = run(ctx, FULL_REQUEST.format(n=1).replace("Tone: warm", "Tone: friendly and warm"))
    req = of_type(events, "requirement")[0]
    assert req.data["request"]["profile"]["tone"] == "warm"
    assert "friendly and warm" in req.text


# ---- clarifying questions ---------------------------------------------------


def test_missing_fields_ask_one_question_and_stop(ctx):
    events = run(ctx, "Segment: young families. Style: story")
    assert [e.type for e in events if e.type] == ["needs_input"]
    ni = events[-1]
    assert set(ni.data["missing"]) == {"product", "count"}, "tone and format get defaults, not questions"
    assert "How many pieces" in ni.text
    with Session(ctx.engine) as s:
        assert s.exec(select(ContentItem)).all() == []


def test_answering_the_question_continues_the_conversation(ctx):
    first = "Product: CI. Segment: young families. Style: story. Tone: warm. Format: single image"
    q = run(ctx, first)[-1]
    assert q.type == "needs_input" and q.data["missing"] == ["count"]
    events = run(ctx, [
        {"role": "user", "content": first},
        {"role": "assistant", "content": q.text},
        {"role": "user", "content": "2"},
    ])
    assert of_type(events, "requirement")[0].data["request"]["segment_id"] == "young_families"
    assert len(of_type(events, "item_ready")) == 2


def test_unknown_segment_asks_and_lists_known_segments(ctx):
    msg = FULL_REQUEST.format(n=1).replace("young families", "pet owners")
    q = run(ctx, msg)[-1]
    assert q.type == "needs_input" and q.data["missing"] == ["segment"]
    assert "pet owners" in q.text and "Young Families" in q.text
    events = run(ctx, [
        {"role": "user", "content": msg},
        {"role": "assistant", "content": q.text},
        {"role": "user", "content": "young families"},
    ])
    assert of_type(events, "requirement")[0].data["request"]["segment_id"] == "young_families"


def test_ambiguous_segment_asks_did_you_mean(ctx):
    msg = FULL_REQUEST.format(n=1).replace("young families", "young families or first jobbers")
    q = run(ctx, msg)[-1]
    assert q.type == "needs_input" and "did you mean" in q.text
    assert "Young Families" in q.text and "First Jobber" in q.text


def test_unrecognised_style_lists_allowed_values(ctx):
    q = run(ctx, FULL_REQUEST.format(n=1).replace("Style: story", "Style: meme"))[-1]
    assert q.type == "needs_input" and q.data["missing"] == ["style"]
    assert '"meme"' in q.text and "testimonial" in q.text


# ---- checks, retries and failures -------------------------------------------


class BannedWordLLM(MockLLM):
    """Writes a banned term into the headline for the first `bad_calls` design_brief calls."""

    def __init__(self, bad_calls: int) -> None:
        super().__init__()
        self.bad_calls = bad_calls
        self.calls = 0

    def respond(self, task, ctx):
        out = super().respond(task, ctx)
        if task == "design_brief":
            self.calls += 1
            if self.calls <= self.bad_calls:
                out["headline"] = "Guaranteed peace of mind"
        return out


def test_copy_check_failure_retries_once_then_passes(ctx):
    ctx.llm = BannedWordLLM(bad_calls=1)
    events = run(ctx, FULL_REQUEST.format(n=1))
    failed = of_type(events, "check_failed")
    assert len(failed) == 1 and failed[0].data["stage"] == "copy"
    assert any("guaranteed" in f for f in failed[0].data["findings"])
    assert of_type(events, "item_ready")[0].data["status"] == "pending_review"
    item = items_of(ctx, events)[0]
    assert "Guaranteed" not in json.loads(item.brief_json)["headline"]
    with Session(ctx.engine) as s:
        checks = s.exec(select(CheckResult).where(CheckResult.item_id == item.id, CheckResult.stage == "copy")).all()
        assert [c.passed for c in sorted(checks, key=lambda c: c.attempt)] == [False, True]


def test_copy_check_failing_twice_needs_attention_and_skips_images(ctx):
    ctx.llm = BannedWordLLM(bad_calls=99)
    events = run(ctx, FULL_REQUEST.format(n=1))
    assert len(of_type(events, "check_failed")) == 2
    ready = of_type(events, "item_ready")[0]
    assert ready.data["status"] == "needs_attention"
    assert not of_type(events, "images_ready")
    assert of_type(events, "done")[0].data["summary"]["needs_attention"] == 1


class TextInImageJudge(MockLLM):
    def respond(self, task, ctx):
        out = super().respond(task, ctx)
        if task == "image_judge":
            out["has_text_or_logo"] = True
        return out


def test_image_check_failure_retries_once_then_needs_attention(ctx):
    ctx.judge_llm = TextInImageJudge()
    events = run(ctx, FULL_REQUEST.format(n=1))
    failed = of_type(events, "check_failed")
    assert [f.data["stage"] for f in failed] == ["image", "image"]
    assert of_type(events, "item_ready")[0].data["status"] == "needs_attention"
    with Session(ctx.engine) as s:
        prompts = [i.prompt for i in s.exec(select(ImageAsset)).all()]
    assert len(prompts) == 2 and "Strictly no text" in max(prompts, key=len), "retry uses an adjusted prompt"


class SlowLLM(MockLLM):
    async def complete(self, **kw):
        if kw.get("task") == "design_brief":
            await asyncio.sleep(1)
        return await super().complete(**kw)


def test_piece_timeout_marks_needs_attention(data_dir):
    s = make_settings(data_dir, item_timeout_seconds=0.2)
    from app.kb.ingest import ingest_with_settings

    ingest_with_settings(s, kb_version="t")
    ctx = AppContext.build(s)
    ctx.llm = SlowLLM()
    events = run(ctx, FULL_REQUEST.format(n=2))
    ready = of_type(events, "item_ready")
    assert [e.data["status"] for e in ready] == ["needs_attention"] * 2
    assert all("timed out" in e.data["reason"] for e in ready)


class CrashingLLM(MockLLM):
    def respond(self, task, ctx):
        if task == "design_brief" and ctx["angle"].startswith("A parent making breakfast"):
            raise RuntimeError("provider exploded")
        return super().respond(task, ctx)


def test_one_failing_piece_does_not_stop_the_others(ctx):
    ctx.llm = CrashingLLM()
    events = run(ctx, FULL_REQUEST.format(n=3))
    statuses = sorted(e.data["status"] for e in of_type(events, "item_ready"))
    assert statuses == ["needs_attention", "pending_review", "pending_review"]


# ---- variety check ----------------------------------------------------------


class DuplicateBriefLLM(MockLLM):
    """Writes the same brief for every angle of the first plan; new angles get normal briefs."""

    def __init__(self) -> None:
        super().__init__()
        self.first_plan: list[str] | None = None

    def respond(self, task, ctx):
        out = super().respond(task, ctx)
        if task == "plan_angles" and self.first_plan is None:
            self.first_plan = out["angles"]
        if task == "design_brief" and self.first_plan and ctx["angle"] in self.first_plan:
            out = super().respond(task, {**ctx, "angle": self.first_plan[0]})
        return out


def test_near_duplicates_are_regenerated_with_new_angles(ctx):
    ctx.llm = DuplicateBriefLLM()
    events = run(ctx, FULL_REQUEST.format(n=3))
    assert "too similar" in "".join(e.text for e in events)
    items = items_of(ctx, events)
    assert [i.version for i in sorted(items, key=lambda i: i.piece_no)] == [1, 2, 2]
    assert all(i.status == "pending_review" for i in items)
    assert len({json.loads(i.brief_json)["headline"] for i in items}) == 3
    assert max(p[2] for p in similarity_pairs(ctx, items)) <= ctx.settings.variety_threshold
    assert of_type(events, "done")[0].data["summary"]["pending_review"] == 3


# ---- judge disabled ---------------------------------------------------------


class ExplodingJudge(MockLLM):
    def respond(self, task, ctx):
        raise AssertionError(f"judge must not be called when JUDGE_ENABLED=false (task {task})")


def judge_off_ctx(data_dir) -> AppContext:
    from app.kb.ingest import ingest_with_settings

    s = make_settings(data_dir, judge_enabled=False)
    ingest_with_settings(s, kb_version="t")
    ctx = AppContext.build(s)
    ctx.judge_llm = ExplodingJudge()
    return ctx


def test_judge_disabled_uses_rules_only(data_dir):
    ctx = judge_off_ctx(data_dir)
    events = run(ctx, FULL_REQUEST.format(n=2))
    assert [e.data["status"] for e in of_type(events, "item_ready")] == ["pending_review"] * 2
    with Session(ctx.engine) as s:
        checks = s.exec(select(CheckResult)).all()
        assert {c.stage for c in checks} == {"copy", "image"}
        assert all(c.judge_score is None for c in checks)
        assert all("judge disabled" in (c.judge_notes or "") for c in checks if c.stage == "copy")
        logs = s.exec(select(RunLog).where(RunLog.step.in_(["copy_check", "image_check"]))).all()
        assert logs and {r.provider for r in logs} == {"rules"}


def test_judge_disabled_rules_still_catch_banned_terms(data_dir):
    ctx = judge_off_ctx(data_dir)
    ctx.llm = BannedWordLLM(bad_calls=1)
    events = run(ctx, FULL_REQUEST.format(n=1))
    assert of_type(events, "check_failed")[0].data["stage"] == "copy"
    assert of_type(events, "item_ready")[0].data["status"] == "pending_review"


# ---- defaults and visual style ---------------------------------------------


def test_missing_style_tone_format_use_defaults(ctx):
    events = run(ctx, "Product: CI. Segment: young families. Make 1 piece.")
    req = of_type(events, "requirement")[0]
    p = req.data["request"]["profile"]
    assert (p["style"], p["tone"], p["format"]) == (ctx.settings.default_style, ctx.settings.default_tone, "single_image_cta")
    assert req.data["request"]["visual_style"] == "photo"
    assert "style: story (default" in req.text and "tone: warm (default" in req.text
    assert len(of_type(events, "item_ready")) == 1


def test_cartoon_given_as_style_becomes_visual_style(ctx):
    events = run(ctx, "Product: Endowment. Segment: young families. Style: การ์ตูน. Tone: เป็นกันเอง. Format: 2 ภาพ. Make 1 piece.")
    req = of_type(events, "requirement")[0]
    r = req.data["request"]
    assert r["visual_style"] == "cartoon"
    assert r["profile"]["style"] == "story" and r["profile"]["tone"] == "friendly" and r["profile"]["format"] == "single_image_cta"
    assert "แนวภาพ" in req.text, "Thai words in the message -> Thai summary"
    with Session(ctx.engine) as s:
        prompt = s.exec(select(ImageAsset)).one().prompt
        tags = json.loads(s.exec(select(ContentItem)).one().tags_json)
    assert "flat 2D cartoon illustration" in prompt and "photorealistic lifestyle photography" not in prompt
    assert tags["visual_style"] == "cartoon"


def test_explicit_visual_style_label(ctx):
    events = run(ctx, FULL_REQUEST.format(n=1) + " Visual style: 3D.")
    assert of_type(events, "requirement")[0].data["request"]["visual_style"] == "3d"


def test_unknown_style_that_is_not_visual_still_asks(ctx):
    q = run(ctx, FULL_REQUEST.format(n=1).replace("Style: story", "Style: meme"))[-1]
    assert q.type == "needs_input" and q.data["missing"] == ["style"]


# ---- chat vs request, and request scope -------------------------------------


def test_chat_message_gets_a_reply_and_generates_nothing(ctx):
    events = run(ctx, "who are you")
    assert [e.type for e in events if e.type] == ["reply"]
    assert "content creator" in events[-1].text
    with Session(ctx.engine) as s:
        assert s.exec(select(ContentItem)).all() == []


def test_chat_after_a_finished_request_does_not_repeat_it(ctx):
    first = run(ctx, FULL_REQUEST.format(n=1))
    done_text = "".join(e.text for e in first)
    events = run(ctx, [
        {"role": "user", "content": FULL_REQUEST.format(n=1)},
        {"role": "assistant", "content": done_text},
        {"role": "user", "content": "who are you"},
    ])
    assert [e.type for e in events if e.type] == ["reply"]
    with Session(ctx.engine) as s:
        assert len(s.exec(select(ContentItem)).all()) == 1, "no new pieces were generated"


def test_new_request_after_done_does_not_inherit_old_fields(ctx):
    first = run(ctx, FULL_REQUEST.format(n=1))
    events = run(ctx, [
        {"role": "user", "content": FULL_REQUEST.format(n=1)},
        {"role": "assistant", "content": "".join(e.text for e in first)},
        {"role": "user", "content": "Segment: first jobber. Make 2 pieces."},
    ])
    q = events[-1]
    assert q.type == "needs_input" and q.data["missing"] == ["product"], "product is not carried over from the finished request"


def test_streaming_reply_is_a_final_event(client):
    with client.stream("POST", "/v1/chat/completions", json={
        "model": "content-creator", "stream": True, "messages": [{"role": "user", "content": "hello"}],
    }) as r:
        import json as _json
        chunks = [_json.loads(l[6:]) for l in r.read().decode().splitlines() if l.startswith("data: {")]
    final = chunks[-1]
    assert final["choices"][0]["finish_reason"] == "stop" and final["x_event"]["type"] == "reply"


# ---- reply language follows the user ---------------------------------------


def test_detect_language():
    from app.pipeline.i18n import detect_language

    assert detect_language([{"role": "user", "content": "ขอ 2 ภาพ"}]) == "th"
    assert detect_language([{"role": "user", "content": "make 2 posts"}]) == "en"
    # a bare number keeps the previous message's language
    assert detect_language([{"role": "user", "content": "ภาษาไทยได้ไหม"}, {"role": "assistant", "content": "x"},
                            {"role": "user", "content": "3"}]) == "th"


def test_thai_request_gets_thai_messages(ctx):
    events = run(ctx, "ผลิตภัณฑ์: CI. กลุ่มลูกค้า: young families. ขอ 1 ชิ้น")
    text = "".join(e.text for e in events)
    assert "สรุปสิ่งที่เข้าใจครับ" in text and "แผนงาน:" in text and "เสร็จแล้ว:" in text and "[ชิ้นที่ 1]" in text
    assert "Here's what I understood" not in text and "Done:" not in text
    assert "สไตล์: story (ค่าเริ่มต้น" in text


def test_thai_question_is_in_thai(ctx):
    q = run(ctx, "กลุ่มลูกค้า: young families. ขอ 1 ชิ้น")[-1]
    assert q.type == "needs_input"
    assert q.text.startswith("ขอข้อมูลเพิ่มอีกนิด") and "ต้องการทำคอนเทนต์สำหรับผลิตภัณฑ์อะไรครับ" in q.text


def test_thai_did_you_mean_is_in_thai(ctx):
    q = run(ctx, "ผลิตภัณฑ์: CI. กลุ่มลูกค้า: young families or first jobbers. ขอ 1 ชิ้น")[-1]
    assert q.type == "needs_input" and "ใช่ไหมครับ" in q.text and " หรือ " in q.text


def test_thai_done_ends_the_request_scope(ctx):
    first = run(ctx, "ผลิตภัณฑ์: CI. กลุ่มลูกค้า: young families. ขอ 1 ชิ้น")
    events = run(ctx, [
        {"role": "user", "content": "ผลิตภัณฑ์: CI. กลุ่มลูกค้า: young families. ขอ 1 ชิ้น"},
        {"role": "assistant", "content": "".join(e.text for e in first)},
        {"role": "user", "content": "คุณเป็นใคร"},
    ])
    assert [e.type for e in events if e.type] == ["reply"]
