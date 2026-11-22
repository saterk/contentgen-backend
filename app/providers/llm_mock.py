"""Deterministic mock LLM for offline development and tests.

It reads the structured `context` the pipeline passes alongside each prompt and
returns JSON in the shape of the task's schema. It is rule-based: it reads
labelled chat requirements such as "Product: CI. Segment: ... Make 5 pieces."
Free-form understanding arrives with the real LLM adapter (build step 4).
"""

from __future__ import annotations

import json
import re
import zlib
from typing import Any

from app.pipeline.i18n import DONE_MARKERS
from app.providers.llm_base import LLMProvider, LLMResponse, estimate_tokens

# --- requirement extraction --------------------------------------------------

_LABELS: dict[str, str] = {
    "product_text": r"product|ผลิตภัณฑ์|สินค้า",
    "segment_text": r"segment|audience|กลุ่มลูกค้า|กลุ่มเป้าหมาย",
    "style": r"style|สไตล์",
    "tone": r"tone|โทน",
    "format": r"format|รูปแบบ",
    "visual_style": r"visual style|image style|art style|แนวภาพ",
    "channel": r"channel|ช่องทาง",
    "objective": r"objective|goal|เป้าหมาย",
    "journey_stage": r"journey stage|stage",
    "language": r"language|ภาษา",
}
_COUNT_RES = [
    re.compile(r"(?:count|pieces|จำนวน)\s*[:：]\s*(\d+)", re.I),
    re.compile(r"(\d+)\s*(?:content\s+)?(?:pieces?|items?|posts?|ads?|variations?|versions?|ชิ้น|แบบ)", re.I),
    re.compile(r"(?:make|create|generate|give me|ขอ|ทำ)\s*(\d+)", re.I),
]
# keyword in the assistant's question -> field an unlabelled reply answers
_ASKED_FIELD = [
    ("how many", "count"),
    ("segment", "segment_text"),
    ("product", "product_text"),
    ("style", "style"),
    ("tone", "tone"),
    ("format", "format"),
]


def _text(content: Any) -> str:
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def _parse_labelled(text: str) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for fld, label in _LABELS.items():
        m = re.search(rf"(?:^|[\s.;,])(?:{label})\s*[:：]\s*([^\n;]+?)(?=\.\s|\.$|;|\n|$)", text, re.I)
        if m:
            found[fld] = m.group(1).strip().rstrip(".")
    for rx in _COUNT_RES:
        m = rx.search(text)
        if m:
            found["count"] = int(m.group(1))
            break
    return found


def _unlabelled_answer(text: str, question: str, req: dict[str, Any], has_count: bool) -> dict[str, Any]:
    """Treat an unlabelled user reply as the answer to the assistant's last question."""
    reply = text.strip().rstrip(".")
    if not reply:
        return {}
    if re.fullmatch(r"\d+", reply):
        return {"count": int(reply)} if "how many" in question else {}
    if has_count:  # e.g. "5 pieces" — nothing else to read
        return {}
    asked = [fld for kw, fld in _ASKED_FIELD if fld != "count" and kw in question]
    if not asked:
        return {}
    missing = [f for f in asked if not req.get(f)]
    # "Did you mean A or B?" re-asks a field that is already set, so fall back to the first asked field
    return {(missing or asked)[0]: reply}


MOCK_REPLY = (
    "I'm the content creator assistant for AIA Thailand's Marketing Department (mock). Tell me the product, the customer segment and how many "
    "pieces you need, e.g. Product: CI. Segment: young families. Make 3 pieces."
)


def extract_requirement(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify the latest message and merge requirement fields within the current request.

    A request ends with the assistant's "Done:" summary; later messages start fresh.
    A latest message with nothing to read (and not answering a question) is chat.
    """
    last_done = max((i for i, m in enumerate(messages) if m.get("role") == "assistant"
                     and any(k in _text(m.get("content")) for k in DONE_MARKERS)), default=-1)
    scope = messages[last_done + 1:]
    req: dict[str, Any] = {}
    last_assistant = ""
    latest_found: dict[str, Any] = {}
    for msg in scope:
        role, text = msg.get("role"), _text(msg.get("content"))
        if role == "assistant":
            last_assistant = text.lower()
            continue
        if role != "user":
            continue
        found = _parse_labelled(text)
        if last_assistant and not any(k != "count" for k in found):
            found.update(_unlabelled_answer(text, last_assistant, req, has_count="count" in found))
        req.update(found)
        latest_found = found
    if not latest_found:
        return {"intent": "chat", "reply": MOCK_REPLY}
    return {**req, "intent": "answer" if last_assistant else "request"}


# --- angles and briefs -------------------------------------------------------

SCENES: list[dict[str, str]] = [
    {"angle": "A couple on the sofa at home planning next year",
     "th": "วางแผนปีหน้าด้วยกัน อย่างอุ่นใจ", "en": "Plan next year together, with peace of mind",
     "subject": "Two Asian adults in their 30s on a sofa at home, looking at a notebook together"},
    {"angle": "A parent making breakfast before work",
     "th": "ทุกเช้าธรรมดา คือสิ่งที่อยากรักษาไว้", "en": "Keep the everyday mornings you love",
     "subject": "An Asian parent in their late 30s making breakfast in a bright kitchen"},
    {"angle": "A small-business owner opening the shop in the morning",
     "th": "ให้แผนของคุณเดินต่อ แม้ในวันที่ต้องพัก", "en": "Keep your plans moving, even on days you rest",
     "subject": "An Asian shop owner in their 30s opening a small cafe in the morning"},
    {"angle": "A weekend walk in the park with the family",
     "th": "เวลาของครอบครัว ไม่ควรต้องสะดุด", "en": "Family time shouldn't have to pause",
     "subject": "An Asian family of three walking in a green park on a weekend afternoon"},
    {"angle": "Sharing dinner with elderly parents",
     "th": "ดูแลคนที่คุณรัก ได้อย่างมั่นใจ", "en": "Care for the people you love, with confidence",
     "subject": "An Asian adult in their 40s sharing dinner with elderly parents at home"},
    {"angle": "Moving into a first home",
     "th": "บ้านหลังแรก กับแผนที่มั่นคง", "en": "A first home, and a steady plan",
     "subject": "An Asian couple in their 30s carrying boxes into a new apartment"},
    {"angle": "A coffee break at the office talking about plans",
     "th": "คุยเรื่องอนาคต เริ่มได้ตั้งแต่วันนี้", "en": "Talk about the future, starting today",
     "subject": "Two Asian colleagues in their 30s chatting over coffee by an office window"},
    {"angle": "An early morning run by the river",
     "th": "ใส่ใจตัวเอง และพร้อมไว้เสมอ", "en": "Look after yourself, and stay ready",
     "subject": "An Asian adult in their 30s jogging along a riverside path at sunrise"},
    {"angle": "Walking a child to school",
     "th": "ให้แผนของลูกเดินหน้าต่อได้", "en": "Keep your child's plans on track",
     "subject": "An Asian parent in their 30s walking a young child to the school gate"},
    {"angle": "Planning a family trip at the dining table",
     "th": "ทริปครอบครัวครั้งหน้า วางแผนได้เต็มที่", "en": "Plan the next family trip with an easy mind",
     "subject": "An Asian couple in their 30s looking at a map on a laptop at the dining table"},
    {"angle": "Gardening on a quiet Sunday",
     "th": "วันสบาย ๆ ที่อยากให้มีไปอีกนาน", "en": "Easy Sundays you want to keep",
     "subject": "An Asian adult in their 40s tending plants on a sunny balcony garden"},
    {"angle": "A relaxed chat with an advisor at a cafe",
     "th": "คุยกับที่ปรึกษา แบบสบาย ๆ ไม่กดดัน", "en": "A relaxed, no-pressure chat with an advisor",
     "subject": "An Asian adult in their 30s talking with a friendly advisor at a cafe table"},
]
EMPHASES = [
    "family plans stay on track",
    "freedom to use the lump sum",
    "simple to start with an advisor",
    "cover that fits the budget",
]
_MOOD = {
    "warm": "Cozy and secure; nothing clinical",
    "friendly": "Light, relaxed and everyday",
    "reassuring": "Calm and steady; a sense of being looked after",
    "urgent": "Purposeful and focused, never fearful",
}
_BODY = {
    "th": "เตรียมความพร้อมไว้ตั้งแต่วันนี้ เพื่อให้แผนของคุณและคนที่คุณรักเดินหน้าต่อได้ในทุกวัน",
    "en": "Get ready today, so your plans and the people you love can keep moving forward.",
}
_CTA = {"th": "คุยกับที่ปรึกษา", "en": "Talk to an advisor"}


def plan_angles(count: int, avoid: list[str]) -> list[str]:
    used = {a.split(" — ")[0] for a in avoid}
    angles: list[str] = []
    i = 0
    while len(angles) < count:
        scene = SCENES[i % len(SCENES)]["angle"]
        emphasis = EMPHASES[i % len(EMPHASES)]
        if scene not in used or i >= len(SCENES):
            angles.append(f"{scene} — emphasis: {emphasis}")
            used.add(scene)
        i += 1
    return angles


def _scene_for(angle: str) -> dict[str, str]:
    for s in SCENES:
        if angle.startswith(s["angle"]):
            return s
    return SCENES[zlib.crc32(angle.encode()) % len(SCENES)]


def design_brief(ctx: dict[str, Any]) -> dict[str, Any]:
    req = ctx["request"]
    profile = req["profile"]
    lang = req.get("language", "th")
    angle = ctx["angle"]
    scene = _scene_for(angle)
    facts = ctx.get("product_facts", [])
    disclaimers = ctx.get("disclaimers", [])
    idx = SCENES.index(scene)
    picked = [facts[(idx + k) % len(facts)] for k in range(min(2, len(facts)))] if facts else []
    mood = _MOOD.get(profile["tone"], "Calm and everyday; nothing clinical")
    visual = {
        "subject": scene["subject"],
        "mood_story": mood,
        "light_color": "Warm window light, one red accent",
        "composition": "4:5, people in middle third, space top and bottom",
        "must_avoid": ["hospital scenes", "text in image", "logos", "people in distress"],
    }
    return {
        "headline": scene[lang],
        "body": _BODY[lang],
        "benefit_lines": [f["text"] for f in picked],
        "benefit_fact_ids": [f["fact_id"] for f in picked],
        "cta_text": _CTA[lang],
        "disclaimer_id": disclaimers[0]["disclaimer_id"] if disclaimers else "MISSING",
        "visual": visual,
        "image_prompt": f"{scene['subject']}. {mood}. {visual['light_color']}. {visual['composition']}.",
        "negative_prompt": "text, letters, logo, watermark, hospital, medical equipment, distorted hands",
        "language": lang,
        "rationale": f"{profile['style']} style, {profile['tone']} tone for {req.get('segment_id')}: {angle}",
    }


# --- the provider ------------------------------------------------------------


class MockLLM(LLMProvider):
    provider_name = "mock"

    def __init__(self, model: str = "mock-llm-1") -> None:
        self.model = model

    async def complete(
        self,
        *,
        task: str,
        system: str,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any] | None = None,
        images=None,
        context: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        ctx = context or {}
        result = self.respond(task, ctx)
        text = json.dumps(result, ensure_ascii=False)
        prompt_len = len(system) + sum(len(_text(m.get("content"))) for m in messages)
        return LLMResponse(text=text, input_tokens=max(1, prompt_len // 4), output_tokens=estimate_tokens(text))

    def respond(self, task: str, ctx: dict[str, Any]) -> dict[str, Any]:
        if task == "extract_requirement":
            return extract_requirement(ctx.get("messages", []))
        if task == "plan_angles":
            return {"angles": plan_angles(int(ctx.get("count", 1)), list(ctx.get("avoid", [])))}
        if task == "design_brief":
            return design_brief(ctx)
        if task == "copy_judge":
            risky = list(ctx.get("banned_hits", []))
            return {
                "profile_adherence": 4,
                "factual_support": 4,
                "clarity": 4,
                "risky_claims": risky,
                "notes": "mock judge" + (f": flagged {risky}" if risky else ""),
            }
        if task == "image_judge":
            return {
                "matches_subject": True,
                "must_avoid_present": [],
                "has_text_or_logo": False,
                "distortions": [],
                "suitable_for_insurance_ad": True,
                "score": 4,
                "notes": "mock judge: placeholder image accepted",
            }
        raise ValueError(f"MockLLM has no handler for task '{task}'")
