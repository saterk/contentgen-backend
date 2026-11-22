"""The agent's own messages in the user's language (English or Thai).

The reply language follows the user's latest message: Thai if it contains Thai characters,
English if it has Latin letters, otherwise the previous message's language. It is separate
from the ad-copy language (ContentRequest.language, Thai by default).
"""

from __future__ import annotations

import re
from typing import Any

Lang = str  # "en" | "th"

_THAI = re.compile(r"[฀-๿]")
_LATIN = re.compile(r"[A-Za-z]")


def detect_language(messages: list[dict[str, Any]], default: Lang = "en") -> Lang:
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        content = m.get("content")
        text = content if isinstance(content, str) else " ".join(
            p.get("text", "") for p in (content or []) if isinstance(p, dict)
        )
        if _THAI.search(text):
            return "th"
        if _LATIN.search(text):
            return "en"
    return default


FIELD_NAMES = {
    "en": {"product": "product", "segment": "segment", "style": "style", "tone": "tone",
           "format": "format", "visual_style": "visual style", "count": "number of pieces"},
    "th": {"product": "ผลิตภัณฑ์", "segment": "กลุ่มลูกค้า", "style": "สไตล์", "tone": "โทน",
           "format": "รูปแบบ", "visual_style": "แนวภาพ", "count": "จำนวนชิ้น"},
}

MESSAGES: dict[str, dict[Lang, str]] = {
    # ---- clarifying questions ----
    "need_more": {"en": "I need a bit more before I start.", "th": "ขอข้อมูลเพิ่มอีกนิดก่อนเริ่มนะครับ"},
    "q_product": {"en": "Which product is this for?", "th": "ต้องการทำคอนเทนต์สำหรับผลิตภัณฑ์อะไรครับ?"},
    "q_segment": {"en": "Which customer segment?", "th": "สำหรับกลุ่มลูกค้ากลุ่มไหนครับ?"},
    "q_count": {"en": "How many pieces would you like (1–{max})?", "th": "ต้องการกี่ชิ้นครับ (1–{max})?"},
    "q_unknown_value": {"en": 'I don\'t recognise the {field} "{raw}". Pick one of: {allowed}.',
                        "th": 'ไม่รู้จัก{field} "{raw}" ครับ เลือกได้จาก: {allowed}'},
    "hint": {"en": "You can answer like: Product: … Segment: … Style: … Tone: … Make 3 pieces.",
             "th": "ตอบแบบนี้ได้เลยครับ เช่น ผลิตภัณฑ์: … กลุ่มลูกค้า: … สไตล์: … โทน: … ขอ 3 ชิ้น"},
    "did_you_mean": {"en": 'For the {field} "{text}", did you mean {options}?',
                     "th": '{field} "{text}" หมายถึง {options} ใช่ไหมครับ?'},
    "or": {"en": " or ", "th": " หรือ "},
    "unknown_entity": {"en": 'I couldn\'t match the {field} "{text}" to anything I know. Available: {known}. Which one?',
                       "th": 'ไม่พบ{field} "{text}" ในระบบครับ ที่มีคือ: {known} ต้องการแบบไหนครับ?'},
    "none_loaded": {"en": "none loaded", "th": "ยังไม่มีข้อมูล"},
    # ---- notes in the summary ----
    "note_mapped": {"en": '{field} "{raw}" → {value}', "th": '{field} "{raw}" → {value}'},
    "note_visual_moved": {"en": '"{raw}" is a visual style → images in {value} style',
                          "th": '"{raw}" เป็นแนวภาพ → ใช้ภาพแนว {value}'},
    "note_default": {"en": "{field}: {value} (default; tell me if you want another)",
                     "th": "{field}: {value} (ค่าเริ่มต้น บอกได้ถ้าต้องการแบบอื่น)"},
    "note_only_format": {"en": "format: {value} (the only format available)", "th": "รูปแบบ: {value} (มีรูปแบบเดียว)"},
    "note_visual_unavailable": {"en": 'visual style "{raw}" isn\'t available ({allowed}); using {value}',
                                "th": 'ไม่มีแนวภาพ "{raw}" (มี {allowed}) ใช้ {value} แทน'},
    "note_clamped": {"en": "You asked for {n} pieces; the prototype makes up to {max}, so I'll make {max}.",
                     "th": "ขอมา {n} ชิ้น แต่ระบบทำได้สูงสุด {max} ชิ้น จะทำให้ {max} ชิ้นครับ"},
    "note_channel": {"en": 'channel "{raw}" is not supported yet; using {value}',
                     "th": 'ยังไม่รองรับช่องทาง "{raw}" ใช้ {value} แทน'},
    "note_language": {"en": 'language "{raw}" is not supported; using {value}',
                      "th": 'ไม่รองรับภาษา "{raw}" ใช้ {value} แทน'},
    # ---- summary of the request ----
    "understood": {"en": "Here's what I understood:", "th": "สรุปสิ่งที่เข้าใจครับ:"},
    "s_product": {"en": "Product", "th": "ผลิตภัณฑ์"},
    "s_segment": {"en": "Segment", "th": "กลุ่มลูกค้า"},
    "s_profile": {"en": "Style / tone / format", "th": "สไตล์ / โทน / รูปแบบ"},
    "s_visual": {"en": "Visual style", "th": "แนวภาพ"},
    "s_pieces": {"en": "Pieces", "th": "จำนวน"},
    "s_channel": {"en": "Channel", "th": "ช่องทาง"},
    "s_language": {"en": "Copy language", "th": "ภาษาของข้อความ"},
    "s_objective": {"en": "Objective", "th": "วัตถุประสงค์"},
    "plan": {"en": "Plan:", "th": "แผนงาน:"},
    # ---- progress per piece ----
    "piece": {"en": "Piece {n}", "th": "ชิ้นที่ {n}"},
    "step_drafting": {"en": "Drafting copy and visual brief…", "th": "กำลังเขียนข้อความและบรีฟภาพ…"},
    "step_redrafting": {"en": "Redrafting copy with the check feedback…", "th": "กำลังแก้ข้อความตามผลการตรวจ…"},
    "step_copy_check": {"en": "Checking copy…", "th": "กำลังตรวจข้อความ…"},
    "step_images": {"en": "Generating prototype images…", "th": "กำลังสร้างภาพตัวอย่าง…"},
    "step_reimages": {"en": "Regenerating images with an adjusted prompt…", "th": "กำลังสร้างภาพใหม่ด้วยคำสั่งที่ปรับแล้ว…"},
    "step_image_check": {"en": "Checking images…", "th": "กำลังตรวจภาพ…"},
    "copy_failed": {"en": "Copy check failed: {findings}", "th": "ข้อความไม่ผ่านการตรวจ: {findings}"},
    "image_failed": {"en": "Image check failed: {findings}", "th": "ภาพไม่ผ่านการตรวจ: {findings}"},
    "ready": {"en": "Ready for review.", "th": "พร้อมให้รีวิวครับ"},
    "needs_attention": {"en": "Needs attention: {reason}.", "th": "ต้องตรวจสอบ: {reason}"},
    "reason_copy_twice": {"en": "copy failed the checks twice", "th": "ข้อความไม่ผ่านการตรวจ 2 ครั้ง"},
    "reason_images": {"en": "no image passed the checks after a retry", "th": "ภาพไม่ผ่านการตรวจแม้ลองใหม่แล้ว"},
    "reason_timeout": {"en": "timed out after {secs}s", "th": "ใช้เวลาเกิน {secs} วินาที"},
    "reason_request_timeout": {"en": "request timed out", "th": "คำขอใช้เวลานานเกินกำหนด"},
    "reason_error": {"en": "error: {detail}", "th": "เกิดข้อผิดพลาด: {detail}"},
    # ---- whole request ----
    "too_similar": {"en": "Pieces {a} and {b} are too similar (similarity {score}). Regenerating piece {b} with a new angle: {angle}",
                    "th": "ชิ้นที่ {a} และ {b} คล้ายกันเกินไป (ความคล้าย {score}) กำลังทำชิ้นที่ {b} ใหม่ด้วยมุมใหม่: {angle}"},
    "request_timeout": {"en": "The request reached its time limit; stopping the remaining pieces.",
                        "th": "หมดเวลาแล้ว หยุดทำชิ้นที่เหลือครับ"},
    "done": {"en": "Done: {n} piece(s) in {secs}s. {ok} passed the checks and are ready for review.",
             "th": "เสร็จแล้ว: {n} ชิ้น ใช้เวลา {secs} วินาที ผ่านการตรวจ {ok} ชิ้น พร้อมให้รีวิวครับ"},
    "done_attention": {"en": "Needs attention: {items}", "th": "ต้องตรวจสอบ: {items}"},
    "error": {"en": "Sorry, something went wrong: {detail}", "th": "ขออภัย เกิดข้อผิดพลาด: {detail}"},
    "default_reply": {
        "en": "I'm the content creator assistant for AIA Thailand's Marketing Department. Tell me the product, the customer segment and how many pieces "
              "you need, and I'll write the copy and make prototype images for each piece.",
        "th": "ผมคือผู้ช่วยสร้างคอนเทนต์ของฝ่ายการตลาด เอไอเอ ประเทศไทยครับ บอกผลิตภัณฑ์ กลุ่มลูกค้า และจำนวนชิ้นที่ต้องการ "
              "แล้วผมจะเขียนข้อความและสร้างภาพตัวอย่างให้ครับ",
    },
}

# Markers that end a request in the conversation (see the orchestrator prompt's "Scope" rule).
DONE_MARKERS = ("Done:", "เสร็จแล้ว:")


def t(lang: Lang, key: str, **kw: Any) -> str:
    entry = MESSAGES[key]
    return entry.get(lang, entry["en"]).format(**kw)


def field_name(lang: Lang, field: str) -> str:
    return FIELD_NAMES.get(lang, FIELD_NAMES["en"]).get(field, field)
