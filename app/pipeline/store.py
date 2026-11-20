"""Small DB helpers used by the pipeline."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import Engine, func
from sqlmodel import Session, select

from app.db import session_scope
from app.models import (
    CheckResult,
    ContentItem,
    ContentRequestRow,
    ImageAsset,
    ItemStatus,
    can_transition,
    new_id,
    utcnow,
)


class InvalidTransition(RuntimeError):
    pass


def next_content_ids(session: Session, n: int) -> list[str]:
    start = session.exec(select(func.count()).select_from(ContentItem)).one() + 1
    return [f"V-{i:04d}" for i in range(start, start + n)]


def create_request(engine: Engine, row: ContentRequestRow) -> None:
    with session_scope(engine) as s:
        s.add(row)


def update_request(engine: Engine, request_id: str, **fields: Any) -> None:
    with session_scope(engine) as s:
        row = s.get(ContentRequestRow, request_id)
        if row:
            for k, v in fields.items():
                setattr(row, k, v)
            s.add(row)


def create_items(
    engine: Engine, request_id: str, conversation_id: str, angles: list[str], tags: dict[str, Any]
) -> list[ContentItem]:
    with session_scope(engine) as s:
        ids = next_content_ids(s, len(angles))
        items = [
            ContentItem(
                id=new_id("itm"),
                request_id=request_id,
                conversation_id=conversation_id,
                piece_no=i,
                angle=angle,
                content_id=cid,
                tags_json=json.dumps({**tags, "content_id": cid}, ensure_ascii=False),
            )
            for i, (angle, cid) in enumerate(zip(angles, ids), start=1)
        ]
        s.add_all(items)
    return items


def get_item(engine: Engine, item_id: str) -> ContentItem | None:
    with Session(engine) as s:
        return s.get(ContentItem, item_id)


def set_status(engine: Engine, item_id: str, new: ItemStatus, **fields: Any) -> ContentItem:
    with session_scope(engine) as s:
        item = s.get(ContentItem, item_id)
        if item is None:
            raise KeyError(item_id)
        if item.status != new.value:
            if not can_transition(item.status, new):
                raise InvalidTransition(f"{item_id}: {item.status} -> {new.value} is not allowed")
            item.status = new.value
        for k, v in fields.items():
            setattr(item, k, v)
        item.updated_at = utcnow()
        s.add(item)
        return item


def save_check(
    engine: Engine,
    *,
    item_id: str,
    item_version: int,
    stage: str,
    attempt: int,
    rule_hits: list[dict],
    judge_score: float | None,
    judge_notes: str | None,
    passed: bool,
) -> None:
    with session_scope(engine) as s:
        s.add(
            CheckResult(
                id=new_id("chk"),
                item_id=item_id,
                item_version=item_version,
                stage=stage,
                attempt=attempt,
                rule_hits_json=json.dumps(rule_hits, ensure_ascii=False),
                judge_score=judge_score,
                judge_notes=judge_notes,
                passed=passed,
            )
        )


def save_image(engine: Engine, asset: ImageAsset) -> None:
    with session_scope(engine) as s:
        s.add(asset)
