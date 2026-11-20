"""SQLModel tables (spec section 4.4). Review and batch tables come in later phases."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlmodel import Field, SQLModel

from app.models.status import ItemStatus, RequestStatus, can_transition

__all__ = [
    "CheckResult",
    "ContentItem",
    "ContentRequestRow",
    "Conversation",
    "ImageAsset",
    "ItemStatus",
    "RequestStatus",
    "RunLog",
    "can_transition",
    "new_id",
    "utcnow",
]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Conversation(SQLModel, table=True):
    __tablename__ = "conversation"
    id: str = Field(primary_key=True)
    messages_json: str = "[]"
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class ContentRequestRow(SQLModel, table=True):
    __tablename__ = "content_request"
    id: str = Field(primary_key=True)
    conversation_id: str = Field(index=True)
    mode: str  # content-creator | content-creator-brief
    request_json: str  # full ContentRequest (spec 4.2)
    angles_json: str = "[]"
    status: str = Field(default=RequestStatus.running.value)
    kb_version: str = ""
    created_at: datetime = Field(default_factory=utcnow)


class ContentItem(SQLModel, table=True):
    __tablename__ = "content_item"
    id: str = Field(primary_key=True)
    request_id: str = Field(index=True)
    conversation_id: str = Field(index=True)
    piece_no: int
    angle: str
    content_id: str = Field(index=True)  # e.g. V-0418
    brief_json: str | None = None
    status: str = Field(default=ItemStatus.queued.value)
    version: int = 1
    tags_json: str = "{}"
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class ImageAsset(SQLModel, table=True):
    __tablename__ = "image_asset"
    id: str = Field(primary_key=True)
    item_id: str = Field(index=True)
    item_version: int = 1
    path: str  # file name inside DATA_DIR/images
    prompt: str
    negative_prompt: str | None = None
    provider: str
    model: str
    seed: int | None = None
    variant_no: int
    attempt: int = 1
    width: int
    height: int
    check_json: str = "{}"
    passed: bool = False
    created_at: datetime = Field(default_factory=utcnow)


class CheckResult(SQLModel, table=True):
    __tablename__ = "check_result"
    id: str = Field(primary_key=True)
    item_id: str = Field(index=True)
    item_version: int = 1
    stage: str  # copy | image
    attempt: int = 1
    rule_hits_json: str = "[]"
    judge_score: float | None = None
    judge_notes: str | None = None
    passed: bool
    at: datetime = Field(default_factory=utcnow)


class RunLog(SQLModel, table=True):
    __tablename__ = "run_log"
    id: str = Field(primary_key=True)
    request_id: str | None = Field(default=None, index=True)
    item_id: str | None = Field(default=None, index=True)  # None for request-level steps
    step: str
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    kb_version: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    error: str | None = None
    at: datetime = Field(default_factory=utcnow)
