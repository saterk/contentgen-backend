"""OpenAI Chat Completions–compatible endpoint (spec 7.1).

POST /v1/chat/completions runs the content-creator pipeline and answers in the
Chat Completions format. Progress text goes in `delta.content`; structured events
go in an extra top-level `x_event` field on each chunk (and `x_events` for
non-streaming responses).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any, Literal

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlmodel import Session

from app.api.deps import get_ctx, require_app_token
from app.api.errors import APIError
from app.models import Conversation, new_id, utcnow
from app.pipeline import events as ev
from app.pipeline.context import AppContext
from app.pipeline.i18n import detect_language
from app.pipeline.i18n import t as tr
from app.pipeline.piece import MODE_BRIEF, MODE_FULL
from app.pipeline.runner import run_chat

log = logging.getLogger("content_creator")
router = APIRouter(prefix="/v1", dependencies=[Depends(require_app_token)])

MODELS = {
    MODE_FULL: "Full flow: clarify if needed, brief, copy check, images, image check; saved as pending_review",
    MODE_BRIEF: "Brief and copy check only, no images",
}


# ---- request models ---------------------------------------------------------


class ContentPart(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str
    text: str | None = None


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[ContentPart] | None = None

    def text(self) -> str:
        if isinstance(self.content, list):
            return "\n".join(p.text or "" for p in self.content if p.type == "text")
        return self.content or ""


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="allow")
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")  # unknown fields are ignored
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    metadata: dict[str, Any] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None


# ---- helpers ----------------------------------------------------------------


def _conversation_id(body: ChatCompletionRequest) -> str:
    cid = (body.metadata or {}).get("conversation_id")
    return str(cid) if cid else new_id("conv")


def _save_conversation(ctx: AppContext, conversation_id: str, messages: list[dict[str, str]]) -> None:
    with Session(ctx.engine) as s:
        conv = s.get(Conversation, conversation_id) or Conversation(id=conversation_id)
        conv.messages_json = json.dumps(messages, ensure_ascii=False)
        conv.updated_at = utcnow()
        s.add(conv)
        s.commit()


class _ChunkFactory:
    def __init__(self, model: str) -> None:
        self.id = f"chatcmpl-{new_id('c').split('_', 1)[1]}"
        self.created = int(time.time())
        self.model = model

    def chunk(self, delta: dict[str, Any], finish_reason: str | None = None, x_event: dict | None = None) -> dict:
        c: dict[str, Any] = {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "system_fingerprint": None,
            "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}],
        }
        if x_event is not None:
            c["x_event"] = x_event
        return c

    def usage_chunk(self, usage: dict) -> dict:
        return {
            "id": self.id, "object": "chat.completion.chunk", "created": self.created,
            "model": self.model, "choices": [], "usage": usage,
        }


def _sse(obj: dict | str) -> str:
    data = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return f"data: {data}\n\n"


def _usage(done_event: ev.Event | None) -> dict[str, int]:
    u = (done_event.data.get("usage") if done_event else None) or {}
    p, c = int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


async def _events(ctx: AppContext, body: ChatCompletionRequest, conversation_id: str) -> AsyncIterator[ev.Event]:
    """Pipeline events, with unexpected errors turned into a final `error` event."""
    messages = [{"role": m.role, "content": m.text()} for m in body.messages]
    try:
        async for e in run_chat(ctx, messages, conversation_id, mode=body.model):
            yield e
    except Exception as exc:  # report in-band so the chat UI shows it
        log.exception("chat request failed")
        yield ev.Event(
            type=ev.ERROR,
            text="\n" + tr(detect_language(messages), "error", detail=f"{type(exc).__name__}: {exc}") + "\n",
            data={"conversation_id": conversation_id, "message": str(exc)},
        )


# ---- routes -----------------------------------------------------------------


@router.get("/models")
async def list_models() -> dict:
    return {
        "object": "list",
        "data": [
            {"id": mid, "object": "model", "created": 0, "owned_by": "content-creator", "description": desc}
            for mid, desc in MODELS.items()
        ],
    }


@router.post("/chat/completions")
async def chat_completions(body: ChatCompletionRequest, ctx: AppContext = Depends(get_ctx)):
    if body.model not in MODELS:
        raise APIError(404, f"Unknown model '{body.model}'", "invalid_request_error", "model_not_found")
    if not any(m.role == "user" and m.text().strip() for m in body.messages):
        raise APIError(400, "messages must include a user message", "invalid_request_error", "invalid_request")
    # temperature / max_tokens are accepted and capped; the pipeline uses its own settings in the prototype.
    if body.temperature is not None:
        body.temperature = max(0.0, min(body.temperature, ctx.settings.max_temperature))
    if body.max_tokens is not None:
        body.max_tokens = min(body.max_tokens, ctx.settings.max_tokens_cap)

    conversation_id = _conversation_id(body)
    history = [{"role": m.role, "content": m.text()} for m in body.messages]
    _save_conversation(ctx, conversation_id, history)
    factory = _ChunkFactory(body.model)
    include_usage = bool(body.stream_options and body.stream_options.include_usage)

    if body.stream:
        async def stream() -> AsyncIterator[str]:
            transcript: list[str] = []
            done: ev.Event | None = None
            yield _sse(factory.chunk({"role": "assistant", "content": ""}))
            async for e in _events(ctx, body, conversation_id):
                transcript.append(e.text)
                if e.final:
                    done = e
                    if e.text:
                        yield _sse(factory.chunk({"content": e.text}))
                    yield _sse(factory.chunk({}, finish_reason="stop", x_event=e.x_event()))
                else:
                    yield _sse(factory.chunk({"content": e.text} if e.text else {}, x_event=e.x_event()))
            if done is None:  # generator ended without a final event
                yield _sse(factory.chunk({}, finish_reason="stop"))
            if include_usage:
                yield _sse(factory.usage_chunk(_usage(done)))
            yield _sse("[DONE]")
            _save_conversation(ctx, conversation_id, history + [{"role": "assistant", "content": "".join(transcript)}])

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    content: list[str] = []
    x_events: list[dict] = []
    done: ev.Event | None = None
    async for e in _events(ctx, body, conversation_id):
        content.append(e.text)
        if (x := e.x_event()) is not None:
            x_events.append(x)
        if e.final:
            done = e
    text = "".join(content)
    _save_conversation(ctx, conversation_id, history + [{"role": "assistant", "content": text}])
    return {
        "id": factory.id,
        "object": "chat.completion",
        "created": factory.created,
        "model": body.model,
        "system_fingerprint": None,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text, "refusal": None},
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": _usage(done),
        "x_events": x_events,
    }
