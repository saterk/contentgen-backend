from __future__ import annotations

import json

import pytest
from openai import AuthenticationError, NotFoundError, OpenAI

from tests.conftest import API_KEY, FULL_REQUEST


def chat(n: int = 2, **extra) -> dict:
    return {
        "model": "content-creator",
        "messages": [{"role": "user", "content": FULL_REQUEST.format(n=n)}],
        "metadata": {"conversation_id": "conv_api"},
        **extra,
    }


def sse_chunks(text: str) -> list[dict | str]:
    out: list[dict | str] = []
    for line in text.splitlines():
        if line.startswith("data: "):
            data = line[6:]
            out.append(data if data == "[DONE]" else json.loads(data))
    return out


# ---- basics -----------------------------------------------------------------


def test_health_needs_no_token(client):
    r = client.get("/api/health", headers={"Authorization": ""})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["kb_version"] == "test.1"
    assert body["providers"]["llm"] == "mock:mock-llm-1"


def test_missing_or_wrong_token_is_401_in_openai_shape(client):
    for headers in ({"Authorization": ""}, {"Authorization": "Bearer nope"}):
        r = client.get("/v1/models", headers=headers)
        assert r.status_code == 401
        assert r.json()["error"]["code"] == "invalid_api_key"


def test_models(client):
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert ids == ["content-creator", "content-creator-brief"]


def test_unknown_model_error_shape(client):
    r = client.post("/v1/chat/completions", json={**chat(), "model": "x"})
    assert r.status_code == 404
    assert r.json() == {"error": {"message": "Unknown model 'x'", "type": "invalid_request_error", "code": "model_not_found"}}


def test_invalid_body_error_shape(client):
    r = client.post("/v1/chat/completions", json={"model": "content-creator", "messages": []})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"


# ---- chat completions -------------------------------------------------------


def test_non_streaming(client):
    r = client.post("/v1/chat/completions", json=chat(n=2, temperature=5, max_tokens=999999, unknown_field=1))
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert "Done: 2 piece(s)" in body["choices"][0]["message"]["content"]
    types = [e["type"] for e in body["x_events"]]
    assert types[0] == "requirement" and types[-1] == "done"
    assert types.count("item_ready") == 2
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"] > 0


def test_streaming_chunks(client):
    with client.stream("POST", "/v1/chat/completions", json=chat(n=3, stream=True, stream_options={"include_usage": True})) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        chunks = sse_chunks(r.read().decode("utf-8"))
    assert chunks[-1] == "[DONE]"
    objs = [c for c in chunks if isinstance(c, dict)]
    assert all(c["object"] == "chat.completion.chunk" for c in objs)
    assert len({c["id"] for c in objs}) == 1
    assert objs[0]["choices"][0]["delta"]["role"] == "assistant"

    events = [c["x_event"] for c in objs if "x_event" in c]
    ready = [e for e in events if e["type"] == "item_ready"]
    assert sorted(e["piece"] for e in ready) == [1, 2, 3]
    assert all(e["item_id"].startswith("itm_") for e in ready)
    assert {e["type"] for e in events} >= {"requirement", "plan", "step", "brief_ready", "images_ready", "done"}

    usage_chunk = objs[-1]
    assert usage_chunk["choices"] == [] and usage_chunk["usage"]["total_tokens"] > 0
    final = objs[-2]
    assert final["choices"][0]["finish_reason"] == "stop" and final["x_event"]["type"] == "done"
    finish = [c["choices"][0]["finish_reason"] for c in objs if c["choices"]]
    assert finish.count("stop") == 1


def test_streaming_needs_input_ends_the_stream(client):
    body = {**chat(), "stream": True, "messages": [{"role": "user", "content": "Product: CI. Segment: young families"}]}
    with client.stream("POST", "/v1/chat/completions", json=body) as r:
        objs = [c for c in sse_chunks(r.read().decode("utf-8")) if isinstance(c, dict)]
    final = objs[-1]
    assert final["choices"][0]["finish_reason"] == "stop"
    assert final["x_event"]["type"] == "needs_input"
    assert set(final["x_event"]["missing"]) == {"count"}, "style, tone and format get defaults"
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in objs)
    assert "How many pieces" in text and "Which style" not in text


def test_items_and_images_after_chat(client):
    body = client.post("/v1/chat/completions", json=chat(n=2)).json()
    item_ids = [e["item_id"] for e in body["x_events"] if e["type"] == "item_ready"]

    item = client.get(f"/api/items/{item_ids[0]}").json()
    assert item["status"] == "pending_review"
    assert item["brief"]["benefit_fact_ids"] and item["disclaimer"]["id"].startswith("DISC-CI")
    assert item["request"]["product_id"] == "critical_illness"
    assert item["images"] and item["chosen_image_id"] == item["images"][0]["id"]
    assert {c["stage"] for c in item["checks"]} == {"copy", "image"}
    assert {r["step"] for r in item["run_logs"]} >= {"design_brief", "copy_check", "generate_images", "image_check"}

    img = client.get(item["images"][0]["url"])
    assert img.status_code == 200 and img.content[:8] == b"\x89PNG\r\n\x1a\n"

    conv = client.get("/api/conversations/conv_api/items").json()
    assert sorted(i["id"] for i in conv["items"]) == sorted(item_ids)

    assert client.get("/api/items/itm_missing").status_code == 404


def test_conversation_is_stored_with_the_reply(client):
    from sqlmodel import Session

    from app.models import Conversation

    client.post("/v1/chat/completions", json=chat(n=1))
    with Session(client.app.state.ctx.engine) as s:
        msgs = json.loads(s.get(Conversation, "conv_api").messages_json)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert "Done: 1 piece(s)" in msgs[-1]["content"]


# ---- official OpenAI SDK ----------------------------------------------------


def sdk(client, key: str = API_KEY) -> OpenAI:
    return OpenAI(base_url="http://testserver/v1", api_key=key, http_client=client, max_retries=0)


def test_openai_sdk_streaming_make_3_pieces(client):
    stream = sdk(client).chat.completions.create(
        model="content-creator",
        stream=True,
        messages=[{"role": "user", "content": FULL_REQUEST.format(n=3)}],
        extra_body={"metadata": {"conversation_id": "conv_sdk"}},
    )
    text, events = "", []
    for chunk in stream:
        if chunk.choices:
            text += chunk.choices[0].delta.content or ""
        x = (chunk.model_extra or {}).get("x_event")
        if x:
            events.append(x)
    assert [e["piece"] for e in events if e["type"] == "item_ready"].__len__() == 3
    assert events[-1]["type"] == "done" and events[-1]["conversation_id"] == "conv_sdk"
    assert "Here's what I understood" in text


def test_openai_sdk_non_streaming(client):
    completion = sdk(client).chat.completions.create(
        model="content-creator-brief",
        messages=[{"role": "user", "content": FULL_REQUEST.format(n=2)}],
    )
    assert completion.choices[0].message.content.startswith("Here's what I understood")
    x_events = completion.model_extra["x_events"]
    assert [e["status"] for e in x_events if e["type"] == "item_ready"] == ["pending_review"] * 2
    assert completion.usage and completion.usage.total_tokens > 0


def test_openai_sdk_raises_on_errors(client):
    with pytest.raises(NotFoundError):
        sdk(client).chat.completions.create(model="gpt-x", messages=[{"role": "user", "content": "hi"}])
    with pytest.raises(AuthenticationError):
        sdk(client, key="wrong").models.list()
