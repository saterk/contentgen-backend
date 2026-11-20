"""OpenAI-compatible adapters, tested against a fake HTTP transport (no network)."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.kb.embeddings import OpenAICompatEmbedding
from app.providers.llm_base import StructuredOutputError, structured_call
from app.providers.llm_openai_compat import OpenAICompatLLM
from app.schemas.llm import RequirementDraft


def chat_response(content: str, cost: float | None = 0.00012) -> dict:
    usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
    if cost is not None:
        usage["cost"] = cost
    return {
        "id": "gen-1", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": usage,
    }


class FakeAPI:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.requests: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        return httpx.Response(200, json=chat_response(self.replies.pop(0)))


def make_llm(api: FakeAPI, provider: str = "openrouter", reasoning: str = "off") -> OpenAICompatLLM:
    return OpenAICompatLLM(
        provider_name=provider, model="qwen/test", api_key="k", base_url="https://example.test/v1",
        reasoning=reasoning, http_client=httpx.AsyncClient(transport=httpx.MockTransport(api.handler)),
    )


def test_structured_call_request_shape_and_usage():
    api = FakeAPI(['{"product_text": "CI", "count": 3}'])
    res = asyncio.run(structured_call(
        make_llm(api), RequirementDraft, task="extract_requirement", system="SYS", user="hello", temperature=0,
    ))
    assert res.value.product_text == "CI" and res.value.count == 3
    assert (res.input_tokens, res.output_tokens, res.cost_usd) == (100, 20, 0.00012)
    req = api.requests[0]
    assert req["model"] == "qwen/test" and req["temperature"] == 0
    assert req["response_format"] == {"type": "json_object"}
    assert req["messages"][0]["role"] == "system" and req["messages"][0]["content"].startswith("SYS")
    assert '"product_text"' in req["messages"][0]["content"], "schema is included in the system prompt"
    assert req["messages"][1] == {"role": "user", "content": "hello"}
    assert req["reasoning"] == {"enabled": False} and req["usage"] == {"include": True}


@pytest.mark.parametrize(("reasoning", "expected"), [("low", {"effort": "low"}), ("default", None)])
def test_reasoning_setting(reasoning, expected):
    api = FakeAPI(["{}"])
    asyncio.run(structured_call(make_llm(api, reasoning=reasoning), RequirementDraft, task="t", system="s", user="u"))
    assert api.requests[0].get("reasoning") == expected


def test_openai_provider_sends_no_openrouter_extras():
    api = FakeAPI(["{}"])
    asyncio.run(structured_call(make_llm(api, provider="openai"), RequirementDraft, task="t", system="s", user="u"))
    assert "reasoning" not in api.requests[0] and "usage" not in api.requests[0]


def test_invalid_json_is_retried_with_the_error():
    api = FakeAPI(["not json", '```json\n{"count": "many"}\n```', '{"count": 2}'])
    res = asyncio.run(structured_call(make_llm(api), RequirementDraft, task="t", system="s", user="u"))
    assert res.value.count == 2 and res.attempts == 3
    retry_msgs = api.requests[2]["messages"]
    assert retry_msgs[-1]["role"] == "user" and "did not match the required JSON schema" in retry_msgs[-1]["content"]
    assert res.cost_usd == pytest.approx(0.00036), "cost is summed across attempts"


def test_gives_up_after_retries():
    api = FakeAPI(["nope"] * 3)
    with pytest.raises(StructuredOutputError):
        asyncio.run(structured_call(make_llm(api), RequirementDraft, task="t", system="s", user="u"))


def test_images_are_sent_as_data_urls(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    api = FakeAPI(["{}"])
    asyncio.run(make_llm(api).complete(task="image_judge", system="s", messages=[{"role": "user", "content": "check"}], images=[img]))
    parts = api.requests[0]["messages"][1]["content"]
    assert parts[0] == {"type": "text", "text": "check"}
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_missing_key_fails_fast():
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        OpenAICompatLLM(provider_name="openrouter", model="m", api_key="", base_url=None)


def test_embeddings_batch_and_cache():
    calls: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        calls.append(inputs)
        data = [{"object": "embedding", "index": i, "embedding": [float(len(t)), 1.0]} for i, t in enumerate(inputs)]
        return httpx.Response(200, json={"object": "list", "data": data, "model": "e", "usage": {"prompt_tokens": 1, "total_tokens": 1}})

    emb = OpenAICompatEmbedding(provider_name="openrouter", model="e", api_key="k", base_url="https://example.test/v1",
                                http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    emb.BATCH = 2
    out = emb.embed(["a", "bb", "ccc", "a"])
    assert [v[0] for v in out] == [1.0, 2.0, 3.0, 1.0]
    assert calls == [["a", "bb"], ["ccc"]], "deduplicated and batched"
    emb.embed(["bb"])
    assert len(calls) == 2, "cached"


# ---- OpenRouter images ------------------------------------------------------


def _jpeg(size=(928, 1152)) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, (200, 150, 120)).save(buf, format="JPEG")
    return buf.getvalue()


def test_nearest_aspect_ratio():
    from app.providers.image_openrouter import nearest_aspect_ratio

    assert nearest_aspect_ratio((1080, 1350)) == "4:5"
    assert nearest_aspect_ratio((1080, 1080)) == "1:1"
    assert nearest_aspect_ratio((1920, 1080)) == "16:9"


def test_openrouter_image_request_and_parsing():
    import base64

    from app.providers.image_openrouter import OpenRouterImageProvider

    seen: list[dict] = []
    jpg = _jpeg()

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        body = chat_response("", cost=0.0336)
        body["choices"][0]["message"]["images"] = [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()}}
        ]
        return httpx.Response(200, json=body)

    p = OpenRouterImageProvider(model="google/gemini-3.1-flash-lite-image", api_key="k",
                                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    imgs = asyncio.run(p.generate("A couple at home", "text, logo", (1080, 1350), n=2, seed=7))
    assert [(i.width, i.height, i.mime, i.seed, i.cost_usd) for i in imgs] == [
        (928, 1152, "image/jpeg", 7, 0.0336), (928, 1152, "image/jpeg", 8, 0.0336)]
    req = seen[0]
    assert req["modalities"] == ["image", "text"] and req["image_config"] == {"aspect_ratio": "4:5"}
    assert "Avoid: text, logo." in req["messages"][0]["content"]


def test_openrouter_image_refusal_raises():
    from app.providers.image_openrouter import ImageGenerationError, OpenRouterImageProvider

    def handler(request):
        return httpx.Response(200, json=chat_response("I can't make that image."))

    p = OpenRouterImageProvider(model="m", api_key="k", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(ImageGenerationError, match="can't make that"):
        asyncio.run(p.generate("x", None, (1080, 1350), n=1, seed=None))


def test_fit_to_size_crops_and_saves_jpeg():
    from app.pipeline.image_agent import fit_to_size

    data, w, h, ext = fit_to_size(_jpeg(), (1080, 1350))
    assert (w, h, ext) == (1080, 1350, "jpg") and data[:2] == b"\xff\xd8"
