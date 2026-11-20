"""Image generation through OpenRouter's chat completions (modalities: image).

Works with OpenRouter image-output models such as google/gemini-3.1-flash-lite-image.
The model returns images as data URLs in `choices[0].message.images`.
"""

from __future__ import annotations

import asyncio
import base64
import io
from typing import Any

from openai import AsyncOpenAI
from PIL import Image

from app.providers.image_base import GeneratedImage, ImageProvider

ASPECT_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]


def nearest_aspect_ratio(size: tuple[int, int]) -> str:
    target = size[0] / size[1]

    def ratio(r: str) -> float:
        w, h = r.split(":")
        return int(w) / int(h)

    return min(ASPECT_RATIOS, key=lambda r: abs(ratio(r) - target))


class ImageGenerationError(RuntimeError):
    pass


class OpenRouterImageProvider(ImageProvider):
    provider_name = "openrouter"

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_seconds: float = 120,
        http_client: Any = None,  # httpx.AsyncClient, for tests
    ) -> None:
        if not api_key:
            raise ValueError("IMAGE_API_KEY (or LLM_API_KEY) is required for IMAGE_PROVIDER=openrouter")
        if not model:
            raise ValueError("IMAGE_MODEL is required for IMAGE_PROVIDER=openrouter")
        self.model = model
        self.client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout_seconds, max_retries=2, http_client=http_client
        )

    async def generate(
        self,
        prompt: str,
        negative_prompt: str | None,
        size: tuple[int, int],
        n: int,
        seed: int | None,
    ) -> list[GeneratedImage]:
        text = prompt
        if negative_prompt:
            text += f"\n\nAvoid: {negative_prompt}."
        text += "\n\nGenerate one image only. Do not include any text, letters or logos in the image."
        aspect = nearest_aspect_ratio(size)
        return list(await asyncio.gather(*(self._one(text, aspect, None if seed is None else seed + i) for i in range(n))))

    async def _one(self, text: str, aspect: str, seed: int | None) -> GeneratedImage:
        extra: dict[str, Any] = {
            "modalities": ["image", "text"],
            "image_config": {"aspect_ratio": aspect},
            "usage": {"include": True},
        }
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": text}],
            seed=seed,
            extra_body=extra,
        )
        msg = resp.choices[0].message if resp.choices else None
        images = ((msg.model_extra or {}).get("images") if msg else None) or []
        if not images:
            said = (msg.content or "")[:300] if msg else ""
            raise ImageGenerationError(f"{self.model} returned no image" + (f": {said}" if said else ""))
        url = images[0]["image_url"]["url"]
        header, b64 = url.split(",", 1)
        data = base64.b64decode(b64)
        with Image.open(io.BytesIO(data)) as im:
            width, height = im.size
        cost = float(((resp.usage.model_extra or {}).get("cost") or 0.0)) if resp.usage else 0.0
        mime = header.removeprefix("data:").split(";")[0] or "image/png"
        return GeneratedImage(data=data, width=width, height=height, seed=seed, mime=mime, cost_usd=cost)
