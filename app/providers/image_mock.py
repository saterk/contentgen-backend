"""Placeholder images so the flow works without an image API key."""

from __future__ import annotations

import asyncio
import io
import random
import textwrap

from PIL import Image, ImageDraw, ImageFont

from app.providers.image_base import GeneratedImage, ImageProvider


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


class MockImageProvider(ImageProvider):
    provider_name = "mock"

    def __init__(self, model: str = "mock-image-1") -> None:
        self.model = model

    async def generate(
        self,
        prompt: str,
        negative_prompt: str | None,
        size: tuple[int, int],
        n: int,
        seed: int | None,
    ) -> list[GeneratedImage]:
        out = []
        for i in range(n):
            s = (seed or 0) + i
            data = await asyncio.to_thread(self._render, prompt, size, s)  # keep the event loop free
            out.append(GeneratedImage(data=data, width=size[0], height=size[1], seed=s))
        return out

    def _render(self, prompt: str, size: tuple[int, int], seed: int) -> bytes:
        rng = random.Random(seed)
        w, h = size
        top = (rng.randint(150, 235), rng.randint(110, 190), rng.randint(90, 160))
        bottom = (rng.randint(60, 130), rng.randint(50, 110), rng.randint(60, 120))
        img = Image.new("RGB", size)
        draw = ImageDraw.Draw(img)
        for y in range(h):
            t = y / max(1, h - 1)
            draw.line([(0, y), (w, y)], fill=tuple(int(a + (b - a) * t) for a, b in zip(top, bottom)))
        # a soft "subject" shape and the single red accent from the brand guide
        cx, cy = w // 2, int(h * 0.52)
        draw.ellipse([cx - w // 4, cy - h // 6, cx + w // 4, cy + h // 6], fill=(245, 225, 200))
        ax, ay = rng.randint(w // 5, 4 * w // 5), rng.randint(h // 3, 2 * h // 3)
        draw.ellipse([ax - 28, ay - 28, ax + 28, ay + 28], fill=(200, 30, 45))

        title = _font(max(18, w // 22))
        body = _font(max(14, w // 40))
        draw.text((40, 40), "PROTOTYPE PLACEHOLDER", fill=(255, 255, 255), font=title)
        draw.text((40, 40 + w // 18), f"mock image · seed {seed}", fill=(255, 255, 255), font=body)
        wrapped = textwrap.fill(prompt[:420], width=max(30, w // 22))
        draw.multiline_text((40, h - h // 4), wrapped, fill=(255, 255, 255), font=body, spacing=6)

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=False)
        return buf.getvalue()
