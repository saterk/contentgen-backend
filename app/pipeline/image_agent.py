"""Image generation agent (spec 5.5): build the final prompt, generate, fit to size, save."""

from __future__ import annotations

import asyncio
import io
import zlib
from dataclasses import dataclass

from PIL import Image, ImageOps

from app.pipeline.context import AppContext
from app.schemas.content import ContentBrief


@dataclass
class SavedImage:
    file_name: str
    prompt: str
    negative_prompt: str | None
    seed: int | None
    variant_no: int
    width: int
    height: int
    cost_usd: float


def build_prompt(brief: ContentBrief, style_suffix: str, adjustment: str | None = None) -> str:
    parts = [brief.image_prompt.strip().rstrip(".")]
    if adjustment:
        parts.append(adjustment)
    if style_suffix:
        parts.append(style_suffix)
    return ". ".join(parts)


_EXT = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp"}


def fit_to_size(data: bytes, target: tuple[int, int]) -> tuple[bytes, int, int, str]:
    """Center-crop and resize to the target size if the provider returned another size.

    Returns (bytes, width, height, file extension). Resized images are saved as JPEG.
    """
    img = Image.open(io.BytesIO(data))
    if img.size == target:
        return data, *img.size, _EXT.get(img.format or "", "png")
    fitted = ImageOps.fit(img.convert("RGB"), target, method=Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    fitted.save(buf, format="JPEG", quality=92)
    return buf.getvalue(), *target, "jpg"


async def generate_images(
    ctx: AppContext,
    item_id: str,
    item_version: int,
    brief: ContentBrief,
    attempt: int,
    adjustment: str | None = None,
    visual_style: str | None = None,
) -> list[SavedImage]:
    s = ctx.settings
    target = s.image_wh
    prompt = build_prompt(brief, ctx.kb.image_style_suffix(visual_style), adjustment)
    base_seed = zlib.crc32(f"{item_id}:{item_version}:{attempt}".encode()) % 1_000_000
    generated = await ctx.images.generate(
        prompt, brief.negative_prompt, ctx.images.supported_size(target), s.image_variants, base_seed
    )
    saved: list[SavedImage] = []
    for n, g in enumerate(generated, start=1):
        data, w, h, ext = await asyncio.to_thread(fit_to_size, g.data, target)
        name = f"{item_id}_v{item_version}_a{attempt}_{n}.{ext}"
        mime = {"jpg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "application/octet-stream")
        await asyncio.to_thread(ctx.storage.save, name, data, mime)
        saved.append(
            SavedImage(
                file_name=name,
                prompt=prompt,
                negative_prompt=brief.negative_prompt,
                seed=g.seed,
                variant_no=n,
                width=w,
                height=h,
                cost_usd=g.cost_usd,
            )
        )
    return saved
