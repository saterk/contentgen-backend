"""Image provider interface (spec section 5.5)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GeneratedImage:
    data: bytes  # encoded image (PNG/JPEG/WebP)
    width: int
    height: int
    seed: int | None = None
    mime: str = "image/png"
    cost_usd: float = 0.0


class ImageProvider(ABC):
    provider_name: str
    model: str

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        negative_prompt: str | None,
        size: tuple[int, int],
        n: int,
        seed: int | None,
    ) -> list[GeneratedImage]: ...

    def supported_size(self, size: tuple[int, int]) -> tuple[int, int]:
        """Closest size the provider can produce. The image agent crops to the target."""
        return size
