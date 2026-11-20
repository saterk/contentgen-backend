"""Pipeline events. The chat endpoint maps each one to a chat.completion.chunk (spec 7.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# x_event types
REQUIREMENT = "requirement"
PLAN = "plan"
STEP = "step"
BRIEF_READY = "brief_ready"
IMAGES_READY = "images_ready"
CHECK_FAILED = "check_failed"
NEEDS_INPUT = "needs_input"
ITEM_READY = "item_ready"
DONE = "done"
ERROR = "error"

FINAL_TYPES = {NEEDS_INPUT, DONE, ERROR}


@dataclass
class Event:
    type: str | None = None  # None: progress text only, no x_event
    text: str = ""
    piece: int | None = None
    item_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def final(self) -> bool:
        return self.type in FINAL_TYPES

    def x_event(self) -> dict[str, Any] | None:
        if not self.type:
            return None
        ev: dict[str, Any] = {"type": self.type}
        if self.piece is not None:
            ev["piece"] = self.piece
        if self.item_id:
            ev["item_id"] = self.item_id
        ev.update(self.data)
        return ev


def text(t: str) -> Event:
    return Event(text=t)
