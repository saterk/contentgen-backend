"""Item and request status flow (spec section 4.5)."""

from __future__ import annotations

from enum import Enum


class ItemStatus(str, Enum):
    queued = "queued"
    generating_copy = "generating_copy"
    copy_check_failed = "copy_check_failed"
    generating_images = "generating_images"
    image_check_failed = "image_check_failed"
    pending_review = "pending_review"
    needs_attention = "needs_attention"
    # later phases
    approved = "approved"
    rejected = "rejected"
    edited = "edited"
    exported = "exported"


class RequestStatus(str, Enum):
    running = "running"
    done = "done"
    failed = "failed"


S = ItemStatus

_ALLOWED: dict[ItemStatus, set[ItemStatus]] = {
    S.queued: {S.generating_copy},
    S.generating_copy: {S.copy_check_failed, S.generating_images, S.pending_review},
    S.copy_check_failed: {S.generating_copy},  # one automatic retry
    S.generating_images: {S.image_check_failed, S.pending_review},
    S.image_check_failed: {S.generating_images},  # one automatic retry
    # The variety check may regenerate a finished piece with a new angle.
    S.pending_review: {S.generating_copy, S.approved, S.rejected, S.edited},
    S.edited: {S.pending_review},
    S.approved: {S.exported},
    S.needs_attention: set(),
    S.rejected: set(),
    S.exported: set(),
}

# Any in-progress state may fall to needs_attention (failed retry, timeout, error).
_TO_NEEDS_ATTENTION = {
    S.queued,
    S.generating_copy,
    S.copy_check_failed,
    S.generating_images,
    S.image_check_failed,
}


def can_transition(current: ItemStatus | str, new: ItemStatus | str) -> bool:
    cur, nxt = ItemStatus(current), ItemStatus(new)
    if nxt is S.needs_attention:
        return cur in _TO_NEEDS_ATTENTION
    return nxt in _ALLOWED[cur]
