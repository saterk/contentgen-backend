"""Orchestrator tools and the per-phase tool-call budget."""

from __future__ import annotations


class ToolBudgetExceeded(RuntimeError):
    pass


class ToolBudget:
    """Counts tool calls against a limit (spec 5.1: 6 per request phase, 3 per piece)."""

    def __init__(self, limit: int, scope: str) -> None:
        self.limit = limit
        self.scope = scope
        self.calls: list[str] = []

    def use(self, tool: str) -> None:
        if len(self.calls) >= self.limit:
            raise ToolBudgetExceeded(
                f"{self.scope}: tool call limit of {self.limit} reached (calls: {self.calls}); refused '{tool}'"
            )
        self.calls.append(tool)

    @property
    def remaining(self) -> int:
        return self.limit - len(self.calls)
