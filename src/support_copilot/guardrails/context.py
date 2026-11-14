"""Layer 2 - context guardrail: indirect prompt injection in the retrieved passages.

The retrieved chunks are pasted into the prompt, so a poisoned page in the knowledge base is an
attack vector that never passes through the user's input (Greshake et al., 2023). Each chunk is
checked with the injection rules scoped to ``context``; flagged chunks are dropped, the clean ones
still ground the answer. If nothing clean is left, the LLM is not called.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar

from support_copilot.guardrails.rules import RuleSet


class HasText(Protocol):
    @property
    def text(self) -> str: ...


T = TypeVar("T", bound=HasText)


@dataclass(frozen=True)
class ContextVerdict(Generic[T]):
    kept: list[T]
    dropped_rules: list[str] = field(default_factory=list)

    @property
    def dropped(self) -> int:
        return len(self.dropped_rules)


class ContextGuard:
    def __init__(self, rules: RuleSet) -> None:
        self.rules = rules

    def filter(self, hits: list[T]) -> ContextVerdict[T]:
        kept: list[T] = []
        dropped: list[str] = []
        for hit in hits:
            match = self.rules.first_match(hit.text, "context")
            if match is None:
                kept.append(hit)
            else:
                dropped.append(match.rule)
        return ContextVerdict(kept, dropped)
