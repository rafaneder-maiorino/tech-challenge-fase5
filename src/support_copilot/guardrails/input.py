"""Layer 1 - input guardrail: forbidden words, harmful requests and direct prompt injection.

Runs BEFORE retrieval and before the LLM. A blocked question costs no embedding search and no
token; the caller turns the verdict into HTTP 400 naming the guardrail and the category.
"""

from __future__ import annotations

from dataclasses import dataclass

from support_copilot.guardrails.rules import RuleSet


@dataclass(frozen=True)
class InputVerdict:
    allowed: bool
    guardrail: str | None = None  # "input.blocklist" | "input.injection"
    category: str | None = None  # e.g. forbidden_words, instruction_override, role_play
    rule: str | None = None  # rule id: logged, never returned to the client

    @classmethod
    def ok(cls) -> InputVerdict:
        return cls(True)


class InputGuard:
    def __init__(self, rules: RuleSet) -> None:
        self.rules = rules

    def check(self, text: str) -> InputVerdict:
        match = self.rules.first_match(text, "input")
        if match is None:
            return InputVerdict.ok()
        return InputVerdict(False, f"input.{match.kind}", match.category, match.rule)
