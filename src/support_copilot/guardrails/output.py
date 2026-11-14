"""Layer 3 - output guardrail: the answer must be grounded, cited and free of personal data.

Checks, in order:
1. prompt leak: the answer quotes the system prompt -> replaced by the refusal;
2. citations: at least one ``[p. N]``, and every cited page must be a page of the passages that
   were actually sent to the model -> otherwise replaced by the refusal ("ungrounded_output");
3. PII: any personal data left in the answer is masked (the model may echo the context).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from support_copilot.guardrails.pii import mask_pii

CITATION = re.compile(r"\[p\.\s*(\d+)\]")


@dataclass(frozen=True)
class OutputVerdict:
    answer: str
    ok: bool
    reason: str | None = None  # prompt_leak | missing_citation | citation_not_in_sources
    pii_masked: dict[str, int] = field(default_factory=dict)


class OutputGuard:
    def __init__(self, refusal: str, canaries: Iterable[str] = ()) -> None:
        self.refusal = refusal
        self.canaries = [c.casefold() for c in canaries if c]

    def check(self, answer: str, source_pages: Iterable[int]) -> OutputVerdict:
        folded = answer.casefold()
        if any(c in folded for c in self.canaries):
            return OutputVerdict(self.refusal, False, "prompt_leak")
        cited = {int(p) for p in CITATION.findall(answer)}
        if not cited:
            return OutputVerdict(self.refusal, False, "missing_citation")
        if not cited <= set(source_pages):
            return OutputVerdict(self.refusal, False, "citation_not_in_sources")
        masked = mask_pii(answer)
        return OutputVerdict(masked.text, True, None, masked.counts)
