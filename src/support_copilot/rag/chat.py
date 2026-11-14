"""Question -> guardrails -> evidence -> answer -> guardrails.

Order of the pipeline (each step can stop it, and the LLM is the last thing to run):
1. input guardrail   blocked -> GuardrailBlocked (HTTP 400), nothing else runs
2. PII masking       the masked question is what retrieval and the LLM see
3. retrieval         nothing above the threshold -> refusal, LLM not called
4. context guardrail poisoned chunks dropped; none left -> refusal, LLM not called
5. LLM
6. output guardrail  no/foreign citation or prompt leak -> refusal; PII masked in the answer
"""

from __future__ import annotations

from dataclasses import dataclass, field

from support_copilot.guardrails import ContextGuard, InputGuard, InputVerdict, OutputGuard, RuleSet, mask_pii
from support_copilot.rag.llm import AnswerGenerator
from support_copilot.rag.prompts import REFUSAL, SYSTEM_PROMPT_CANARIES
from support_copilot.rag.retriever import Hit, ManualRetriever


class GuardrailBlocked(Exception):
    """Raised when the input guardrail blocks a question. Carries the verdict for the response."""

    def __init__(self, verdict: InputVerdict) -> None:
        super().__init__(f"{verdict.guardrail}:{verdict.category}")
        self.verdict = verdict


@dataclass(frozen=True)
class ChatResult:
    answer: str
    grounded: bool
    provider: str
    sources: list[Hit] = field(default_factory=list)
    refusal_reason: str | None = None
    llm_called: bool = False
    pii_masked: dict[str, int] = field(default_factory=dict)  # type -> count, in the question
    context_dropped: list[str] = field(default_factory=list)  # rule ids of dropped chunks
    output_pii_masked: dict[str, int] = field(default_factory=dict)


class ChatService:
    def __init__(
        self,
        retriever: ManualRetriever,
        generator: AnswerGenerator,
        rules: RuleSet | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.rules = rules or RuleSet.load()
        self.input_guard = InputGuard(self.rules)
        self.context_guard = ContextGuard(self.rules)
        self.output_guard = OutputGuard(REFUSAL, SYSTEM_PROMPT_CANARIES)

    def ask(self, question: str) -> ChatResult:
        verdict = self.input_guard.check(question)
        if not verdict.allowed:
            raise GuardrailBlocked(verdict)

        name = self.generator.name
        masked = mask_pii(question)
        hits = self.retriever.relevant(masked.text)
        if not hits:
            return ChatResult(REFUSAL, False, name, refusal_reason="no_relevant_context", pii_masked=masked.counts)

        ctx = self.context_guard.filter(hits)
        if not ctx.kept:
            return ChatResult(
                REFUSAL,
                False,
                name,
                refusal_reason="context_rejected",
                pii_masked=masked.counts,
                context_dropped=ctx.dropped_rules,
            )

        answer = self.generator.generate(masked.text, ctx.kept)
        if REFUSAL.rstrip(".").lower() in answer.lower():
            reason, out_pii = "model_declined", {}
        else:
            out = self.output_guard.check(answer, (h.page for h in ctx.kept))
            if out.ok:
                return ChatResult(
                    out.answer,
                    True,
                    name,
                    ctx.kept,
                    None,
                    llm_called=True,
                    pii_masked=masked.counts,
                    context_dropped=ctx.dropped_rules,
                    output_pii_masked=out.pii_masked,
                )
            reason, out_pii = out.reason or "output_rejected", {}
        return ChatResult(
            REFUSAL,
            False,
            name,
            ctx.kept,
            reason,
            llm_called=True,
            pii_masked=masked.counts,
            context_dropped=ctx.dropped_rules,
            output_pii_masked=out_pii,
        )
