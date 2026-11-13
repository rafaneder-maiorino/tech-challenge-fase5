"""Question -> evidence -> answer. Without evidence above the threshold the LLM is never called."""

from __future__ import annotations

from dataclasses import dataclass, field

from support_copilot.rag.llm import AnswerGenerator
from support_copilot.rag.prompts import REFUSAL
from support_copilot.rag.retriever import Hit, ManualRetriever


@dataclass(frozen=True)
class ChatResult:
    answer: str
    grounded: bool
    provider: str
    sources: list[Hit] = field(default_factory=list)
    refusal_reason: str | None = None
    llm_called: bool = False


class ChatService:
    def __init__(self, retriever: ManualRetriever, generator: AnswerGenerator) -> None:
        self.retriever = retriever
        self.generator = generator

    def ask(self, question: str) -> ChatResult:
        hits = self.retriever.relevant(question)
        if not hits:
            return ChatResult(REFUSAL, False, self.generator.name, [], "no_relevant_context", llm_called=False)
        answer = self.generator.generate(question, hits)
        if REFUSAL.rstrip(".").lower() in answer.lower():
            return ChatResult(REFUSAL, False, self.generator.name, hits, "model_declined", llm_called=True)
        return ChatResult(answer, True, self.generator.name, hits, None, llm_called=True)
