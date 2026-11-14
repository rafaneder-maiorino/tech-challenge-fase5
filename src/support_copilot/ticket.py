"""Ticket -> triage -> (technical? RAG : queue). The end-to-end flow behind POST /ticket.

A ticket is never dropped. Possible outcomes:
- routed     non-technical queue: goes to its human queue, no LLM involved
- answered   technical and the manual answers it (grounded, cited)
- escalated  technical but the assistant does not answer: the manual does not cover it, the
             guardrails stopped it, or the LLM is unavailable. It goes to the predicted queue
             with the reason, so a person picks it up.

Unlike /chat, a guardrail block here is not an HTTP error: the ticket is a customer request that
must land somewhere. The guardrail still keeps it away from retrieval and the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from support_copilot.api.classifier import TicketClassifier
from support_copilot.api.schemas import PredictionOut
from support_copilot.guardrails import InputVerdict
from support_copilot.rag.chat import ChatResult, ChatService, GuardrailBlocked

Action = Literal["routed", "answered", "escalated"]


@dataclass(frozen=True)
class TicketResult:
    action: Action
    triage: PredictionOut
    chat: ChatResult | None = None
    escalation_reason: str | None = None  # refusal reason, "guardrail_blocked" or "llm_unavailable"
    blocked_by: InputVerdict | None = None
    error: str | None = None  # exception class name when the LLM failed (for logs only)

    @property
    def queue(self) -> str:
        return self.triage.queue


class TicketService:
    def __init__(self, classifier: TicketClassifier, chat: ChatService) -> None:
        self.classifier = classifier
        self.chat = chat

    def handle(self, text: str) -> TicketResult:
        triage = self.classifier.predict(text)
        if not triage.is_technical:
            return TicketResult("routed", triage)
        try:
            result = self.chat.ask(text)
        except GuardrailBlocked as blocked:
            return TicketResult("escalated", triage, escalation_reason="guardrail_blocked", blocked_by=blocked.verdict)
        except Exception as exc:  # provider outage/quota/timeout: degrade to the human queue, never 5xx
            return TicketResult("escalated", triage, escalation_reason="llm_unavailable", error=type(exc).__name__)
        if result.grounded:
            return TicketResult("answered", triage, chat=result)
        return TicketResult("escalated", triage, chat=result, escalation_reason=result.refusal_reason)
