"""Request/response contracts. Validation happens here, before any model code runs."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from support_copilot.data.source import MAX_CHARS

# Whitespace is stripped BEFORE the length checks, so "   " is rejected as empty.
TicketText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_CHARS)]


class TicketIn(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "text": "Locked out of the firewall\nI forgot the root password. "
                    "How can I reset it without reinstalling?"
                },
                {"text": "VPN keeps disconnecting\nSince the firmware update the VPN drops every few minutes."},
                {"text": "Invoice charged twice\nI was billed two times for the November subscription."},
            ]
        },
    )
    text: TicketText = Field(description=f"Ticket subject and body. 1 to {MAX_CHARS} characters after trimming.")


class QueueScore(BaseModel):
    queue: str
    probability: float = Field(ge=0.0, le=1.0)


class PredictionOut(BaseModel):
    queue: str = Field(description="Most likely support queue.")
    confidence: float = Field(ge=0.0, le=1.0, description="Probability of the predicted queue.")
    is_technical: bool = Field(description="True when the queue is routed to the RAG assistant.")
    route: Literal["rag", "queue"]
    top: list[QueueScore] = Field(description="Top 3 queues by probability.")
    model_version: str


class ChatIn(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"question": "How do I reset the root password without reinstalling?"},
                {"question": "How do I configure WAN failover between two internet providers?"},
            ]
        },
    )
    question: TicketText = Field(description=f"Question about the manual. 1 to {MAX_CHARS} characters after trimming.")


class SourceOut(BaseModel):
    page: int = Field(description="1-based page in data/manuals/opnsense-support-manual.pdf")
    section: str
    score: float = Field(description="Cosine similarity between question and passage.")
    snippet: str


class ChatOut(BaseModel):
    answer: str
    grounded: bool = Field(description="False when the manual does not cover the question.")
    refusal_reason: str | None = Field(
        default=None,
        description="no_relevant_context or context_rejected (LLM not called); model_declined, "
        "missing_citation, citation_not_in_sources or prompt_leak (answer replaced by the output guardrail).",
    )
    sources: list[SourceOut]
    provider: str = Field(description="fake | gemini | openai")
    pii_masked: dict[str, int] = Field(
        default_factory=dict,
        description="Personal data masked in the question before retrieval and the LLM (type -> count).",
    )


class GuardrailBlock(BaseModel):
    guardrail: Literal["input.blocklist", "input.injection"]
    category: str = Field(description="e.g. forbidden_words, harmful_content, instruction_override, role_play")
    message: str


class GuardrailBlockOut(BaseModel):
    detail: GuardrailBlock


class GuardrailHit(BaseModel):
    guardrail: Literal["input.blocklist", "input.injection"]
    category: str


class TicketOut(BaseModel):
    action: Literal["routed", "answered", "escalated"] = Field(
        description="routed: non-technical queue, no LLM. answered: technical and answered from the manual. "
        "escalated: technical, but sent to a person (see escalation_reason)."
    )
    queue: str = Field(description="Queue the ticket lands in (the triage prediction).")
    triage: PredictionOut
    answer: ChatOut | None = Field(default=None, description="Assistant output when the RAG ran.")
    escalation_reason: str | None = Field(
        default=None,
        description="guardrail_blocked, llm_unavailable, or the /chat refusal reason "
        "(no_relevant_context, context_rejected, model_declined, missing_citation, ...).",
    )
    blocked_by: GuardrailHit | None = Field(default=None, description="Set when the input guardrail stopped the RAG.")


class RagHealth(BaseModel):
    chunks: int
    provider: str
    llm_model: str
    min_score: float


class HealthOut(BaseModel):
    status: Literal["ok"]
    model_version: str
    candidate: str
    rag: RagHealth | None = None
