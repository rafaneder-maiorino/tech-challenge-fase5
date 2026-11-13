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
        default=None, description="no_relevant_context (LLM not called) or model_declined."
    )
    sources: list[SourceOut]
    provider: str = Field(description="fake | gemini | openai")


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
