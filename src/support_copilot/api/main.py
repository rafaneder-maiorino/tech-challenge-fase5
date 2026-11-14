"""FastAPI app. Run with: uv run uvicorn support_copilot.api.main:app --port 8000

Privacy by design (LGPD): the raw ticket/question text never reaches a log line or an error
response. Logs carry a truncated SHA-256 of the text and its length; 422 responses drop
pydantic's ``input`` echo, which would otherwise return the full (possibly personal) text.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from support_copilot.api.classifier import TicketClassifier
from support_copilot.api.schemas import (
    ChatIn,
    ChatOut,
    GuardrailBlockOut,
    GuardrailHit,
    HealthOut,
    PredictionOut,
    RagHealth,
    SourceOut,
    TicketIn,
    TicketOut,
)
from support_copilot.rag.chat import ChatResult, ChatService, GuardrailBlocked
from support_copilot.rag.config import RagSettings
from support_copilot.rag.ingest import load_or_build
from support_copilot.rag.llm import build_generator
from support_copilot.rag.retriever import ManualRetriever
from support_copilot.ticket import TicketService

logger = logging.getLogger("support_copilot.api")


def text_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def log_event(event: str, **fields: object) -> None:
    """One JSON object per line; never pass raw ticket or question text here."""
    logger.info(json.dumps({"event": event, **fields}, ensure_ascii=False))


def configure_logging() -> None:
    """uvicorn configures only its own loggers; give the app's INFO lines a handler too.
    Skipped when the root logger already has handlers (e.g. pytest's caplog)."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logger.setLevel(logging.INFO)


def build_chat_service(settings: RagSettings) -> ChatService:
    store = load_or_build(settings)
    retriever = ManualRetriever(store, top_k=settings.top_k, min_score=settings.min_score)
    return ChatService(retriever, build_generator())


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    models_dir = Path(os.environ.get("MODEL_DIR", "models"))
    app.state.classifier = TicketClassifier.load(models_dir)
    log_event("model_loaded", model_version=app.state.classifier.model_version, models_dir=str(models_dir))
    t0 = time.perf_counter()
    app.state.rag_settings = RagSettings()
    app.state.chat = build_chat_service(app.state.rag_settings)
    app.state.tickets = TicketService(app.state.classifier, app.state.chat)
    log_event(
        "rag_ready",
        chunks=app.state.chat.retriever.store.index.ntotal,
        provider=app.state.chat.generator.name,
        seconds=round(time.perf_counter() - t0, 1),
    )
    yield


app = FastAPI(
    title="Support Copilot",
    description="Hybrid support copilot: ticket triage with a classical model, RAG over the OPNsense manual, "
    "guardrails and drift monitoring. FIAP MLET Phase 5 Datathon.",
    version="0.4.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [{"loc": list(e.get("loc", ())), "msg": e.get("msg", ""), "type": e.get("type", "")} for e in exc.errors()]
    log_event("validation_error", path=request.url.path, errors=[e["type"] for e in errors])
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health", response_model=HealthOut, tags=["ops"])
def health(request: Request) -> HealthOut:
    clf: TicketClassifier = request.app.state.classifier
    chat: ChatService | None = getattr(request.app.state, "chat", None)
    rag = None
    if chat is not None:
        rag = RagHealth(
            chunks=chat.retriever.store.index.ntotal,
            provider=chat.generator.name,
            llm_model=chat.generator.model,
            min_score=chat.retriever.min_score,
        )
    return HealthOut(status="ok", model_version=clf.model_version, candidate=clf.candidate, rag=rag)


@app.post("/predict", response_model=PredictionOut, tags=["triage"])
def predict(ticket: TicketIn, request: Request) -> PredictionOut:
    """Classify a support ticket into one of the 10 queues and decide whether it goes to the RAG."""
    clf: TicketClassifier = request.app.state.classifier
    t0 = time.perf_counter()
    out = clf.predict(ticket.text)
    log_event(
        "predict",
        text_sha=text_fingerprint(ticket.text),
        text_len=len(ticket.text),
        queue=out.queue,
        confidence=round(out.confidence, 4),
        route=out.route,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        model_version=out.model_version,
    )
    return out


@app.post(
    "/chat",
    response_model=ChatOut,
    tags=["rag"],
    responses={400: {"model": GuardrailBlockOut, "description": "Blocked by the input guardrail; LLM not called."}},
)
def chat(body: ChatIn, request: Request) -> ChatOut | JSONResponse:
    """Answer from the manual only, citing pages, behind three guardrail layers.

    Input guardrail (forbidden words, prompt injection) -> 400. PII is masked before retrieval and
    the LLM. Below the similarity threshold, or if every retrieved chunk looks poisoned, the LLM is
    not called. Answers without a citation to the retrieved pages are replaced by a refusal.
    """
    service: ChatService = request.app.state.chat
    t0 = time.perf_counter()
    try:
        result = service.ask(body.question)
    except GuardrailBlocked as blocked:
        v = blocked.verdict
        log_event(
            "guardrail_block",
            text_sha=text_fingerprint(body.question),
            text_len=len(body.question),
            guardrail=v.guardrail,
            category=v.category,
            rule=v.rule,
            llm_called=False,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        return JSONResponse(
            status_code=400,
            content={
                "detail": {
                    "guardrail": v.guardrail,
                    "category": v.category,
                    "message": "Request blocked by the input guardrail. Ask a question about the manual.",
                }
            },
        )
    except Exception as exc:  # provider outage, quota, timeout: never leak the provider's message
        log_event("chat_error", text_sha=text_fingerprint(body.question), error=type(exc).__name__)
        raise HTTPException(status_code=502, detail="llm_unavailable") from exc
    log_event(
        "chat",
        text_sha=text_fingerprint(body.question),
        text_len=len(body.question),
        grounded=result.grounded,
        refusal_reason=result.refusal_reason,
        llm_called=result.llm_called,
        pii_masked=result.pii_masked,
        context_dropped=result.context_dropped,
        output_pii_masked=result.output_pii_masked,
        n_sources=len(result.sources),
        top_score=round(result.sources[0].score, 4) if result.sources else None,
        provider=result.provider,
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )
    return chat_out(result)


def chat_out(result: ChatResult) -> ChatOut:
    return ChatOut(
        answer=result.answer,
        grounded=result.grounded,
        refusal_reason=result.refusal_reason,
        provider=result.provider,
        pii_masked=result.pii_masked,
        sources=[
            SourceOut(page=h.page, section=h.section, score=round(h.score, 4), snippet=h.text[:300])
            for h in result.sources
        ],
    )


@app.post("/ticket", response_model=TicketOut, tags=["copilot"])
def ticket(body: TicketIn, request: Request) -> TicketOut:
    """End-to-end copilot: triage the ticket and, if it is technical, answer it from the manual.

    Never drops a ticket and never returns a guardrail 400: a non-technical ticket is routed to its
    queue; a technical one is answered from the manual or escalated to the predicted queue with the
    reason (manual does not cover it, guardrail blocked it, or the LLM is unavailable).
    """
    service: TicketService = request.app.state.tickets
    t0 = time.perf_counter()
    res = service.handle(body.text)
    chat = res.chat
    log_event(
        "ticket",
        text_sha=text_fingerprint(body.text),
        text_len=len(body.text),
        queue=res.queue,
        confidence=round(res.triage.confidence, 4),
        action=res.action,
        escalation_reason=res.escalation_reason,
        guardrail=res.blocked_by.guardrail if res.blocked_by else None,
        rule=res.blocked_by.rule if res.blocked_by else None,
        error=res.error,
        llm_called=chat.llm_called if chat else False,
        pii_masked=chat.pii_masked if chat else {},
        n_sources=len(chat.sources) if chat else 0,
        top_score=round(chat.sources[0].score, 4) if chat and chat.sources else None,
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        model_version=res.triage.model_version,
    )
    return TicketOut(
        action=res.action,
        queue=res.queue,
        triage=res.triage,
        answer=chat_out(chat) if chat else None,
        escalation_reason=res.escalation_reason,
        blocked_by=(
            GuardrailHit(guardrail=res.blocked_by.guardrail, category=res.blocked_by.category)
            if res.blocked_by
            else None
        ),
    )
