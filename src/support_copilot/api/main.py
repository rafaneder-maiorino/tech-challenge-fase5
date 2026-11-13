"""FastAPI app. Run with: uv run uvicorn support_copilot.api.main:app --port 8000

Privacy by design (LGPD): the raw ticket text never reaches a log line or an error response.
Logs carry a truncated SHA-256 of the text and its length; 422 responses drop pydantic's
``input`` echo, which would otherwise return the full (possibly personal) text to the caller.
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

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from support_copilot.api.classifier import TicketClassifier
from support_copilot.api.schemas import HealthOut, PredictionOut, TicketIn

logger = logging.getLogger("support_copilot.api")


def text_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def log_event(event: str, **fields: object) -> None:
    """One JSON object per line; never pass raw ticket text here."""
    logger.info(json.dumps({"event": event, **fields}, ensure_ascii=False))


def configure_logging() -> None:
    """uvicorn configures only its own loggers; give the app's INFO lines a handler too.
    Skipped when the root logger already has handlers (e.g. pytest's caplog)."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logger.setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    models_dir = Path(os.environ.get("MODEL_DIR", "models"))
    app.state.classifier = TicketClassifier.load(models_dir)
    log_event("model_loaded", model_version=app.state.classifier.model_version, models_dir=str(models_dir))
    yield


app = FastAPI(
    title="Support Copilot",
    description="Hybrid support copilot: ticket triage with a classical model, RAG over a manual (next), "
    "guardrails and drift monitoring. FIAP MLET Phase 5 Datathon.",
    version="0.1.0",
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
    return HealthOut(status="ok", model_version=clf.model_version, candidate=clf.candidate)


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
