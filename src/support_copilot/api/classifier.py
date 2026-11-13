"""Serving wrapper around the exported champion (models/ticket_classifier.joblib + .json)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from support_copilot.api.schemas import PredictionOut, QueueScore


@dataclass(frozen=True)
class TicketClassifier:
    pipeline: Pipeline
    labels: tuple[str, ...]
    technical: frozenset[str]
    model_version: str
    candidate: str

    @classmethod
    def load(cls, models_dir: Path) -> TicketClassifier:
        model_path = models_dir / "ticket_classifier.joblib"
        meta_path = models_dir / "ticket_classifier.json"
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"model artifacts not found in {models_dir}/ — run `make train` (needs `make mlflow-server`)"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return cls(
            pipeline=joblib.load(model_path),
            labels=tuple(meta["labels"]),
            technical=frozenset(meta["technical_queues"]),
            model_version=meta["model_version"],
            candidate=meta["candidate"],
        )

    def predict(self, text: str, top_k: int = 3) -> PredictionOut:
        proba = self.pipeline.predict_proba([text])[0]
        order = np.argsort(proba)[::-1]
        best = self.labels[order[0]]
        is_technical = best in self.technical
        return PredictionOut(
            queue=best,
            confidence=float(proba[order[0]]),
            is_technical=is_technical,
            route="rag" if is_technical else "queue",
            top=[QueueScore(queue=self.labels[i], probability=float(proba[i])) for i in order[:top_k]],
            model_version=self.model_version,
        )
