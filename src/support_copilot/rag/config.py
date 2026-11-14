"""RAG settings. Every knob that changes the index is part of the index cache key."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

MANUAL_PATH = Path("data/manuals/opnsense-support-manual.pdf")
MANUAL_SHA256 = "792bc484e786f4d41f339bc059ef78b7aa057d93cdfd0f0211d628925e878c08"
INGEST_VERSION = 3  # bump whenever cleaning/splitting code changes: invalidates cached indexes
RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"  # cross-encoder, ONNX via fastembed (~80 MB)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # 384-d, ONNX via fastembed: no torch, runs offline after 1st download


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class RagSettings:
    manual_path: Path = MANUAL_PATH
    manual_sha256: str = MANUAL_SHA256
    embedding_model: str = EMBEDDING_MODEL
    # 1500/225 won on data/eval/rag_questions.yaml: hit@1 0.958, MRR 0.979 (reports/retrieval_eval.md)
    chunk_size: int = field(default_factory=lambda: _env_int("RAG_CHUNK_SIZE", 1500))
    chunk_overlap: int = field(default_factory=lambda: _env_int("RAG_CHUNK_OVERLAP", 225))
    top_k: int = field(default_factory=lambda: _env_int("RAG_TOP_K", 4))
    # Cosine similarity below which a question is treated as "not in the manual" and the LLM is
    # never called. Calibrated in reports/retrieval_eval.md (in-domain vs out-of-domain questions).
    min_score: float = field(default_factory=lambda: _env_float("RAG_MIN_SCORE", 0.67))
    # Cross-encoder reranker (D7): the bi-encoder fetches fetch_k candidates, the cross-encoder reads
    # question + passage together and only passages scoring >= rerank_min ground an answer. "none" =
    # cosine threshold only. rerank_min = midpoint of the gap on data/eval/rag_questions.yaml
    # (reports/rag_eval.md). Scores are ms-marco logits, not probabilities.
    reranker: str = field(default_factory=lambda: os.environ.get("RAG_RERANKER", "none"))
    rerank_min: float = field(default_factory=lambda: _env_float("RAG_RERANK_MIN", -2.5))
    fetch_k: int = field(default_factory=lambda: _env_int("RAG_FETCH_K", 10))
    index_dir: Path = field(default_factory=lambda: Path(os.environ.get("RAG_INDEX_DIR", "data/index")))
    model_cache_dir: Path = field(default_factory=lambda: Path(os.environ.get("RAG_MODEL_CACHE", "data/models_cache")))

    def cache_key(self) -> str:
        raw = f"v{INGEST_VERSION}|{self.manual_sha256}|{self.embedding_model}|{self.chunk_size}|{self.chunk_overlap}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
