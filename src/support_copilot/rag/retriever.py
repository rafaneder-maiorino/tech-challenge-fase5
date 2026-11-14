"""Search the manual and keep only evidence strong enough to ground an answer."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

from langchain_community.vectorstores import FAISS

if TYPE_CHECKING:
    from support_copilot.rag.config import RagSettings


class Reranker(Protocol):
    def rerank(self, query: str, documents: list[str]) -> object: ...  # iterable of floats


@dataclass(frozen=True)
class Hit:
    text: str
    page: int
    section: str
    score: float  # cosine similarity in [-1, 1]
    chunk_id: int
    rerank: float | None = None  # cross-encoder logit, when a reranker is configured


class ManualRetriever:
    def __init__(
        self,
        store: FAISS,
        top_k: int,
        min_score: float,
        reranker: Reranker | None = None,
        rerank_min: float = 0.0,
        fetch_k: int = 10,
    ) -> None:
        self.store = store
        self.top_k = top_k
        self.min_score = min_score
        self.reranker = reranker
        self.rerank_min = rerank_min
        self.fetch_k = fetch_k

    def search(self, question: str, k: int | None = None) -> list[Hit]:
        """Top-k hits, best first, whatever their score."""
        results = self.store.similarity_search_with_score(question, k=k or self.top_k)
        return [
            Hit(
                text=str(doc.metadata.get("body", doc.page_content)),
                page=int(doc.metadata.get("page", 0)),
                section=str(doc.metadata.get("section", "unknown")),
                score=float(score),
                chunk_id=int(doc.metadata.get("chunk_id", -1)),
            )
            for doc, score in results
        ]

    def rerank(self, question: str, hits: list[Hit]) -> list[Hit]:
        """Hits re-scored by the cross-encoder, best first."""
        if self.reranker is None or not hits:
            return hits
        scores = [float(x) for x in self.reranker.rerank(question, [h.text for h in hits])]
        return [replace(h, rerank=sc) for sc, h in sorted(zip(scores, hits, strict=True), key=lambda p: -p[0])]

    def relevant(self, question: str) -> list[Hit]:
        """Evidence strong enough to ground an answer. Empty list = the manual does not cover it.

        Without a reranker: top_k hits with cosine >= min_score. With one: fetch_k candidates by
        cosine, re-scored by the cross-encoder, top_k with logit >= rerank_min."""
        if self.reranker is None:
            return [h for h in self.search(question) if h.score >= self.min_score]
        ranked = self.rerank(question, self.search(question, k=self.fetch_k))
        return [h for h in ranked[: self.top_k] if h.rerank is not None and h.rerank >= self.rerank_min]

    @property
    def mode(self) -> str:
        return "cosine" if self.reranker is None else "rerank"


def load_reranker(settings: RagSettings) -> Reranker | None:
    if settings.reranker.lower() in ("", "none", "off"):
        return None
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    return TextCrossEncoder(settings.reranker, cache_dir=str(settings.model_cache_dir))


def build_retriever(settings: RagSettings) -> ManualRetriever:
    """The one place that turns settings into a retriever (API, eval scripts)."""
    from support_copilot.rag.ingest import load_or_build

    return ManualRetriever(
        load_or_build(settings),
        top_k=settings.top_k,
        min_score=settings.min_score,
        reranker=load_reranker(settings),
        rerank_min=settings.rerank_min,
        fetch_k=settings.fetch_k,
    )
