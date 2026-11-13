"""Search the manual and keep only evidence strong enough to ground an answer."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_community.vectorstores import FAISS


@dataclass(frozen=True)
class Hit:
    text: str
    page: int
    section: str
    score: float  # cosine similarity in [-1, 1]
    chunk_id: int


class ManualRetriever:
    def __init__(self, store: FAISS, top_k: int, min_score: float) -> None:
        self.store = store
        self.top_k = top_k
        self.min_score = min_score

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

    def relevant(self, question: str) -> list[Hit]:
        """Hits at or above the calibrated threshold. Empty list = the manual does not cover it."""
        return [h for h in self.search(question) if h.score >= self.min_score]
