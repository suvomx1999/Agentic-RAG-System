"""Cross-encoder reranker for scoring query-document relevance.

Uses sentence-transformers CrossEncoder (ms-marco-MiniLM-L-6-v2) to
rerank retrieved documents by relevance to the query.
"""

from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document

from agentic_rag.config import RERANKER_MODEL, RERANK_TOP_N


# ── Singleton CrossEncoder ────────────────────────────────────────────────────
_cross_encoder = None


def _get_cross_encoder():
    """Lazily initialize the cross-encoder model."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(RERANKER_MODEL)
    return _cross_encoder


# ── Reranking ─────────────────────────────────────────────────────────────────

def rerank(
    query: str,
    docs: List[Document],
    top_n: Optional[int] = None,
) -> List[Document]:
    """Rerank documents by query relevance using the cross-encoder.

    Args:
        query: The search query
        docs: List of candidate documents to rerank
        top_n: Number of top documents to return (default: RERANK_TOP_N)

    Returns:
        Top-N documents sorted descending by rerank_score
    """
    if not docs:
        return []

    top_n = top_n or RERANK_TOP_N
    encoder = _get_cross_encoder()

    # Score each (query, document) pair
    pairs = [(query, doc.page_content) for doc in docs]
    scores = encoder.predict(pairs)

    # Attach scores and sort
    scored_docs: List[Document] = []
    for doc, score in zip(docs, scores):
        new_doc = Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "rerank_score": float(score)},
        )
        scored_docs.append(new_doc)

    scored_docs.sort(key=lambda d: d.metadata["rerank_score"], reverse=True)

    return scored_docs[:top_n]
