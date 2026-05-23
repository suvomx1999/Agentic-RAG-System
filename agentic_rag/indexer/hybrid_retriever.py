"""Hybrid retriever combining dense (ChromaDB) and sparse (BM25) retrieval.

Uses Reciprocal Rank Fusion (RRF) to merge results and deduplicates
by (source, chunk_index) to ensure unique documents.
"""

from __future__ import annotations

from typing import List, Tuple

from langchain_core.documents import Document

from agentic_rag.config import TOP_K
from agentic_rag.indexer.vector_store import similarity_search
from agentic_rag.indexer.bm25_store import bm25_search


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def _rrf_score(rank: int, k: int = 60) -> float:
    """Compute RRF score for a given rank. k=60 is the standard constant."""
    return 1.0 / (k + rank)


def _doc_key(doc: Document) -> Tuple[str, str]:
    """Create a unique key for a document based on source + chunk_index."""
    source = doc.metadata.get("source", "unknown")
    chunk_idx = doc.metadata.get("chunk_index", "0")
    return (str(source), str(chunk_idx))


def reciprocal_rank_fusion(
    result_lists: List[List[Document]],
    k: int = 60,
    top_k: int = TOP_K,
) -> List[Document]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        result_lists: List of ranked document lists from different retrievers
        k: RRF constant (default 60)
        top_k: Number of documents to return

    Returns:
        Deduplicated, merged list of top-k documents sorted by RRF score
    """
    # Accumulate RRF scores per document
    scores: dict[Tuple[str, str], float] = {}
    doc_map: dict[Tuple[str, str], Document] = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + _rrf_score(rank, k)
            # Keep the doc with the most metadata
            if key not in doc_map:
                doc_map[key] = doc

    # Sort by RRF score descending
    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

    # Build result list with RRF score in metadata
    results: List[Document] = []
    for key in sorted_keys[:top_k]:
        doc = doc_map[key]
        doc.metadata["rrf_score"] = round(scores[key], 6)
        results.append(doc)

    return results


# ── Hybrid Retriever ──────────────────────────────────────────────────────────

class HybridRetriever:
    """Combines dense and BM25 retrieval with Reciprocal Rank Fusion."""

    def __init__(self, top_k: int = TOP_K):
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Document]:
        """Run hybrid retrieval: dense + BM25 → RRF fusion → deduplicated top-k."""
        # Get results from both retrievers (fetch more than top_k for better fusion)
        fetch_k = self.top_k * 2

        dense_results = similarity_search(query, k=fetch_k)
        sparse_results = bm25_search(query, k=fetch_k)

        # Fuse with RRF
        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            top_k=self.top_k,
        )

        return fused
