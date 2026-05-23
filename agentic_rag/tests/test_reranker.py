"""Tests for reranker tool."""

import pytest
from langchain_core.documents import Document

from agentic_rag.tools.reranker import rerank
from agentic_rag.config import RERANK_TOP_N


def _make_docs(n: int) -> list:
    """Create n dummy documents."""
    return [
        Document(
            page_content=f"Document {i} about topic {chr(65 + i % 26)}. "
                         f"This is content for testing purposes.",
            metadata={"source": f"test_{i}.txt", "chunk_index": i}
        )
        for i in range(n)
    ]


class TestReranker:
    def test_output_length(self):
        docs = _make_docs(10)
        result = rerank("test query about topics", docs)
        assert len(result) == RERANK_TOP_N

    def test_scores_descending(self):
        docs = _make_docs(10)
        result = rerank("test query", docs)
        scores = [d.metadata["rerank_score"] for d in result]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_score_in_metadata(self):
        docs = _make_docs(5)
        result = rerank("test query", docs, top_n=3)
        for doc in result:
            assert "rerank_score" in doc.metadata

    def test_empty_input(self):
        result = rerank("test query", [])
        assert result == []

    def test_fewer_than_top_n(self):
        docs = _make_docs(2)
        result = rerank("test query", docs, top_n=5)
        assert len(result) == 2  # Can't return more than input
