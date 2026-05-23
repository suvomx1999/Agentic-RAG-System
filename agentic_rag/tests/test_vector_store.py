"""Tests for vector store and hybrid retrieval."""

import pytest
from langchain_core.documents import Document


class TestRRF:
    def test_rrf_deduplication(self):
        from agentic_rag.indexer.hybrid_retriever import reciprocal_rank_fusion

        # Same doc appears in both lists
        doc1 = Document(page_content="doc1", metadata={"source": "a.txt", "chunk_index": 0})
        doc2 = Document(page_content="doc2", metadata={"source": "b.txt", "chunk_index": 0})
        doc1_dup = Document(page_content="doc1", metadata={"source": "a.txt", "chunk_index": 0})

        results = reciprocal_rank_fusion(
            [[doc1, doc2], [doc1_dup]],
            top_k=5,
        )

        # doc1 should appear only once
        keys = [(d.metadata["source"], str(d.metadata["chunk_index"])) for d in results]
        assert len(keys) == len(set(keys)), "Duplicates found!"

    def test_rrf_score_in_metadata(self):
        from agentic_rag.indexer.hybrid_retriever import reciprocal_rank_fusion

        doc1 = Document(page_content="doc1", metadata={"source": "a.txt", "chunk_index": 0})
        results = reciprocal_rank_fusion([[doc1]], top_k=5)
        assert "rrf_score" in results[0].metadata

    def test_rrf_top_k_limit(self):
        from agentic_rag.indexer.hybrid_retriever import reciprocal_rank_fusion

        docs = [
            Document(page_content=f"doc{i}", metadata={"source": f"{i}.txt", "chunk_index": 0})
            for i in range(20)
        ]
        results = reciprocal_rank_fusion([docs], top_k=5)
        assert len(results) == 5
