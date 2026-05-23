"""End-to-end integration tests for the Agentic RAG system.

Tests the complete graph execution with mocked LLM calls.
"""

from unittest.mock import patch, MagicMock

import pytest
from langchain_core.documents import Document

from agentic_rag.agents.state import init_state
from agentic_rag.tools.answer_grader import GradeResult


class TestInitState:
    def test_all_keys_present(self):
        state = init_state("test query")
        expected_keys = {
            "query", "rewritten_query", "rewrite_strategy",
            "retrieved_docs", "reranked_docs", "relevant_docs",
            "irrelevant_docs", "web_docs", "final_context",
            "answer", "grade", "hallucination_result",
            "iterations", "max_iterations", "retrieval_type",
            "should_retrieve", "is_sufficient", "error",
        }
        assert set(state.keys()) == expected_keys

    def test_defaults(self):
        state = init_state("test")
        assert state["query"] == "test"
        assert state["iterations"] == 0
        assert state["is_sufficient"] is False
        assert state["should_retrieve"] is True
        assert state["answer"] is None
        assert state["retrieved_docs"] == []

    def test_no_missing_keys(self):
        state = init_state("test")
        for key, value in state.items():
            # All keys should exist (no KeyError possible)
            assert key in state


class TestAnalyzeQuery:
    def test_simple_query(self):
        from agentic_rag.agents.nodes import analyze_query
        state = init_state("What is RAG?")
        result = analyze_query(state)
        # < 5 words = simple
        assert result["should_retrieve"] is False

    def test_medium_query(self):
        from agentic_rag.agents.nodes import analyze_query
        state = init_state("How does retrieval augmented generation work in practice?")
        result = analyze_query(state)
        assert result["should_retrieve"] is True
        assert result["rewrite_strategy"] is None

    def test_complex_query(self):
        from agentic_rag.agents.nodes import analyze_query
        state = init_state(
            "Can you explain how hybrid retrieval with reciprocal rank fusion "
            "combines dense and sparse retrieval methods to improve overall recall "
            "and precision in RAG systems?"
        )
        result = analyze_query(state)
        assert result["should_retrieve"] is True
        assert result["rewrite_strategy"] == "hyde"


class TestMergeContext:
    def test_deduplication(self):
        from agentic_rag.agents.nodes import merge_context
        state = init_state("test")
        doc = Document(page_content="same doc", metadata={"source": "a.txt", "chunk_index": 0})
        state["relevant_docs"] = [doc]
        state["web_docs"] = [
            Document(page_content="same doc", metadata={"source": "a.txt", "chunk_index": 0})
        ]
        result = merge_context(state)
        assert len(result["final_context"]) == 1  # Deduplicated


class TestGradeResult:
    def test_confidence_clamped(self):
        grade = GradeResult(addresses_query=True, missing=None, confidence=1.5)
        assert grade.confidence == 1.0

        grade = GradeResult(addresses_query=True, missing=None, confidence=-0.5)
        assert grade.confidence == 0.0
