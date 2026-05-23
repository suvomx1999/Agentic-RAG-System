"""Tests for conditional edge router functions."""

import pytest

from agentic_rag.agents.edges import (
    route_after_analysis,
    route_after_grading,
    route_after_answer_grade,
    route_after_rewrite,
)
from agentic_rag.agents.state import init_state
from agentic_rag.tools.answer_grader import GradeResult


class TestRouteAfterAnalysis:
    def test_no_retrieval(self):
        state = init_state("Hi")
        state["should_retrieve"] = False
        assert route_after_analysis(state) == "generate"

    def test_with_rewrite_strategy(self):
        state = init_state("What is the detailed architecture of RAG systems?")
        state["should_retrieve"] = True
        state["rewrite_strategy"] = "hyde"
        assert route_after_analysis(state) == "rewrite_query"

    def test_standard_retrieval(self):
        state = init_state("What is RAG?")
        state["should_retrieve"] = True
        state["rewrite_strategy"] = None
        assert route_after_analysis(state) == "retrieve"


class TestRouteAfterGrading:
    def test_has_relevant_docs(self):
        from langchain_core.documents import Document
        state = init_state("test")
        state["relevant_docs"] = [Document(page_content="doc")]
        assert route_after_grading(state) == "merge_context"

    def test_no_relevant_max_iterations(self):
        state = init_state("test")
        state["relevant_docs"] = []
        state["iterations"] = 3
        state["max_iterations"] = 3
        assert route_after_grading(state) == "merge_context"

    def test_no_relevant_fallback(self):
        state = init_state("test")
        state["relevant_docs"] = []
        state["iterations"] = 0
        assert route_after_grading(state) == "web_search_fallback"


class TestRouteAfterAnswerGrade:
    def test_sufficient(self):
        state = init_state("test")
        state["is_sufficient"] = True
        state["iterations"] = 1
        assert route_after_answer_grade(state) == "__end__"

    def test_max_iterations(self):
        state = init_state("test")
        state["is_sufficient"] = False
        state["iterations"] = 3
        state["max_iterations"] = 3
        assert route_after_answer_grade(state) == "__end__"

    def test_missing_info_rewrite(self):
        state = init_state("test")
        state["is_sufficient"] = False
        state["iterations"] = 1
        state["max_iterations"] = 3
        state["grade"] = GradeResult(
            addresses_query=False, missing="specific details", confidence=0.3
        )
        state["hallucination_result"] = {"grounded": True}
        assert route_after_answer_grade(state) == "rewrite_query"

    def test_hallucination_regenerate(self):
        state = init_state("test")
        state["is_sufficient"] = False
        state["iterations"] = 1
        state["max_iterations"] = 3
        state["grade"] = GradeResult(
            addresses_query=True, missing=None, confidence=0.7
        )
        state["hallucination_result"] = {"grounded": False, "unsupported_claims": ["claim1"]}
        assert route_after_answer_grade(state) == "generate"

    def test_default_end(self):
        state = init_state("test")
        state["is_sufficient"] = False
        state["iterations"] = 1
        state["max_iterations"] = 3
        state["grade"] = GradeResult(
            addresses_query=True, missing=None, confidence=0.8
        )
        state["hallucination_result"] = {"grounded": True}
        assert route_after_answer_grade(state) == "__end__"


class TestRouteAfterRewrite:
    def test_always_retrieve(self):
        state = init_state("test")
        assert route_after_rewrite(state) == "retrieve"
