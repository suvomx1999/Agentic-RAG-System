"""LangGraph state schema for the Agentic RAG system.

Defines AgentState (TypedDict) with all 19 fields and
a factory function init_state(query) for initialization.
"""

from __future__ import annotations

from typing import List, Optional, TypedDict

from langchain_core.documents import Document

from agentic_rag.tools.answer_grader import GradeResult
from agentic_rag.config import MAX_ITERATIONS


class AgentState(TypedDict):
    """Complete state schema for the Agentic RAG LangGraph."""

    # ── Query ─────────────────────────────────────────────────────────────
    query: str                                  # Original user query
    rewritten_query: Optional[str]              # After rewriting
    rewrite_strategy: Optional[str]             # 'hyde'|'stepback'|'multi'|None

    # ── Retrieval ─────────────────────────────────────────────────────────
    retrieved_docs: List[Document]              # After hybrid retrieval
    reranked_docs: List[Document]               # After reranking
    relevant_docs: List[Document]               # After grading (relevant)
    irrelevant_docs: List[Document]             # After grading (filtered out)
    web_docs: List[Document]                    # From web search fallback
    final_context: List[Document]               # Merged context for generation

    # ── Generation ────────────────────────────────────────────────────────
    answer: Optional[str]                       # Generated answer
    grade: Optional[GradeResult]                # Answer quality grade
    hallucination_result: Optional[dict]        # Hallucination check result

    # ── Control Flow ──────────────────────────────────────────────────────
    iterations: int                             # Loop counter
    max_iterations: int                         # Max loops (from config)
    retrieval_type: str                         # 'vector'|'web'|'hybrid'
    should_retrieve: bool                       # Whether retrieval is needed
    is_sufficient: bool                         # Whether answer is good enough
    error: Optional[str]                        # Error message if any


def init_state(query: str) -> AgentState:
    """Factory function to create an initialized AgentState.

    All optional fields default to None, lists to [], and control
    fields to their default values.
    """
    return AgentState(
        query=query,
        rewritten_query=None,
        rewrite_strategy=None,
        retrieved_docs=[],
        reranked_docs=[],
        relevant_docs=[],
        irrelevant_docs=[],
        web_docs=[],
        final_context=[],
        answer=None,
        grade=None,
        hallucination_result=None,
        iterations=0,
        max_iterations=MAX_ITERATIONS,
        retrieval_type="vector",
        should_retrieve=True,
        is_sufficient=False,
        error=None,
    )
