"""Conditional edge router functions for the Agentic RAG LangGraph.

Each router inspects the current state and returns the name of the
next node to execute (or END to terminate).
"""

from __future__ import annotations

from typing import Literal

import structlog

from agentic_rag.agents.state import AgentState

logger = structlog.get_logger(__name__)


# ── Router 1: After Analysis ─────────────────────────────────────────────────

def route_after_analysis(state: AgentState) -> Literal["generate", "rewrite_query", "retrieve"]:
    """Route after query analysis.

    - should_retrieve == False → "generate" (direct answer)
    - rewrite_strategy set → "rewrite_query"
    - else → "retrieve" (standard retrieval)
    """
    if not state.get("should_retrieve", True):
        logger.info("route_after_analysis", decision="generate (no retrieval)")
        return "generate"
    elif state.get("rewrite_strategy") is not None:
        logger.info("route_after_analysis", decision=f"rewrite ({state['rewrite_strategy']})")
        return "rewrite_query"
    else:
        logger.info("route_after_analysis", decision="retrieve (standard)")
        return "retrieve"


# ── Router 2: After Grading ──────────────────────────────────────────────────

def route_after_grading(state: AgentState) -> Literal["merge_context", "web_search_fallback"]:
    """Route after document grading.

    - relevant_docs >= 1 → "merge_context"
    - iterations >= max_iterations → "merge_context" (use what we have)
    - else → "web_search_fallback"
    """
    relevant = state.get("relevant_docs", [])
    iterations = state.get("iterations", 0)
    max_iter = state.get("max_iterations", 3)

    if len(relevant) >= 1:
        logger.info("route_after_grading", decision="merge_context", relevant=len(relevant))
        return "merge_context"
    elif iterations >= max_iter:
        logger.info("route_after_grading", decision="merge_context (max iterations)")
        return "merge_context"
    else:
        logger.info("route_after_grading", decision="web_search_fallback")
        return "web_search_fallback"


# ── Router 3: After Answer Grade ─────────────────────────────────────────────

def route_after_answer_grade(state: AgentState) -> Literal["__end__", "rewrite_query", "generate"]:
    """Route after answer grading.

    - is_sufficient OR iterations >= max → END
    - answer doesn't address query → "rewrite_query" (try different retrieval)
    - hallucination detected → "generate" (regenerate from same context)
    - else → END
    """
    iterations = state.get("iterations", 0)
    max_iter = state.get("max_iterations", 3)
    is_sufficient = state.get("is_sufficient", False)
    grade = state.get("grade")
    hall_result = state.get("hallucination_result", {})

    # Guard: prevent infinite loops
    if is_sufficient or iterations >= max_iter:
        logger.info("route_after_answer_grade", decision="END", iterations=iterations)
        return "__end__"

    # Answer doesn't address query → rewrite and retrieve again
    if grade and not grade.addresses_query and grade.missing is not None:
        logger.info("route_after_answer_grade", decision="rewrite_query", missing=grade.missing)
        return "rewrite_query"

    # Hallucination detected → regenerate from same context
    if isinstance(hall_result, dict) and not hall_result.get("grounded", True):
        logger.info("route_after_answer_grade", decision="generate (hallucination)")
        return "generate"

    # Default: accept the answer
    logger.info("route_after_answer_grade", decision="END (default)")
    return "__end__"


# ── Router 4: After Rewrite ──────────────────────────────────────────────────

def route_after_rewrite(state: AgentState) -> Literal["retrieve"]:
    """After rewriting, always proceed to retrieval."""
    logger.info("route_after_rewrite", decision="retrieve")
    return "retrieve"
