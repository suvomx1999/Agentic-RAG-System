"""LangGraph graph assembly for the Agentic RAG system.

Assembles all nodes and edges into a compiled StateGraph.

Mermaid diagram:

    graph TD
        A[analyze_query] -->|should_retrieve=False| G[generate]
        A -->|strategy set| RW[rewrite_query]
        A -->|default| R[retrieve]
        RW --> R
        R --> RR[rerank]
        RR --> GD[grade_documents]
        GD -->|relevant docs found| MC[merge_context]
        GD -->|no relevant docs| WS[web_search_fallback]
        WS --> MC
        MC --> G
        G --> GA[grade_answer]
        GA -->|sufficient| END
        GA -->|not addressing query| RW
        GA -->|hallucination| G
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from agentic_rag.agents.state import AgentState
from agentic_rag.agents.nodes import (
    analyze_query,
    rewrite_query,
    retrieve,
    rerank,
    grade_documents,
    web_search_fallback,
    merge_context,
    generate,
    grade_answer_node,
    handle_error,
)
from agentic_rag.agents.edges import (
    route_after_analysis,
    route_after_grading,
    route_after_answer_grade,
)


def build_graph():
    """Build and compile the Agentic RAG LangGraph.

    Returns:
        Compiled LangGraph application ready for invoke/astream
    """
    graph = StateGraph(AgentState)

    # ── Add all nodes ─────────────────────────────────────────────────────
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("web_search_fallback", web_search_fallback)
    graph.add_node("merge_context", merge_context)
    graph.add_node("generate", generate)
    graph.add_node("grade_answer", grade_answer_node)
    graph.add_node("handle_error", handle_error)

    # ── Set entry point ───────────────────────────────────────────────────
    graph.set_entry_point("analyze_query")

    # ── Conditional edge: after analysis ──────────────────────────────────
    graph.add_conditional_edges(
        "analyze_query",
        route_after_analysis,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "retrieve": "retrieve",
        },
    )

    # ── Fixed edge: rewrite → retrieve ────────────────────────────────────
    graph.add_edge("rewrite_query", "retrieve")

    # ── Fixed edges: retrieve → rerank → grade_documents ──────────────────
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "grade_documents")

    # ── Conditional edge: after grading ───────────────────────────────────
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "merge_context": "merge_context",
            "web_search_fallback": "web_search_fallback",
        },
    )

    # ── Fixed edge: web_search → merge_context ────────────────────────────
    graph.add_edge("web_search_fallback", "merge_context")

    # ── Fixed edge: merge_context → generate ──────────────────────────────
    graph.add_edge("merge_context", "generate")

    # ── Fixed edge: generate → grade_answer ───────────────────────────────
    graph.add_edge("generate", "grade_answer")

    # ── Conditional edge: after answer grading ────────────────────────────
    graph.add_conditional_edges(
        "grade_answer",
        route_after_answer_grade,
        {
            "__end__": END,
            "rewrite_query": "rewrite_query",
            "generate": "generate",
        },
    )

    # ── Compile ───────────────────────────────────────────────────────────
    app = graph.compile()
    return app


def get_graph_mermaid() -> str:
    """Return the Mermaid diagram string for the graph."""
    app = build_graph()
    return app.get_graph().draw_mermaid()


if __name__ == "__main__":
    # Smoke test: build the graph
    app = build_graph()
    print("✅ Graph compiled successfully!")
    print("\n📊 Mermaid diagram:")
    print(get_graph_mermaid())
