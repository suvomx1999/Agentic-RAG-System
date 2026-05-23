"""LangGraph node functions for the Agentic RAG system.

Each node accepts the current state and returns a partial state update dict.
LangGraph merges the returned dict into the full state automatically.
"""

from __future__ import annotations

from typing import List

import structlog
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from agentic_rag.config import GROQ_API_KEY, LLM_MODEL
from agentic_rag.agents.state import AgentState

logger = structlog.get_logger(__name__)


def _get_llm() -> ChatGroq:
    return ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,
    )


# ── Node 1: Analyze Query ────────────────────────────────────────────────────

def analyze_query(state: AgentState) -> dict:
    """Classify query complexity and decide retrieval strategy.

    - Simple (factual, <5 words): should_retrieve=False
    - Medium: should_retrieve=True, strategy=None
    - Complex (multi-hop, >15 words): should_retrieve=True, strategy='hyde'
    """
    query = state["query"]
    history = state.get("chat_history", [])
    word_count = len(query.split())

    logger.info("analyze_query", query=query, word_count=word_count, history_len=len(history))

    if word_count < 5 and not history:
        # Simple standalone query — try direct answer without retrieval
        return {
            "should_retrieve": False,
            "rewrite_strategy": None,
        }
    elif word_count > 15:
        # Complex query — use HyDE for better retrieval
        return {
            "should_retrieve": True,
            "rewrite_strategy": "hyde",
        }
    else:
        # Medium complexity — standard retrieval
        return {
            "should_retrieve": True,
            "rewrite_strategy": None,
        }


# ── Node 2: Rewrite Query ────────────────────────────────────────────────────

def rewrite_query(state: AgentState) -> dict:
    """Rewrite the query using the selected strategy."""
    from agentic_rag.tools.query_rewriter import rewrite_query as do_rewrite

    query = state["query"]
    strategy = state.get("rewrite_strategy")

    logger.info("rewrite_query", strategy=strategy)

    result = do_rewrite(query, strategy)

    # For multi-query, join alternatives with the original
    if isinstance(result, list):
        rewritten = " | ".join(result)
    else:
        rewritten = result

    return {"rewritten_query": rewritten}


# ── Node 3: Retrieve ─────────────────────────────────────────────────────────

def retrieve(state: AgentState) -> dict:
    """Run hybrid retrieval (dense + BM25 + RRF)."""
    from agentic_rag.indexer.hybrid_retriever import HybridRetriever

    search_query = state.get("rewritten_query") or state["query"]

    logger.info("retrieve", query=search_query[:100])

    retriever = HybridRetriever()
    docs = retriever.retrieve(search_query)

    return {
        "retrieved_docs": docs,
        "retrieval_type": "hybrid",
    }


# ── Node 4: Rerank ───────────────────────────────────────────────────────────

def rerank(state: AgentState) -> dict:
    """Rerank retrieved documents using the cross-encoder."""
    from agentic_rag.tools.reranker import rerank as do_rerank

    query = state.get("rewritten_query") or state["query"]
    docs = state.get("retrieved_docs", [])

    if not docs:
        return {"reranked_docs": []}

    logger.info("rerank", num_docs=len(docs))

    reranked = do_rerank(query, docs)
    return {"reranked_docs": reranked}


# ── Node 5: Grade Documents ──────────────────────────────────────────────────

def grade_documents(state: AgentState) -> dict:
    """Grade reranked documents for relevance using LLM."""
    from agentic_rag.tools.grader import grade_documents as do_grade

    query = state["query"]
    docs = state.get("reranked_docs", [])

    if not docs:
        return {"relevant_docs": [], "irrelevant_docs": []}

    logger.info("grade_documents", num_docs=len(docs))

    relevant, irrelevant = do_grade(query, docs)

    logger.info(
        "grade_results",
        relevant=len(relevant),
        irrelevant=len(irrelevant),
    )

    return {
        "relevant_docs": relevant,
        "irrelevant_docs": irrelevant,
    }


# ── Node 6: Web Search Fallback ──────────────────────────────────────────────

def web_search_fallback(state: AgentState) -> dict:
    """Fall back to web search when vector retrieval fails."""
    from agentic_rag.tools.web_search import web_search

    query = state["query"]

    logger.info("web_search_fallback", query=query[:100])

    web_docs = web_search(query)

    return {
        "web_docs": web_docs,
        "retrieval_type": "web",
    }


# ── Node 7: Merge Context ────────────────────────────────────────────────────

def merge_context(state: AgentState) -> dict:
    """Merge relevant_docs and web_docs, deduplicate, set final_context."""
    relevant = state.get("relevant_docs", [])
    web = state.get("web_docs", [])

    # Combine all sources
    combined = relevant + web

    # Deduplicate by (source, chunk_index) or (source, title)
    seen = set()
    unique: List[Document] = []
    for doc in combined:
        source = doc.metadata.get("source", "")
        chunk_idx = doc.metadata.get("chunk_index", doc.metadata.get("title", ""))
        key = (source, str(chunk_idx))
        if key not in seen:
            seen.add(key)
            unique.append(doc)

    logger.info("merge_context", total=len(combined), unique=len(unique))

    return {"final_context": unique}


# ── Node 8: Generate ─────────────────────────────────────────────────────────

GENERATE_SYSTEM = """You are a helpful and accurate assistant. Answer the user's question based ONLY \
on the provided context documents. Include source citations in your answer by referencing the document \
source. If the context doesn't contain enough information to answer, say so clearly.

Format citations as [Source: filename] inline."""


def generate(state: AgentState) -> dict:
    """Generate an answer using the final context. Increments iterations."""
    from langchain_core.messages import AIMessage
    
    llm = _get_llm()
    query = state["query"]
    context_docs = state.get("final_context", [])
    iterations = state.get("iterations", 0)
    history = state.get("chat_history", [])

    # Build context string
    if context_docs:
        context = "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
            for d in context_docs
        )
    else:
        context = "No context documents available."

    logger.info("generate", num_context=len(context_docs), iteration=iterations + 1, history_len=len(history))

    # Construct messages with history
    messages = [SystemMessage(content=GENERATE_SYSTEM)]
    
    # Add history
    for msg in history:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg.get("content", "")))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(content=msg.get("content", "")))
            
    # Add current query
    messages.append(HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"))

    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "iterations": iterations + 1,
    }


# ── Node 9: Grade Answer ─────────────────────────────────────────────────────

def grade_answer_node(state: AgentState) -> dict:
    """Grade the generated answer and check for hallucinations."""
    from agentic_rag.tools.answer_grader import grade_answer, hallucination_check

    query = state["query"]
    answer = state.get("answer", "")
    context_docs = state.get("final_context", [])

    logger.info("grade_answer")

    # Grade answer quality
    grade = grade_answer(query, answer)

    # Check hallucination
    hall_result = hallucination_check(answer, context_docs)

    # Determine if answer is sufficient
    is_sufficient = (
        grade.addresses_query
        and grade.confidence >= 0.6
        and hall_result.grounded
    )

    logger.info(
        "grade_result",
        addresses_query=grade.addresses_query,
        confidence=grade.confidence,
        grounded=hall_result.grounded,
        is_sufficient=is_sufficient,
    )

    return {
        "grade": grade,
        "hallucination_result": {
            "grounded": hall_result.grounded,
            "unsupported_claims": hall_result.unsupported_claims,
        },
        "is_sufficient": is_sufficient,
    }


# ── Node 10: Handle Error ────────────────────────────────────────────────────

def handle_error(state: AgentState) -> dict:
    """Set error message and force exit the loop."""
    error_msg = state.get("error", "An unknown error occurred.")
    logger.error("handle_error", error=error_msg)

    return {
        "error": error_msg,
        "is_sufficient": True,  # Force exit
        "answer": f"Sorry, an error occurred: {error_msg}",
    }
