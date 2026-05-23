"""Query rewriting strategies for improving retrieval quality.

Three strategies:
- HyDE: Generate a hypothetical document as the search query
- Step-back: Abstract the query to a broader concept
- Multi-query: Generate multiple alternative phrasings
"""

from __future__ import annotations

import json
from typing import List, Optional

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from agentic_rag.config import GROQ_API_KEY, LLM_MODEL


def _get_llm() -> ChatGroq:
    return ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.7,
    )


# ── Strategy: HyDE ───────────────────────────────────────────────────────────

HYDE_SYSTEM = (
    "Write a detailed hypothetical document that would perfectly answer this query. "
    "Output only the document, no preamble."
)


def hyde_rewrite(query: str) -> str:
    """Generate a hypothetical answer document to use as the search query."""
    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content=HYDE_SYSTEM),
        HumanMessage(content=query),
    ])
    result = response.content.strip()
    return result if result else query


# ── Strategy: Step-back ───────────────────────────────────────────────────────

STEPBACK_SYSTEM = (
    "Rephrase this specific question as a broader, more general search query "
    "that would find useful background context. Output only the rephrased query."
)


def stepback_rewrite(query: str) -> str:
    """Abstract the query to a broader concept for background retrieval."""
    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content=STEPBACK_SYSTEM),
        HumanMessage(content=query),
    ])
    result = response.content.strip()
    return result if result else query


# ── Strategy: Multi-query ─────────────────────────────────────────────────────

MULTI_QUERY_SYSTEM = (
    "Generate {n} alternative search queries for the same information need. "
    "Output as a JSON array of strings only. No other text."
)


def multi_query_expand(query: str, n: int = 3) -> List[str]:
    """Generate n alternative query phrasings."""
    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content=MULTI_QUERY_SYSTEM.format(n=n)),
        HumanMessage(content=query),
    ])

    try:
        text = response.content.strip()
        # Handle markdown code blocks
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        queries = json.loads(text)
        if isinstance(queries, list) and len(queries) > 0:
            return [str(q) for q in queries[:n]]
    except (json.JSONDecodeError, IndexError):
        pass

    # Fallback: return original query
    return [query]


# ── Router ────────────────────────────────────────────────────────────────────

def rewrite_query(
    query: str,
    strategy: Optional[str] = None,
) -> str | List[str]:
    """Route to the appropriate rewriting strategy.

    Args:
        query: Original user query
        strategy: 'hyde' | 'stepback' | 'multi' | None

    Returns:
        Rewritten query string (or list for multi-query)
    """
    if strategy == "hyde":
        return hyde_rewrite(query)
    elif strategy == "stepback":
        return stepback_rewrite(query)
    elif strategy == "multi":
        return multi_query_expand(query)
    else:
        return query
