"""Web search fallback using DuckDuckGo (free, no API key required).

Provides web search capability when the vector index cannot answer the query.
Results are returned as LangChain Document objects with appropriate metadata.
"""

from __future__ import annotations

import time
from typing import List

import structlog
from langchain_core.documents import Document
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

logger = structlog.get_logger(__name__)

# ── Web Search ────────────────────────────────────────────────────────────────

_MAX_CONTENT_LENGTH = 800
_last_search_time = 0.0


def web_search(query: str, max_results: int = 5) -> List[Document]:
    """Search the web via DuckDuckGo and return results as Document objects.

    Args:
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        List of Document objects with metadata: source, title, retrieval_type
    """
    global _last_search_time

    # Rate limiting: 1 request per second
    elapsed = time.time() - _last_search_time
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)

    _last_search_time = time.time()

    logger.info(
        "web_search_triggered",
        query=query,
        max_results=max_results,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    try:
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=max_results)
        raw_results = wrapper.results(query, max_results=max_results)
    except Exception as e:
        logger.error("web_search_failed", error=str(e))
        return []

    docs: List[Document] = []
    for result in raw_results:
        # Truncate content to MAX_CONTENT_LENGTH
        snippet = result.get("snippet", result.get("body", ""))
        if len(snippet) > _MAX_CONTENT_LENGTH:
            snippet = snippet[:_MAX_CONTENT_LENGTH] + "..."

        doc = Document(
            page_content=snippet,
            metadata={
                "source": result.get("link", result.get("href", "unknown")),
                "title": result.get("title", "Untitled"),
                "retrieval_type": "web",
            },
        )
        docs.append(doc)

    logger.info(
        "web_search_completed",
        query=query,
        num_results=len(docs),
    )

    return docs
