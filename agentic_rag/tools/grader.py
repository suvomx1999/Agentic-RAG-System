"""LLM-based relevance grader for retrieved documents.

Uses Groq to evaluate whether each document is relevant to the query.
Supports async batch grading for concurrent evaluation.
"""

from __future__ import annotations

import asyncio
import json
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from agentic_rag.config import GROQ_API_KEY, LLM_MODEL


GRADER_SYSTEM_PROMPT = """You are a strict relevance grader. Given a user query and a document chunk, \
output ONLY a JSON object: {"relevant": true/false, "reason": "<10 words"}. No other text."""


def _get_llm() -> ChatGroq:
    """Get the grading LLM."""
    return ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,
    )


def _parse_grade(response_text: str) -> dict:
    """Parse LLM grading response. Falls back to relevant=True on failure."""
    try:
        # Try to extract JSON from the response
        text = response_text.strip()
        # Handle markdown code blocks
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        result = json.loads(text)
        return {
            "relevant": bool(result.get("relevant", True)),
            "reason": str(result.get("reason", ""))[:100],
        }
    except (json.JSONDecodeError, KeyError, IndexError):
        # Fallback: assume relevant to avoid dropping good docs
        return {"relevant": True, "reason": "parse_error_fallback"}


def grade_single(query: str, doc: Document, llm: ChatGroq) -> Document:
    """Grade a single document for relevance to the query."""
    human_msg = f"Query: {query}\n\nDocument:\n{doc.page_content[:1500]}"

    response = llm.invoke([
        SystemMessage(content=GRADER_SYSTEM_PROMPT),
        HumanMessage(content=human_msg),
    ])

    grade = _parse_grade(response.content)
    doc.metadata["grade_relevant"] = grade["relevant"]
    doc.metadata["grade_reason"] = grade["reason"]
    return doc


def grade_documents(
    query: str,
    docs: List[Document],
) -> Tuple[List[Document], List[Document]]:
    """Grade all documents for relevance. Returns (relevant_docs, irrelevant_docs)."""
    llm = _get_llm()
    relevant: List[Document] = []
    irrelevant: List[Document] = []

    for doc in docs:
        graded = grade_single(query, doc, llm)
        if graded.metadata.get("grade_relevant", True):
            relevant.append(graded)
        else:
            irrelevant.append(graded)

    return relevant, irrelevant


async def _async_grade_single(
    query: str, doc: Document, llm: ChatGroq
) -> Document:
    """Async wrapper for grading a single document."""
    human_msg = f"Query: {query}\n\nDocument:\n{doc.page_content[:1500]}"

    response = await llm.ainvoke([
        SystemMessage(content=GRADER_SYSTEM_PROMPT),
        HumanMessage(content=human_msg),
    ])

    grade = _parse_grade(response.content)
    doc.metadata["grade_relevant"] = grade["relevant"]
    doc.metadata["grade_reason"] = grade["reason"]
    return doc


async def batch_grade_documents(
    query: str,
    docs: List[Document],
    max_concurrent: int = 5,
) -> Tuple[List[Document], List[Document]]:
    """Async batch grading with concurrency limit."""
    llm = _get_llm()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _limited_grade(doc: Document) -> Document:
        async with semaphore:
            return await _async_grade_single(query, doc, llm)

    graded = await asyncio.gather(*[_limited_grade(doc) for doc in docs])

    relevant = [d for d in graded if d.metadata.get("grade_relevant", True)]
    irrelevant = [d for d in graded if not d.metadata.get("grade_relevant", True)]

    return relevant, irrelevant
