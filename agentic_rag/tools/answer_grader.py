"""Answer quality grader and hallucination detector.

Two grading tools:
- grade_answer: Checks if the answer addresses the query
- hallucination_check: Checks if the answer is grounded in context
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from agentic_rag.config import GOOGLE_API_KEY, LLM_MODEL


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class GradeResult:
    """Result from answer quality grading."""
    addresses_query: bool
    missing: Optional[str]
    confidence: float

    def __post_init__(self):
        # Clamp confidence to [0.0, 1.0]
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class HallucinationResult:
    """Result from hallucination detection."""
    grounded: bool
    unsupported_claims: List[str]


# ── LLM Setup ─────────────────────────────────────────────────────────────────

ANSWER_GRADE_SYSTEM = """You are an answer quality judge. Given a user query and a generated answer, \
output ONLY JSON: {"addresses_query": true/false, "missing": "<what is missing, or null>", \
"confidence": 0.0-1.0}. No other text."""

HALLUCINATION_SYSTEM = """You are a hallucination detector. Given a generated answer and the source \
context documents it was based on, check if every claim in the answer is grounded in the context. \
Output ONLY JSON: {"grounded": true/false, "unsupported_claims": ["list of unsupported claims"]}. \
No other text."""


def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
    )


def _parse_json_safe(text: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


# ── Answer Grader ─────────────────────────────────────────────────────────────

def grade_answer(query: str, answer: str) -> GradeResult:
    """Grade the answer's quality relative to the query.

    Returns:
        GradeResult with addresses_query, missing, and confidence fields
    """
    llm = _get_llm()

    response = llm.invoke([
        SystemMessage(content=ANSWER_GRADE_SYSTEM),
        HumanMessage(content=f"Query: {query}\n\nAnswer: {answer}"),
    ])

    try:
        result = _parse_json_safe(response.content)
        return GradeResult(
            addresses_query=bool(result.get("addresses_query", True)),
            missing=result.get("missing"),
            confidence=float(result.get("confidence", 0.5)),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        # Fallback: assume adequate
        return GradeResult(
            addresses_query=True,
            missing=None,
            confidence=0.5,
        )


# ── Hallucination Grader ─────────────────────────────────────────────────────

def hallucination_check(answer: str, docs: List[Document]) -> HallucinationResult:
    """Check if the answer is grounded in the provided context documents.

    Returns:
        HallucinationResult with grounded flag and list of unsupported claims
    """
    llm = _get_llm()

    context = "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )

    response = llm.invoke([
        SystemMessage(content=HALLUCINATION_SYSTEM),
        HumanMessage(
            content=f"Answer: {answer}\n\nContext documents:\n{context}"
        ),
    ])

    try:
        result = _parse_json_safe(response.content)
        return HallucinationResult(
            grounded=bool(result.get("grounded", True)),
            unsupported_claims=result.get("unsupported_claims", []),
        )
    except (json.JSONDecodeError, KeyError):
        # Fallback: assume grounded
        return HallucinationResult(grounded=True, unsupported_claims=[])
