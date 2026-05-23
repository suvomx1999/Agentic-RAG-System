"""RAGAS metrics configuration for evaluation.

Configures Gemini as the evaluator LLM using RAGAS 0.2 patterns.
Groups metrics into retrieval, generation, and end-to-end categories.
"""

from __future__ import annotations

import os
from typing import List

from agentic_rag.config import GOOGLE_API_KEY, EVAL_LLM_MODEL


# ── LLM & Embeddings Setup ───────────────────────────────────────────────────

def get_evaluator_llm():
    """Get the RAGAS evaluator LLM (Gemini via llm_factory)."""
    from ragas.llms import llm_factory
    from google import genai

    client = genai.Client(api_key=GOOGLE_API_KEY)
    evaluator_llm = llm_factory(
        EVAL_LLM_MODEL,
        provider="google",
        client=client,
    )
    return evaluator_llm


def get_evaluator_embeddings():
    """Get embeddings for RAGAS semantic similarity metrics."""
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from agentic_rag.config import EMBED_MODEL

    hf_embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(hf_embeddings)


# ── Metric Initialization ────────────────────────────────────────────────────

def _init_metrics():
    """Initialize all 6 RAGAS metrics with Gemini evaluator."""
    from ragas.metrics import (
        LLMContextPrecisionWithoutReference,
        LLMContextRecall,
        Faithfulness,
        ResponseRelevancy,
        FactualCorrectness,
        SemanticSimilarity,
    )

    evaluator_llm = get_evaluator_llm()
    evaluator_embeddings = get_evaluator_embeddings()

    context_precision = LLMContextPrecisionWithoutReference(llm=evaluator_llm)
    context_recall = LLMContextRecall(llm=evaluator_llm)
    faithfulness = Faithfulness(llm=evaluator_llm)
    answer_relevancy = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
    answer_correctness = FactualCorrectness(llm=evaluator_llm)
    answer_similarity = SemanticSimilarity(embeddings=evaluator_embeddings)

    return {
        "context_precision": context_precision,
        "context_recall": context_recall,
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "answer_correctness": answer_correctness,
        "answer_similarity": answer_similarity,
    }


# ── Metric Groups ────────────────────────────────────────────────────────────

def get_metrics(subset: str = "all") -> list:
    """Get metrics by group.

    Args:
        subset: 'all' | 'retrieval' | 'generation' | 'e2e'

    Returns:
        List of RAGAS metric instances
    """
    metrics = _init_metrics()

    RETRIEVAL_METRICS = [metrics["context_precision"], metrics["context_recall"]]
    GENERATION_METRICS = [metrics["faithfulness"], metrics["answer_relevancy"]]
    END_TO_END_METRICS = [metrics["answer_correctness"], metrics["answer_similarity"]]
    ALL_METRICS = RETRIEVAL_METRICS + GENERATION_METRICS + END_TO_END_METRICS

    if subset == "retrieval":
        return RETRIEVAL_METRICS
    elif subset == "generation":
        return GENERATION_METRICS
    elif subset == "e2e":
        return END_TO_END_METRICS
    else:
        return ALL_METRICS
