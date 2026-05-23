"""Build RAGAS evaluation dataset from indexed corpus.

Generates 60 QA pairs (50 answerable + 10 unanswerable) using Groq,
categorized by question type: factual, multi-hop, reasoning, unanswerable.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from tqdm import tqdm

from agentic_rag.config import GROQ_API_KEY, LLM_MODEL

EVAL_DATASET_PATH = Path(__file__).parent / "eval_dataset.json"

# ── Prompts ───────────────────────────────────────────────────────────────────

QA_GEN_SYSTEM = """Given this document chunk, generate one question whose answer is explicitly \
contained in the chunk. Output ONLY a JSON object: \
{"question": "<question>", "ground_truth": "<answer from chunk>", "context": "<relevant excerpt>", \
"type": "<factual|multi-hop|reasoning>"}. No other text."""

UNANSWERABLE_SYSTEM = """Generate a plausible question about AI/ML/NLP that CANNOT be answered \
from typical RAG system documentation. The question should be about a very specific recent event, \
a niche implementation detail, or require real-time data. Output ONLY a JSON object: \
{"question": "<question>", "ground_truth": "This information is not available in the knowledge base.", \
"context": "", "type": "unanswerable"}. No other text."""


def _get_llm() -> ChatGroq:
    return ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.7,
    )


def _parse_qa(text: str) -> Optional[dict]:
    """Parse QA generation response."""
    try:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        return json.loads(text)
    except (json.JSONDecodeError, IndexError):
        return None


def generate_qa_pairs(chunks: List[Document], n: int = 50) -> List[dict]:
    """Generate QA pairs from sampled chunks."""
    llm = _get_llm()

    # Sample chunks
    if len(chunks) > n:
        sampled = random.sample(chunks, n)
    else:
        sampled = chunks

    qa_pairs = []
    for chunk in tqdm(sampled, desc="Generating QA pairs"):
        try:
            response = llm.invoke([
                SystemMessage(content=QA_GEN_SYSTEM),
                HumanMessage(content=f"Document chunk:\n{chunk.page_content}"),
            ])
            qa = _parse_qa(response.content)
            if qa and all(k in qa for k in ["question", "ground_truth", "context", "type"]):
                qa_pairs.append(qa)
        except Exception as e:
            print(f"⚠️  Error generating QA: {e}")
            continue

    return qa_pairs


def generate_unanswerable(n: int = 10) -> List[dict]:
    """Generate unanswerable questions."""
    llm = _get_llm()
    questions = []

    for _ in tqdm(range(n), desc="Generating unanswerable Qs"):
        try:
            response = llm.invoke([
                SystemMessage(content=UNANSWERABLE_SYSTEM),
                HumanMessage(content=f"Generate unanswerable question #{len(questions) + 1}"),
            ])
            qa = _parse_qa(response.content)
            if qa:
                qa["type"] = "unanswerable"
                questions.append(qa)
        except Exception as e:
            print(f"⚠️  Error: {e}")
            continue

    return questions


def build_dataset(chunks: List[Document]) -> List[dict]:
    """Build the complete evaluation dataset (50 answerable + 10 unanswerable)."""
    print("📝 Generating answerable QA pairs...")
    answerable = generate_qa_pairs(chunks, n=50)

    print("📝 Generating unanswerable questions...")
    unanswerable = generate_unanswerable(n=10)

    dataset = answerable + unanswerable

    # Save to disk
    EVAL_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_DATASET_PATH, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"✅ Saved {len(dataset)} QA pairs to {EVAL_DATASET_PATH}")

    # Print stats
    types = {}
    for qa in dataset:
        t = qa.get("type", "unknown")
        types[t] = types.get(t, 0) + 1
    print(f"📊 By type: {types}")

    return dataset


def load_dataset() -> List[dict]:
    """Load the evaluation dataset from disk."""
    with open(EVAL_DATASET_PATH) as f:
        return json.load(f)
