"""BM25-based sparse retrieval for the Agentic RAG system.

Uses rank_bm25.BM25Okapi with pickle-based persistence.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from agentic_rag.config import BM25_PATH, TOP_K


# ── BM25 Store ────────────────────────────────────────────────────────────────

_INDEX_FILE = "bm25_index.pkl"
_CORPUS_FILE = "bm25_corpus.pkl"
_DOCS_FILE = "bm25_docs.pkl"


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenization."""
    return text.lower().split()


def build_bm25_index(chunks: List[Document]) -> None:
    """Build and persist a BM25 index from document chunks."""
    if not chunks:
        print("⚠️  No chunks to index.")
        return

    os.makedirs(BM25_PATH, exist_ok=True)

    tokenized_corpus = [_tokenize(c.page_content) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    # Persist
    with open(Path(BM25_PATH) / _INDEX_FILE, "wb") as f:
        pickle.dump(bm25, f)
    with open(Path(BM25_PATH) / _CORPUS_FILE, "wb") as f:
        pickle.dump(tokenized_corpus, f)
    with open(Path(BM25_PATH) / _DOCS_FILE, "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ BM25 index built with {len(chunks)} documents")


def load_bm25_index() -> Optional[BM25Okapi]:
    """Load the persisted BM25 index. Returns None if not found."""
    index_path = Path(BM25_PATH) / _INDEX_FILE
    if not index_path.exists():
        return None
    with open(index_path, "rb") as f:
        return pickle.load(f)


def _load_docs() -> List[Document]:
    """Load the persisted document list."""
    docs_path = Path(BM25_PATH) / _DOCS_FILE
    if not docs_path.exists():
        return []
    with open(docs_path, "rb") as f:
        return pickle.load(f)


def bm25_search(query: str, k: int = TOP_K) -> List[Document]:
    """Search the BM25 index and return top-k documents."""
    bm25 = load_bm25_index()
    if bm25 is None:
        return []

    docs = _load_docs()
    if not docs:
        return []

    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # Get top-k indices
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    results: List[Document] = []
    for idx in ranked_indices:
        if scores[idx] > 0:
            doc = Document(
                page_content=docs[idx].page_content,
                metadata={**docs[idx].metadata, "bm25_score": float(scores[idx])},
            )
            results.append(doc)

    return results
