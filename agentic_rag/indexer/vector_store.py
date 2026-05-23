"""ChromaDB-backed dense vector store for the Agentic RAG system.

Uses HuggingFace embeddings (BAAI/bge-m3 by default) with persistent ChromaDB.
"""

from __future__ import annotations

import os
from typing import List, Optional

import chromadb
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from agentic_rag.config import CHROMA_PATH, EMBED_MODEL, TOP_K


# ── Singleton embedding model ────────────────────────────────────────────────
_embeddings: Optional[HuggingFaceEmbeddings] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Lazily initialize and return the embedding model."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


# ── Vector Store Operations ───────────────────────────────────────────────────

def _get_client() -> chromadb.PersistentClient:
    """Create or connect to persistent ChromaDB."""
    os.makedirs(CHROMA_PATH, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PATH)


def _get_collection(client: chromadb.PersistentClient):
    """Get or create the rag_corpus collection."""
    return client.get_or_create_collection(
        name="rag_corpus",
        metadata={"hnsw:space": "cosine"},
    )


def build_index(chunks: List[Document]) -> None:
    """Build a ChromaDB index from document chunks.

    Each chunk is embedded and stored with its metadata.
    """
    if not chunks:
        print("⚠️  No chunks to index.")
        return

    embedder = get_embeddings()
    client = _get_client()
    collection = _get_collection(client)

    # Prepare data in batches (ChromaDB recommends batches of ~5000)
    batch_size = 500
    current_count = collection.count()
    
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        texts = [c.page_content for c in batch]
        metadatas = [
            {k: str(v) for k, v in c.metadata.items()}
            for c in batch
        ]
        ids = [f"chunk_{current_count + start + i}" for i in range(len(batch))]
        embeddings = embedder.embed_documents(texts)

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    print(f"✅ Indexed {len(chunks)} chunks in ChromaDB (Total: {collection.count()})")


def load_index():
    """Load the existing ChromaDB index. Returns the collection."""
    client = _get_client()
    return _get_collection(client)


def similarity_search(query: str, k: int = TOP_K) -> List[Document]:
    """Perform dense similarity search against the ChromaDB collection."""
    embedder = get_embeddings()
    client = _get_client()
    collection = _get_collection(client)

    if collection.count() == 0:
        return []

    query_embedding = embedder.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    docs: List[Document] = []
    if results["documents"] and results["documents"][0]:
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            doc = Document(
                page_content=text,
                metadata={**meta, "dense_score": 1 - dist},  # cosine distance → similarity
            )
            docs.append(doc)

    return docs
