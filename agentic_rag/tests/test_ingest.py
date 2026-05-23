"""Tests for document ingestion and chunking."""

import os
import tempfile
from pathlib import Path

import pytest
from langchain_core.documents import Document

from agentic_rag.indexer.ingest import DocumentLoader, Chunker


@pytest.fixture
def sample_text_file(tmp_path):
    """Create a sample text file for testing."""
    content = (
        "This is a test document about retrieval-augmented generation. "
        "RAG systems combine retrieval and generation to produce accurate answers. "
        "The retrieval component finds relevant documents from a knowledge base. "
        "The generation component uses an LLM to synthesize the final answer. "
        "This approach helps reduce hallucinations and improve factual accuracy. "
        "RAG is widely used in enterprise question answering systems. "
    ) * 5  # Make it long enough to produce multiple chunks
    file_path = tmp_path / "test_doc.txt"
    file_path.write_text(content)
    return str(file_path)


@pytest.fixture
def sample_md_file(tmp_path):
    """Create a sample markdown file for testing."""
    content = """# Test Document

## Section 1
This is a comprehensive section about machine learning fundamentals.
Machine learning is a subset of artificial intelligence that enables systems
to learn and improve from experience without being explicitly programmed.

## Section 2
Deep learning uses neural networks with multiple layers to analyze data.
These networks can automatically discover representations needed for
feature detection or classification from raw data.
"""
    file_path = tmp_path / "test_doc.md"
    file_path.write_text(content)
    return str(file_path)


class TestDocumentLoader:
    def test_load_text_file(self, sample_text_file):
        loader = DocumentLoader(paths=[sample_text_file])
        docs = loader.load()
        assert len(docs) > 0
        assert docs[0].metadata["file_type"] == "text"
        assert docs[0].metadata["source"] == sample_text_file

    def test_load_markdown_file(self, sample_md_file):
        loader = DocumentLoader(paths=[sample_md_file])
        docs = loader.load()
        assert len(docs) > 0
        assert docs[0].metadata["file_type"] == "markdown"

    def test_load_directory(self, tmp_path, sample_text_file, sample_md_file):
        loader = DocumentLoader(directory=str(tmp_path))
        docs = loader.load()
        assert len(docs) >= 2  # At least the text and md files

    def test_empty_paths(self):
        loader = DocumentLoader(paths=[])
        docs = loader.load()
        assert docs == []


class TestChunker:
    def test_chunk_count(self, sample_text_file):
        loader = DocumentLoader(paths=[sample_text_file])
        docs = loader.load()
        chunker = Chunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk(docs)
        assert len(chunks) > 1  # Should produce multiple chunks

    def test_metadata_keys(self, sample_text_file):
        loader = DocumentLoader(paths=[sample_text_file])
        docs = loader.load()
        chunker = Chunker()
        chunks = chunker.chunk(docs)
        required_keys = {"source", "chunk_index", "chunk_total"}
        for chunk in chunks:
            assert required_keys.issubset(set(chunk.metadata.keys())), \
                f"Missing keys: {required_keys - set(chunk.metadata.keys())}"

    def test_no_short_chunks(self, sample_text_file):
        loader = DocumentLoader(paths=[sample_text_file])
        docs = loader.load()
        chunker = Chunker(min_length=50)
        chunks = chunker.chunk(docs)
        for chunk in chunks:
            assert len(chunk.page_content.strip()) >= 50, \
                f"Chunk too short: {len(chunk.page_content.strip())} chars"

    def test_no_empty_chunks(self, sample_text_file):
        loader = DocumentLoader(paths=[sample_text_file])
        docs = loader.load()
        chunker = Chunker()
        chunks = chunker.chunk(docs)
        for chunk in chunks:
            assert chunk.page_content.strip(), "Empty chunk found"

    def test_chunk_index_continuous(self, sample_text_file):
        loader = DocumentLoader(paths=[sample_text_file])
        docs = loader.load()
        chunker = Chunker()
        chunks = chunker.chunk(docs)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))
