"""Document ingestion and chunking for the Agentic RAG system.

Supports PDF, .txt, and .md files. Chunks are produced via
RecursiveCharacterTextSplitter with metadata enrichment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from agentic_rag.config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LENGTH


# ── Document Loader ───────────────────────────────────────────────────────────

class DocumentLoader:
    """Load documents from files or a directory.

    Supported formats: .pdf, .txt, .md
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

    def __init__(self, paths: Optional[List[str]] = None, directory: Optional[str] = None):
        self.file_paths: List[Path] = []
        if paths:
            self.file_paths.extend(Path(p) for p in paths)
        if directory:
            dir_path = Path(directory)
            if dir_path.is_dir():
                for ext in self.SUPPORTED_EXTENSIONS:
                    self.file_paths.extend(dir_path.rglob(f"*{ext}"))

    def load(self) -> List[Document]:
        """Load all documents and return as LangChain Document objects."""
        all_docs: List[Document] = []
        for file_path in self.file_paths:
            ext = file_path.suffix.lower()
            if ext not in self.SUPPORTED_EXTENSIONS:
                continue
            try:
                docs = self._load_file(file_path)
                all_docs.extend(docs)
            except Exception as e:
                print(f"⚠️  Failed to load {file_path}: {e}")
        return all_docs

    def _load_file(self, file_path: Path) -> List[Document]:
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            return self._load_pdf(file_path)
        else:
            return self._load_text(file_path, ext)

    def _load_pdf(self, file_path: Path) -> List[Document]:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()
        for i, doc in enumerate(pages):
            doc.metadata.update({
                "source": str(file_path),
                "page": i,
                "file_type": "pdf",
            })
        return pages

    def _load_text(self, file_path: Path, ext: str) -> List[Document]:
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()
        file_type = "markdown" if ext == ".md" else "text"
        for doc in docs:
            doc.metadata.update({
                "source": str(file_path),
                "page": 0,
                "file_type": file_type,
            })
        return docs


# ── Chunker ───────────────────────────────────────────────────────────────────

class Chunker:
    """Split documents into chunks with metadata enrichment.

    Chunks shorter than MIN_CHUNK_LENGTH are discarded.
    Each chunk gets chunk_index and chunk_total in metadata.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        min_length: int = MIN_CHUNK_LENGTH,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.min_length = min_length

    def chunk(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks, enrich metadata, filter short chunks."""
        all_chunks: List[Document] = []

        for doc in documents:
            raw_chunks = self.splitter.split_documents([doc])
            # Filter short chunks
            valid_chunks = [
                c for c in raw_chunks
                if len(c.page_content.strip()) >= self.min_length
            ]
            # Enrich metadata with chunk index and total
            total = len(valid_chunks)
            for idx, chunk in enumerate(valid_chunks):
                chunk.metadata["chunk_index"] = idx
                chunk.metadata["chunk_total"] = total
            all_chunks.extend(valid_chunks)

        return all_chunks
