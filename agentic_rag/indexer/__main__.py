"""CLI entry point for the indexer module.

Usage:
    python -m agentic_rag.indexer --path ./docs --preview 3
"""

from __future__ import annotations

import argparse
import json

from agentic_rag.indexer.ingest import DocumentLoader, Chunker


def main():
    parser = argparse.ArgumentParser(description="Ingest and chunk documents")
    parser.add_argument("--path", required=True, help="File or directory path")
    parser.add_argument("--preview", type=int, default=0, help="Number of chunks to preview")
    parser.add_argument("--build-index", action="store_true", help="Build vector + BM25 indexes")
    args = parser.parse_args()

    # Load documents
    loader = DocumentLoader(directory=args.path)
    docs = loader.load()
    print(f"📄 Loaded {len(docs)} document(s)")

    # Chunk
    chunker = Chunker()
    chunks = chunker.chunk(docs)
    print(f"🔪 Created {len(chunks)} chunk(s)")

    # Preview
    if args.preview > 0:
        print(f"\n── Preview (first {args.preview} chunks) ──")
        for i, chunk in enumerate(chunks[:args.preview]):
            print(f"\n--- Chunk {i} ---")
            print(f"Content: {chunk.page_content[:200]}...")
            print(f"Metadata: {json.dumps(chunk.metadata, indent=2, default=str)}")

    # Build indexes
    if args.build_index:
        from agentic_rag.indexer.vector_store import build_index
        from agentic_rag.indexer.bm25_store import build_bm25_index
        print("\n🔨 Building vector index...")
        build_index(chunks)
        print("🔨 Building BM25 index...")
        build_bm25_index(chunks)
        print("✅ Indexes built successfully!")


if __name__ == "__main__":
    main()
