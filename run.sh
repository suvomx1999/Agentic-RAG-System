#!/bin/bash
set -e

echo "🚀 Starting Agentic RAG system..."

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Please copy .env.example to .env and add your GOOGLE_API_KEY."
    exit 1
fi

# Activate venv
source venv/bin/activate

# Check if index exists, build if not
if [ ! -d "chroma_db" ] || [ ! -d "bm25_store" ]; then
    echo "🔨 Indexes not found. Building indexes from ./docs..."
    python -m agentic_rag.indexer --path ./docs --build-index
else
    echo "✅ Indexes found."
fi

echo "🌐 Starting FastAPI Server..."
uvicorn agentic_rag.api.server:app --reload --host 0.0.0.0 --port 8000
