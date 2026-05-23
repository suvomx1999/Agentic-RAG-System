# Agentic RAG System 🧠🚀

A production-grade, fully agentic Retrieval-Augmented Generation (RAG) system built with **LangGraph**, **FastAPI**, and **RAGAS**. This system intelligently analyzes queries, decides on retrieval strategies (using hybrid dense+sparse retrieval), scores and reranks documents, handles web search fallbacks, and features internal hallucination checks.

Best of all, this repository is configured to run using **100% Free APIs** (Google Gemini API free tier, BAAI/bge-m3 local embeddings, and DuckDuckGo web search).

---

## ✨ Key Features

- **Agentic Workflow (LangGraph)**: Adaptive query routing, automatic query rewriting (HyDE, Step-back, Multi-query), and self-correction loops.
- **Hybrid Retrieval**: Combines **ChromaDB** (Dense Vector) and **BM25** (Sparse Keyword) using Reciprocal Rank Fusion (RRF).
- **Cross-Encoder Reranking**: Uses `ms-marco-MiniLM-L-6-v2` for precise query-document relevance scoring.
- **LLM-Based Grading**: Gemini-powered document relevance grading, answer quality grading, and hallucination detection.
- **Web Search Fallback**: Automatically searches the web (DuckDuckGo) when the local knowledge base is insufficient.
- **RAGAS Evaluation Framework**: Built-in 6-metric evaluation pipeline for dataset generation and quality assessment.
- **Modern UI**: A premium, glassmorphic frontend out of the box.

---

## 🛠️ Setup Instructions

### 1. Requirements
- Python 3.11+
- A free Google Gemini API Key: Get it at [Google AI Studio](https://aistudio.google.com/)

### 2. Installation
Clone the repository, set up your virtual environment, and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configuration
Copy the environment template and insert your API key:
```bash
cp .env.example .env
# Open .env and replace `your-google-api-key-here` with your actual key
```

### 4. Run the Application
You can use the provided convenience script which will build the vector indexes (if they don't exist) and start the server:
```bash
chmod +x run.sh
./run.sh
```

Alternatively, run the steps manually:
```bash
# Build index from the ./docs folder
python -m agentic_rag.indexer --path ./docs --build-index

# Start FastAPI server
uvicorn agentic_rag.api.server:app --reload --host 0.0.0.0 --port 8000
```

Once running, navigate to **http://localhost:8000** to use the interactive frontend.

---

## 🔬 Evaluation with RAGAS

The system includes a fully automated evaluation pipeline.

1. **Build the Evaluation Dataset**: Generates 60 QA pairs (including unanswerable queries).
   ```bash
   python -c "
   from agentic_rag.evaluator.build_dataset import build_dataset
   from agentic_rag.indexer.ingest import DocumentLoader, Chunker
   docs = DocumentLoader(directory='./docs').load()
   chunks = Chunker().chunk(docs)
   build_dataset(chunks)
   "
   ```

2. **Run Evaluation**:
   ```bash
   python -m agentic_rag.evaluator --subset all --save
   ```

3. **View Report**:
   ```bash
   python -m agentic_rag.evaluator --report agentic_rag/evaluator/results_*.json
   ```

---

## 🏗️ Architecture

The system uses **LangGraph** to manage state across 10 distinct nodes:
1. `analyze_query`
2. `rewrite_query`
3. `retrieve`
4. `rerank`
5. `grade_documents`
6. `web_search_fallback`
7. `merge_context`
8. `generate`
9. `grade_answer`
10. `handle_error`

Conditional edges enforce maximum iteration limits and ensure that ungrounded answers or hallucinations are caught before being returned to the user.
