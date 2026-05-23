"""FastAPI serving layer for the Agentic RAG system.

Endpoints:
- POST /query — synchronous query
- GET /health — dependency health checks
- POST /query/stream — SSE streaming
"""

from __future__ import annotations

import os
import time
from typing import Optional

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from agentic_rag.agents.graph import build_graph
from agentic_rag.agents.state import init_state

logger = structlog.get_logger(__name__)

# ── Rate Limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Agentic RAG API",
    description="LangGraph-powered retrieval-augmented generation system",
    version="0.1.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve Frontend ────────────────────────────────────────────────────────────
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
async def root():
    """Serve the frontend interface."""
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Agentic RAG API is running. Frontend not found."}

# ── Compile graph at startup ──────────────────────────────────────────────────
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ── Request/Response Models ───────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    max_iterations: int = 3
    stream: bool = False


class SourceInfo(BaseModel):
    title: Optional[str] = None
    url: Optional[str] = None
    chunk_index: Optional[str] = None


class GradeInfo(BaseModel):
    addresses_query: Optional[bool] = None
    confidence: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    iterations: int
    retrieval_type: str
    grade: GradeInfo


# ── Middleware: Request Logging ────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    logger.info(
        "http_request",
        method=request.method,
        path=str(request.url.path),
        status=response.status_code,
        duration_ms=round(elapsed * 1000),
    )
    return response


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Check health of all dependencies."""
    from agentic_rag.indexer.vector_store import _get_client, _get_collection
    from agentic_rag.indexer.bm25_store import load_bm25_index

    # Check vector store
    try:
        client = _get_client()
        col = _get_collection(client)
        vector_ok = col.count() >= 0
    except Exception:
        vector_ok = False

    # Check BM25
    bm25_ok = load_bm25_index() is not None

    # Check LLM reachability
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from agentic_rag.config import GOOGLE_API_KEY, LLM_MODEL
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )
        resp = llm.invoke("ping")
        llm_ok = bool(resp.content)
    except Exception:
        llm_ok = False

    return {
        "status": "ok" if (vector_ok and bm25_ok and llm_ok) else "degraded",
        "vector_store": vector_ok,
        "bm25": bm25_ok,
        "llm_reachable": llm_ok,
    }


@app.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute")
async def query_endpoint(request: Request, body: QueryRequest):
    """Process a RAG query and return structured response."""
    graph = get_graph()
    state = init_state(body.query)
    state["max_iterations"] = body.max_iterations

    try:
        final_state = graph.invoke(state)
    except Exception as e:
        logger.error("query_failed", error=str(e))
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )

    # Extract sources from final_context
    sources = []
    for doc in final_state.get("final_context", []):
        sources.append(SourceInfo(
            title=doc.metadata.get("title", doc.metadata.get("source", "")),
            url=doc.metadata.get("source", ""),
            chunk_index=str(doc.metadata.get("chunk_index", "")),
        ))

    # Extract grade info
    grade = final_state.get("grade")
    grade_info = GradeInfo()
    if grade:
        grade_info.addresses_query = grade.addresses_query
        grade_info.confidence = grade.confidence

    return QueryResponse(
        answer=final_state.get("answer", "No answer generated"),
        sources=sources,
        iterations=final_state.get("iterations", 0),
        retrieval_type=final_state.get("retrieval_type", "unknown"),
        grade=grade_info,
    )


@app.post("/query/stream")
@limiter.limit("10/minute")
async def stream_endpoint(request: Request, body: QueryRequest):
    """Stream query processing via SSE (Server-Sent Events)."""
    import json

    graph = get_graph()
    state = init_state(body.query)
    state["max_iterations"] = body.max_iterations

    async def event_generator():
        try:
            async for event in graph.astream_events(state, version="v2"):
                kind = event.get("event", "")
                name = event.get("name", "")

                # Stream node start/end events
                if kind == "on_chain_start" and name in [
                    "analyze_query", "rewrite_query", "retrieve",
                    "rerank", "grade_documents", "web_search_fallback",
                    "merge_context", "generate", "grade_answer",
                ]:
                    data = json.dumps({"node": name, "status": "started"})
                    yield f"data: {data}\n\n"

                elif kind == "on_chain_end" and name in [
                    "analyze_query", "rewrite_query", "retrieve",
                    "rerank", "grade_documents", "web_search_fallback",
                    "merge_context", "generate", "grade_answer",
                ]:
                    output = event.get("data", {}).get("output", {})
                    # Only send serializable data
                    safe_output = {}
                    if isinstance(output, dict):
                        for k, v in output.items():
                            if isinstance(v, (str, int, float, bool, type(None))):
                                safe_output[k] = v
                            elif isinstance(v, list):
                                safe_output[k] = f"[{len(v)} items]"

                    data = json.dumps({
                        "node": name,
                        "status": "complete",
                        "data": safe_output,
                    })
                    yield f"data: {data}\n\n"

            yield f"data: {json.dumps({'status': 'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
