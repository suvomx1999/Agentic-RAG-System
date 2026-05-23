"""Centralized configuration — loads from .env with sensible defaults."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env", override=True)

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# ── Paths ─────────────────────────────────────────────────────────────────────
CHROMA_PATH: str = os.getenv("CHROMA_PATH", str(_project_root / "chroma_db"))
BM25_PATH: str = os.getenv("BM25_PATH", str(_project_root / "bm25_store"))

# ── Models ────────────────────────────────────────────────────────────────────
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
EVAL_LLM_MODEL: str = os.getenv("EVAL_LLM_MODEL", "llama-3.1-8b-instant")
RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
MIN_CHUNK_LENGTH: int = int(os.getenv("MIN_CHUNK_LENGTH", "50"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K: int = int(os.getenv("TOP_K", "10"))
RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", "5"))
MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "3"))

# ── Validation ────────────────────────────────────────────────────────────────
def validate_config() -> dict[str, bool]:
    """Check that critical configuration values are set. Returns status dict."""
    checks = {
        "groq_api_key": bool(GROQ_API_KEY),
        "chroma_path_writable": os.access(os.path.dirname(CHROMA_PATH) or ".", os.W_OK),
        "embed_model_set": bool(EMBED_MODEL),
        "llm_model_set": bool(LLM_MODEL),
    }
    return checks


if __name__ == "__main__":
    status = validate_config()
    for key, ok in status.items():
        icon = "✅" if ok else "❌"
        print(f"  {icon} {key}")
