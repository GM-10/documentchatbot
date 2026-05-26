import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM Configuration ────────────────────────────────────────────────────────────
LLM_BACKEND  = os.getenv("LLM_BACKEND",  "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# ── Embedding Configuration ───────────────────────────────────────────────────────
EMBED_MODEL  = os.getenv("EMBED_MODEL",  "all-MiniLM-L6-v2")

# ── RAG Parameters ────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 100

# ── Storage Configuration ─────────────────────────────────────────────────────────
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "./data/vector_db")
