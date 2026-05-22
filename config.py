import os

# Constants
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 300))

# ─── LLM Provider ─────────────────────────────────────────
# Priority: Groq → Ollama (local fallback)
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")   # free on groq.com

# Ollama fallback (used when GROQ_API_KEY is blank)
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "llama3.2")

# Auto-detect provider
LLM_PROVIDER  = "groq" if GROQ_API_KEY else "ollama"
MODEL_NAME    = GROQ_MODEL if LLM_PROVIDER == "groq" else OLLAMA_MODEL

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-l6-v2")

# ─── Storage Paths ────────────────────────────────────────
BASE_STORAGE_PATH = os.path.join(os.getcwd(), "storage")
CHROMA_PATH = os.path.join(BASE_STORAGE_PATH, "chroma")
BM25_PATH   = os.path.join(BASE_STORAGE_PATH, "bm25")

# ─── Retrieval ────────────────────────────────────────────
DEFAULT_VECTOR_K    = 6
DEFAULT_BM25_K      = 4
DEFAULT_SUB_QUERIES = 2
DEFAULT_VIEWPOINTS  = 2