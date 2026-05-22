# Sahayak AI — Chat with your Documents

Sahayak AI is a RAG (Retrieval-Augmented Generation) based document assistant. You upload PDFs, ask questions, and it gives you accurate answers sourced directly from your documents — not from the model's general knowledge.

The name *Sahayak* means **helper/assistant** in Hindi, which is exactly what this project is — a personal document helper powered by local or cloud LLMs.

---

## Why I built this

I was tired of manually searching through long PDFs and reports. I wanted something that could read a document and let me just *talk* to it. So I built Sahayak AI — it supports both local models (via Ollama) and fast cloud inference (via Groq), stores your chat history, and handles different types of questions differently.

---

## What it can do

- Upload one or multiple PDFs and start asking questions immediately
- Handles factual, analytical, opinion-based, and contextual questions differently
- Hybrid search — combines vector similarity (ChromaDB) + keyword search (BM25) for better retrieval
- Streaming responses so you see the answer as it's being generated
- Full chat history with session management
- Storage works with Firebase Firestore **or** plain local JSON — no setup required to get started
- Clean web UI served directly from the FastAPI backend

---

## Tech stack

| Layer | What's used |
|---|---|
| Backend | FastAPI + Uvicorn |
| LLM (cloud) | Groq — `llama-3.3-70b-versatile` |
| LLM (local) | Ollama — `llama3.2` |
| Embeddings | `sentence-transformers/all-MiniLM-l6-v2` via HuggingFace |
| Vector DB | ChromaDB |
| Keyword search | BM25 (rank-bm25) |
| PDF parsing | pymupdf4llm |
| Orchestration | LangChain |
| Storage | Firebase Firestore or local JSON fallback |

---

## Project structure

```
Sahayak AI/
├── app.py                  # FastAPI app — all API endpoints
├── main.py                 # Entry point
├── config.py               # Config constants (chunk size, models, paths)
├── firebase_config.py      # Storage backend (Firebase or local JSON)
├── core/
│   ├── rag_chain.py        # Main RAG pipeline with streaming
│   ├── enhanced_retriever.py  # Hybrid vector + BM25 retriever
│   ├── query_classifier.py    # Classifies query type
│   ├── greeting_handler.py    # Handles greetings naturally
│   └── storage.py          # ChromaDB persistence helper
├── models/
│   └── configs.py          # Pydantic config models
├── utils/
│   ├── document_loader.py  # PDF loading and chunking
│   └── logging_config.py   # Logging setup
├── frontend/               # Web UI (served via FastAPI)
├── requirements.txt
└── .env.example
```

---

## Getting started

### 1. Clone and set up

```bash
git clone https://github.com/your-username/sahayak-ai.git
cd sahayak-ai

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

```env
# Use Groq for fast cloud inference (free tier available at console.groq.com)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# OR use Ollama locally — pull the model first:
# ollama pull llama3.2
OLLAMA_MODEL=llama3.2

# Firebase is optional — leave blank to use local JSON storage
FIREBASE_CREDENTIALS_PATH=
FIREBASE_PROJECT_ID=
```

If `GROQ_API_KEY` is set, Groq is used. If it's blank, it falls back to Ollama automatically.

### 3. Run

```bash
python app.py
```

Open `http://localhost:8000` in your browser. Upload a PDF and start chatting.

---

## How the retrieval works

When you ask a question, it goes through a few steps:

1. **Query classification** — the LLM first decides what kind of question this is (factual, analytical, opinion, contextual, or greeting). Each type has a slightly different retrieval strategy.
2. **Hybrid retrieval** — relevant chunks are fetched using both vector search (semantic similarity) and BM25 (keyword overlap). The results are merged.
3. **Response generation** — the LLM generates an answer grounded in the retrieved chunks and streams it back token by token.
4. **Memory** — the last 10 conversation turns are kept in memory so follow-up questions work naturally.

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serves the web UI |
| GET | `/api/health` | Check if the system is ready |
| POST | `/api/upload` | Upload one or more PDFs |
| POST | `/api/chat` | Send a query, get a streaming response |
| GET | `/api/documents` | List uploaded documents |
| DELETE | `/api/documents/{id}` | Remove a document record |
| GET | `/api/sessions` | List all chat sessions |
| POST | `/api/sessions` | Create a new session |
| GET | `/api/sessions/{id}` | Get a session with full chat history |
| DELETE | `/api/sessions/{id}` | Delete a session |
| DELETE | `/api/clear-storage` | Wipe all data and reset |

---

## Configuration

All tunable parameters live in `config.py` and can be overridden via `.env`:

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 1500 | Characters per text chunk |
| `CHUNK_OVERLAP` | 300 | Overlap between chunks |
| `EMBEDDING_MODEL` | all-MiniLM-l6-v2 | HuggingFace embedding model |
| `GROQ_MODEL` | llama-3.3-70b-versatile | Cloud LLM |
| `OLLAMA_MODEL` | llama3.2 | Local LLM fallback |

---

## Firebase (optional)

By default the app stores everything in local JSON files under `storage/app_data/`. If you want persistent cloud storage across restarts or deployments, you can connect Firebase Firestore:

1. Create a project on [Firebase Console](https://console.firebase.google.com)
2. Generate a service account credentials JSON file
3. Set `FIREBASE_CREDENTIALS_PATH` and `FIREBASE_PROJECT_ID` in `.env`

The app will automatically switch to Firestore — no code changes needed.

---

## Requirements

- Python 3.10+
- If using Ollama: [install Ollama](https://ollama.com) and pull `llama3.2`
- If using Groq: a free API key from [console.groq.com](https://console.groq.com)

---

## License

MIT