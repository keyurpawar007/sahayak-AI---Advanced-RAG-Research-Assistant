"""
FastAPI Backend for Advanced RAG System
Wraps existing RAG core modules with REST API endpoints.
Compatible with LangChain 1.2.x.
"""

import os
import sys
import uuid
import shutil
import asyncio
import json
from typing import List, Optional
from contextlib import asynccontextmanager
from collections import deque

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv()

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CHUNK_SIZE, CHUNK_OVERLAP, MODEL_NAME, EMBEDDING_MODEL, CHROMA_PATH, BM25_PATH,
    LLM_PROVIDER, GROQ_API_KEY, GROQ_MODEL, OLLAMA_MODEL
)
from core.storage import PersistentStorage
from core.query_classifier import QueryClassifier
from core.greeting_handler import GreetingHandler
from core.enhanced_retriever import EnhancedRetriever
from core.rag_chain import EnhancedRAGChain
from models.configs import RetrievalConfig
from utils.document_loader import DocumentLoader
from utils.logging_config import setup_logging
from firebase_config import storage_backend

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.documents import Document

logger = setup_logging()


# --- Simple Chat Memory (replaces deprecated ConversationBufferWindowMemory) ---
class SimpleChatMemory:
    """Lightweight chat memory that stores last K messages."""

    def __init__(self, k=10):
        self.messages = deque(maxlen=k * 2)  # k pairs of user/ai messages

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_ai_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def get_history_string(self) -> str:
        if not self.messages:
            return ""
        return "\n".join(f"{m['role']}: {m['content']}" for m in self.messages)

    @property
    def chat_memory(self):
        """Compatibility shim so existing code that accesses .chat_memory.messages works."""
        return self

    @property
    def messages_list(self):
        return list(self.messages)


# --- Global State -------------------------------------------------------
rag_state = {
    "embeddings": None,
    "llm": None,
    "vector_store": None,
    "rag_chain": None,
    "memory": None,
    "initialized": False,
    "documents_loaded": False,
}

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# --- Pydantic Models ---------------------------------------------------
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    query_type: str
    sources: List[str]
    session_id: str


class SessionCreate(BaseModel):
    title: Optional[str] = "New Chat"


# --- Lifespan ----------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG components on startup."""
    print("[STARTUP] Initializing RAG system...")

    try:
        storage = PersistentStorage(CHROMA_PATH, BM25_PATH)
        
        print(f"[STARTUP] Loading embeddings model: {EMBEDDING_MODEL}...")
        rag_state["embeddings"] = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        print("[STARTUP] Embeddings model loaded")

        if LLM_PROVIDER == "groq":
            print(f"[STARTUP] Initializing Groq LLM: {GROQ_MODEL}...")
            rag_state["llm"] = ChatGroq(
                model=GROQ_MODEL,
                api_key=GROQ_API_KEY,
                temperature=0.7,
                streaming=True,
            )
            print("[STARTUP] Groq LLM initialized")
        else:
            print(f"[STARTUP] Initializing Ollama LLM: {OLLAMA_MODEL}...")
            rag_state["llm"] = ChatOllama(model=OLLAMA_MODEL)
            print("[STARTUP] Ollama LLM initialized")
        
        rag_state["memory"] = SimpleChatMemory(k=10)

        # Load existing embeddings if available
        if storage.storage_exists():
            print("[STARTUP] Loading existing embeddings from ChromaDB...")
            _load_existing_embeddings()
            print("[STARTUP] Loaded existing embeddings from ChromaDB")
        else:
            print("[STARTUP] No existing embeddings found. Upload PDFs to get started.")

        rag_state["initialized"] = True
        print("[STARTUP] RAG system initialized successfully")

    except Exception as e:
        import traceback
        print(f"[STARTUP] Failed to initialize RAG system: {e}")
        traceback.print_exc()
        rag_state["initialized"] = False
        rag_state["error_detail"] = str(e)

    yield

    print("[STARTUP] Shutting down RAG system")


def _load_existing_embeddings():
    """Load existing ChromaDB embeddings and build retrievers."""
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=rag_state["embeddings"],
    )
    rag_state["vector_store"] = vector_store

    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 12})

    # Build BM25 from existing docs
    data = vector_store.get()
    existing_docs = data.get("documents", [])
    existing_metas = data.get("metadatas", [])
    if not existing_metas or len(existing_metas) != len(existing_docs):
        existing_metas = [{"source": "existing"} for _ in existing_docs]
        
    documents = [
        Document(page_content=doc, metadata=meta or {})
        for doc, meta in zip(existing_docs, existing_metas)
    ] if existing_docs else [Document(page_content="placeholder", metadata={"source": "init"})]
    bm25_retriever = BM25Retriever.from_documents(documents)

    # Build RAG chain
    config = RetrievalConfig()
    classifier = QueryClassifier(rag_state["llm"])
    greeting_handler = GreetingHandler(rag_state["llm"])
    enhanced_retriever = EnhancedRetriever(
        vector_retriever, bm25_retriever, rag_state["llm"], config
    )
    rag_state["rag_chain"] = EnhancedRAGChain(
        enhanced_retriever, classifier, greeting_handler, rag_state["memory"], rag_state["llm"]
    )
    rag_state["documents_loaded"] = True


def _rebuild_rag_chain(documents: List[Document]):
    """Rebuild RAG chain after new documents are uploaded."""
    vector_store = Chroma.from_documents(
        documents, rag_state["embeddings"], persist_directory=CHROMA_PATH
    )
    rag_state["vector_store"] = vector_store

    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 12})
    bm25_retriever = BM25Retriever.from_documents(documents)

    config = RetrievalConfig()
    classifier = QueryClassifier(rag_state["llm"])
    greeting_handler = GreetingHandler(rag_state["llm"])
    enhanced_retriever = EnhancedRetriever(
        vector_retriever, bm25_retriever, rag_state["llm"], config
    )
    rag_state["rag_chain"] = EnhancedRAGChain(
        enhanced_retriever, classifier, greeting_handler, rag_state["memory"], rag_state["llm"]
    )
    rag_state["documents_loaded"] = True


# --- App ----------------------------------------------------------------
app = FastAPI(
    title="Advanced RAG System API",
    description="AI-powered document analysis with semantic search and intelligent retrieval",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static Files -------------------------------------------------------
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
app.mount("/src", StaticFiles(directory=os.path.join(FRONTEND_DIR, "src")), name="src")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend page."""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# --- Health -------------------------------------------------------------
@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "initialized": rag_state["initialized"],
        "documents_loaded": rag_state["documents_loaded"],
        "error": rag_state.get("error_detail"),
    }


# --- Upload -------------------------------------------------------------
@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process PDF documents."""
    if not rag_state["initialized"]:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    saved_paths = []
    total_chunks = 0
    uploaded_docs = []

    try:
        for file in files:
            if not file.filename.lower().endswith(".pdf"):
                continue

            # Save uploaded file
            file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            saved_paths.append(file_path)

            # Record document metadata
            doc_record = storage_backend.add_document(
                filename=file.filename,
                num_chunks=0,
                file_size=len(content),
            )
            uploaded_docs.append(doc_record)

        if not saved_paths:
            raise HTTPException(status_code=400, detail="No valid PDF files found")

        # Process documents
        documents = await DocumentLoader.load_pdf_documents(
            saved_paths, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        total_chunks = len(documents)
        
        # Replace the ugly server paths with the real filenames in metadata
        path_to_filename = {path: file.filename for path, file in zip(saved_paths, files)}
        for doc in documents:
            if "source" in doc.metadata:
                doc.metadata["source"] = path_to_filename.get(doc.metadata["source"], doc.metadata["source"])

        # Rebuild RAG chain with new documents
        if rag_state["documents_loaded"] and rag_state["vector_store"]:
            # Add to existing store
            rag_state["vector_store"].add_documents(documents)

            # Rebuild retrievers
            data = rag_state["vector_store"].get()
            all_docs_text = data.get("documents", [])
            all_docs_metas = data.get("metadatas", [])
            if not all_docs_metas or len(all_docs_metas) != len(all_docs_text):
                all_docs_metas = [{"source": "uploaded"} for _ in all_docs_text]
                
            all_documents = [
                Document(page_content=doc, metadata=meta or {})
                for doc, meta in zip(all_docs_text, all_docs_metas)
            ]
            bm25_retriever = BM25Retriever.from_documents(all_documents)
            vector_retriever = rag_state["vector_store"].as_retriever(search_kwargs={"k": 12})

            config = RetrievalConfig()
            classifier = QueryClassifier(rag_state["llm"])
            greeting_handler = GreetingHandler(rag_state["llm"])
            enhanced_retriever = EnhancedRetriever(
                vector_retriever, bm25_retriever, rag_state["llm"], config
            )
            rag_state["rag_chain"] = EnhancedRAGChain(
                enhanced_retriever, classifier, greeting_handler,
                rag_state["memory"], rag_state["llm"]
            )
        else:
            _rebuild_rag_chain(documents)

        return {
            "message": f"Successfully processed {len(saved_paths)} PDF(s)",
            "total_chunks": total_chunks,
            "documents": uploaded_docs,
        }

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Chat ---------------------------------------------------------------
@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Send a query and get an AI-generated streaming response."""
    if not rag_state["initialized"]:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    session_id = request.session_id
    if not session_id:
        session = storage_backend.create_session(title=request.query[:50])
        session_id = session["id"]
    else:
        session = storage_backend.get_session(session_id)
        if not session:
            session = storage_backend.create_session(title=request.query[:50])
            session_id = session["id"]

    # Save user message
    storage_backend.add_message(session_id, "user", request.query)

    async def event_generator():
        print(f"[CHAT] Starting stream for session {session_id}: {request.query}")
        yield f"data: {json.dumps({'type': 'start'})}\n\n"

        # If no RAG chain is available (no documents uploaded), return a friendly message
        if not rag_state["rag_chain"]:
            no_doc_msg = "👋 I'm ready to help! Please **upload a PDF document** first using the upload button. Once you upload documents, I can answer questions about them with full AI-powered analysis."
            yield f"data: {json.dumps({'type': 'metadata', 'query_type': 'GREETING', 'sources': []})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'content': no_doc_msg})}\n\n"
            storage_backend.add_message(session_id, "assistant", no_doc_msg, [])
            yield 'data: {"type": "done"}\n\n'
            return
        
        memory = rag_state["memory"]
        history_context = memory.get_history_string() if memory else ""
        
        # Inject available documents so the AI knows what's out there
        uploaded_docs = storage_backend.get_documents()
        doc_names = ", ".join([d.get("filename", "Unknown") for d in uploaded_docs]) if uploaded_docs else "None"
        
        context = f"Available Documents System Knowledge: {doc_names}\n\nPast Chat Context:\n{history_context}"
        full_answer = []
        token_count = 0
        metadata = {}

        try:
            print("[CHAT] Calling astream_query...")
            async for chunk in rag_state["rag_chain"].astream_query(request.query, context):
                chunk_type = chunk.get("type")
                if chunk_type == "metadata":
                    metadata = chunk
                    print(f"[CHAT] Yielding metadata (sources: {len(chunk.get('sources', []))})")
                    yield f"data: {json.dumps(chunk)}\n\n"
                elif chunk_type == "token":
                    token_count += 1
                    content = chunk["content"]
                    full_answer.append(content)
                    if token_count % 10 == 0:
                        print(f"[CHAT] Yielding token {token_count}...")
                    yield f"data: {json.dumps(chunk)}\n\n"
                elif chunk_type == "step":
                    # Internal step hints — skip (not sent to client)
                    pass
            
            print(f"[CHAT] Stream finished. Total tokens: {token_count}")
            # After streaming finish, finalize
            answer_text = "".join(full_answer)
            source_texts = [s["content"] for s in metadata.get("sources", [])]
            
            # Save assistant message to backend
            storage_backend.add_message(session_id, "assistant", answer_text, source_texts)
            
            # Update memory
            if memory:
                memory.add_user_message(request.query)
                memory.add_ai_message(answer_text)
                
            yield "data: {\"type\": \"done\"}\n\n"

        except Exception as e:
            print(f"[CHAT] STREAMING ERROR: {str(e)}")
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# --- Documents ----------------------------------------------------------
@app.get("/api/documents")
async def list_documents():
    """List all uploaded documents."""
    return {"documents": storage_backend.get_documents()}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document record."""
    storage_backend.delete_document(doc_id)
    return {"message": "Document deleted"}


# --- Sessions -----------------------------------------------------------
@app.get("/api/sessions")
async def list_sessions():
    """List all chat sessions."""
    return {"sessions": storage_backend.get_sessions()}


@app.post("/api/sessions")
async def create_session(request: SessionCreate):
    """Create a new chat session."""
    session = storage_backend.create_session(title=request.title)
    return session


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get a session with its chat history."""
    session = storage_backend.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session."""
    storage_backend.delete_session(session_id)
    return {"message": "Session deleted"}


# --- Clear Storage ------------------------------------------------------
@app.delete("/api/clear-storage")
async def clear_storage():
    """Clear all embeddings, documents, sessions, uploaded files and reset RAG state."""
    try:
        errors = []

        # 1. Use ChromaDB's own API to delete all collections (releases file locks on Windows)
        #    then wipe the directories safely.
        try:
            import chromadb
            chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
            for col in chroma_client.list_collections():
                chroma_client.delete_collection(col.name)
            del chroma_client  # release client / file handles
        except Exception as e:
            errors.append(f"ChromaDB collection delete failed: {e}")

        # Now wipe the storage directories (locks released above)
        try:
            for path in [CHROMA_PATH, BM25_PATH]:
                if os.path.exists(path):
                    shutil.rmtree(path, ignore_errors=True)
                os.makedirs(path, exist_ok=True)
        except Exception as e:
            errors.append(f"Storage directory wipe failed: {e}")

        # 2. Reset in-memory RAG state
        rag_state["vector_store"] = None
        rag_state["rag_chain"] = None
        rag_state["documents_loaded"] = False
        rag_state["memory"] = SimpleChatMemory(k=10)

        # 3 & 4. Clear all document records + sessions from storage backend in one shot
        try:
            storage_backend.clear_all()
        except Exception as e:
            errors.append(f"Metadata clear failed: {e}")

        # 5. Delete all uploaded PDF files from uploads/ directory
        try:
            if os.path.exists(UPLOAD_DIR):
                for filename in os.listdir(UPLOAD_DIR):
                    file_path = os.path.join(UPLOAD_DIR, filename)
                    if os.path.isfile(file_path):
                        try:
                            os.remove(file_path)
                        except Exception:
                            pass  # skip files in use
        except Exception as e:
            errors.append(f"Upload files clear failed: {e}")

        if errors:
            return {"message": "Storage partially cleared", "warnings": errors}

        return {"message": "All data cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Run ----------------------------------------------------------------
if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 10000))

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port
    )