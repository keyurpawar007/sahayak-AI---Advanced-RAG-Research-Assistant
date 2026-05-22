"""
Firebase/Local Storage Configuration
Falls back to local JSON storage if Firebase is not configured.
"""

import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional

# Try Firebase import
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False


class StorageBackend:
    """Abstract storage backend for chat history and document metadata."""

    def __init__(self):
        self.use_firebase = False
        self.db = None
        self._init_storage()

    def _init_storage(self):
        """Initialize storage - Firebase if configured, else local JSON."""
        creds_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "")
        project_id = os.getenv("FIREBASE_PROJECT_ID", "")

        if FIREBASE_AVAILABLE and creds_path and os.path.exists(creds_path):
            try:
                cred = credentials.Certificate(creds_path)
                firebase_admin.initialize_app(cred, {"projectId": project_id})
                self.db = firestore.client()
                self.use_firebase = True
                print("[STORAGE] Firebase Firestore initialized successfully")
                return
            except Exception as e:
                print(f"[STORAGE] Firebase init failed: {e}. Falling back to local storage.")

        # Fallback: local JSON storage
        self.local_storage_path = os.path.join(os.getcwd(), "storage", "app_data")
        os.makedirs(self.local_storage_path, exist_ok=True)
        self.sessions_file = os.path.join(self.local_storage_path, "sessions.json")
        self.documents_file = os.path.join(self.local_storage_path, "documents.json")
        self._ensure_local_files()
        print("[STORAGE] Using local JSON storage")

    def _ensure_local_files(self):
        """Create local JSON files if they don't exist."""
        for filepath in [self.sessions_file, self.documents_file]:
            if not os.path.exists(filepath):
                with open(filepath, "w") as f:
                    json.dump([], f)

    def _read_json(self, filepath: str) -> list:
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _write_json(self, filepath: str, data: list):
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

    # ---- Sessions ----

    def create_session(self, title: str = "New Chat") -> Dict:
        """Create a new chat session."""
        session = {
            "id": str(uuid.uuid4()),
            "title": title,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": []
        }

        if self.use_firebase:
            self.db.collection("sessions").document(session["id"]).set(session)
        else:
            sessions = self._read_json(self.sessions_file)
            sessions.append(session)
            self._write_json(self.sessions_file, sessions)

        return session

    def get_sessions(self) -> List[Dict]:
        """Get all chat sessions."""
        if self.use_firebase:
            docs = self.db.collection("sessions").order_by(
                "updated_at", direction=firestore.Query.DESCENDING
            ).stream()
            return [doc.to_dict() for doc in docs]
        else:
            sessions = self._read_json(self.sessions_file)
            return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get a specific session by ID."""
        if self.use_firebase:
            doc = self.db.collection("sessions").document(session_id).get()
            return doc.to_dict() if doc.exists else None
        else:
            sessions = self._read_json(self.sessions_file)
            for s in sessions:
                if s["id"] == session_id:
                    return s
            return None

    def add_message(self, session_id: str, role: str, content: str, sources: List[str] = None):
        """Add a message to a session."""
        message = {
            "id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "sources": sources or [],
            "timestamp": datetime.now().isoformat()
        }

        if self.use_firebase:
            ref = self.db.collection("sessions").document(session_id)
            ref.update({
                "messages": firestore.ArrayUnion([message]),
                "updated_at": datetime.now().isoformat()
            })
        else:
            sessions = self._read_json(self.sessions_file)
            for s in sessions:
                if s["id"] == session_id:
                    s["messages"].append(message)
                    s["updated_at"] = datetime.now().isoformat()
                    break
            self._write_json(self.sessions_file, sessions)

        return message

    def update_session_title(self, session_id: str, title: str):
        """Update session title."""
        if self.use_firebase:
            self.db.collection("sessions").document(session_id).update({"title": title})
        else:
            sessions = self._read_json(self.sessions_file)
            for s in sessions:
                if s["id"] == session_id:
                    s["title"] = title
                    break
            self._write_json(self.sessions_file, sessions)

    def delete_session(self, session_id: str):
        """Delete a chat session."""
        if self.use_firebase:
            self.db.collection("sessions").document(session_id).delete()
        else:
            sessions = self._read_json(self.sessions_file)
            sessions = [s for s in sessions if s["id"] != session_id]
            self._write_json(self.sessions_file, sessions)

    # ---- Documents ----

    def add_document(self, filename: str, num_chunks: int, file_size: int) -> Dict:
        """Record an uploaded document."""
        doc = {
            "id": str(uuid.uuid4()),
            "filename": filename,
            "num_chunks": num_chunks,
            "file_size": file_size,
            "uploaded_at": datetime.now().isoformat()
        }

        if self.use_firebase:
            self.db.collection("documents").document(doc["id"]).set(doc)
        else:
            docs = self._read_json(self.documents_file)
            docs.append(doc)
            self._write_json(self.documents_file, docs)

        return doc

    def get_documents(self) -> List[Dict]:
        """Get all uploaded documents."""
        if self.use_firebase:
            docs = self.db.collection("documents").order_by(
                "uploaded_at", direction=firestore.Query.DESCENDING
            ).stream()
            return [doc.to_dict() for doc in docs]
        else:
            docs = self._read_json(self.documents_file)
            return sorted(docs, key=lambda x: x.get("uploaded_at", ""), reverse=True)

    def delete_document(self, doc_id: str):
        """Delete a document record."""
        if self.use_firebase:
            self.db.collection("documents").document(doc_id).delete()
        else:
            docs = self._read_json(self.documents_file)
            docs = [d for d in docs if d["id"] != doc_id]
            self._write_json(self.documents_file, docs)

    def clear_all(self):
        """Clear all sessions and documents from storage."""
        if self.use_firebase:
            # Delete all documents in 'sessions' collection
            for doc in self.db.collection("sessions").stream():
                doc.reference.delete()
            # Delete all documents in 'documents' collection
            for doc in self.db.collection("documents").stream():
                doc.reference.delete()
        else:
            self._write_json(self.sessions_file, [])
            self._write_json(self.documents_file, [])


# Singleton instance
storage_backend = StorageBackend()
