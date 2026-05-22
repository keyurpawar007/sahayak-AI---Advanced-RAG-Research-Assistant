/**
 * API Client for RAG Backend
 */

const API_BASE = '/api';

async function request(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;
    const config = {
        headers: { 'Content-Type': 'application/json' },
        ...options,
    };

    // Remove Content-Type for FormData (file uploads)
    if (options.body instanceof FormData) {
        delete config.headers['Content-Type'];
    }

    const response = await fetch(url, config);

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(error.detail || 'Request failed');
    }

    return response.json();
}

// ─── Health ──────────────────────────────────────────────
export function healthCheck() {
    return request('/health');
}

// ─── Upload ──────────────────────────────────────────────
export function uploadDocuments(files) {
    const formData = new FormData();
    for (const file of files) {
        formData.append('files', file);
    }
    return request('/upload', {
        method: 'POST',
        body: formData,
    });
}

// --- Chat ------------------------------------------------
/**
 * Send a message and handle streaming response
 * @param {string} query 
 * @param {string} sessionId 
 * @param {Function} onChunk Call with text tokens
 * @param {Function} onMetadata Call with sources and query info
 * @param {Function} onStep Call with background status strings
 */
export async function sendMessage(query, sessionId, { onChunk, onMetadata, onStep, onError }) {
    try {
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, session_id: sessionId }),
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(error.detail || 'Request failed');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep last incomplete line

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const dataStr = line.slice(6).trim();
                    if (!dataStr) continue;

                    try {
                        const data = JSON.parse(dataStr);
                        if (data.type === 'token') onChunk(data.content);
                        if (data.type === 'metadata') onMetadata(data);
                        if (data.type === 'step') onStep(data.content);
                        if (data.type === 'error') onError(data.content);
                    } catch (e) {
                        console.warn('Failed to parse SSE chunk:', dataStr);
                    }
                }
            }
        }
    } catch (err) {
        if (onError) onError(err.message);
        throw err;
    }
}

// ─── Documents ───────────────────────────────────────────
export function getDocuments() {
    return request('/documents');
}

export function deleteDocument(docId) {
    return request(`/documents/${docId}`, { method: 'DELETE' });
}

// ─── Sessions ────────────────────────────────────────────
export function getSessions() {
    return request('/sessions');
}

export function createSession(title = 'New Chat') {
    return request('/sessions', {
        method: 'POST',
        body: JSON.stringify({ title }),
    });
}

export function getSession(sessionId) {
    return request(`/sessions/${sessionId}`);
}

export function deleteSession(sessionId) {
    return request(`/sessions/${sessionId}`, { method: 'DELETE' });
}

// ─── Clear Storage ───────────────────────────────────────
export function clearStorage() {
    return request('/clear-storage', { method: 'DELETE' });
}
