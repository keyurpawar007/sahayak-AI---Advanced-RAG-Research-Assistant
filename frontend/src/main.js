/**
 * RAG AI — Main Application Logic
 */

import { marked } from 'https://cdn.jsdelivr.net/npm/marked/lib/marked.esm.js';
import * as api from './api.js';

// ─── State ───────────────────────────────────────────────
const state = {
    currentSessionId: null,
    sessions: [],
    documents: [],
    selectedFiles: [],
    isProcessing: false,
};

// ─── DOM References ──────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const els = {
    sidebar: $('#sidebar'),
    sidebarToggle: $('#sidebar-toggle'),
    newChatBtn: $('#new-chat-btn'),
    sessionsList: $('#sessions-list'),
    documentsList: $('#documents-list'),
    chatArea: $('#chat-area'),
    welcomeScreen: $('#welcome-screen'),
    messagesContainer: $('#messages-container'),
    chatInput: $('#chat-input'),
    sendBtn: $('#send-btn'),
    uploadBtn: $('#upload-btn'),
    uploadModal: $('#upload-modal'),
    closeModalBtn: $('#close-modal-btn'),
    dropZone: $('#drop-zone'),
    fileInput: $('#file-input'),
    fileList: $('#file-list'),
    uploadSubmitBtn: $('#upload-submit-btn'),
    uploadProgress: $('#upload-progress'),
    progressFill: $('#progress-fill'),
    progressText: $('#progress-text'),
    clearStorageBtn: $('#clear-storage-btn'),
    statusBadge: $('#status-badge'),
    currentSessionTitle: $('#current-session-title'),
};

// ─── Initialize ──────────────────────────────────────────
async function init() {
    setupEventListeners();
    setupMarked();
    createToastContainer();

    // Check backend health
    try {
        const health = await api.healthCheck();
        if (health.initialized) {
            setStatus('Ready', 'ready');
        } else {
            setStatus('Initializing...', 'loading');
        }

        if (health.documents_loaded) {
            showToast('Documents loaded from previous session', 'info');
        }
    } catch (err) {
        setStatus('Backend offline', 'error');
        showToast('Cannot reach backend. Make sure the API server is running.', 'error');
    }

    // Load sessions and documents
    await Promise.all([loadSessions(), loadDocuments()]);
}

function setupMarked() {
    marked.setOptions({
        breaks: true,
        gfm: true,
    });
}

// ─── Event Listeners ─────────────────────────────────────
function setupEventListeners() {
    // Sidebar toggle (mobile)
    els.sidebarToggle.addEventListener('click', () => {
        els.sidebar.classList.toggle('open');
    });

    // Close sidebar on outside click (mobile)
    els.chatArea.addEventListener('click', () => {
        els.sidebar.classList.remove('open');
    });

    // New chat
    els.newChatBtn.addEventListener('click', startNewChat);

    // Chat input
    els.chatInput.addEventListener('input', onInputChange);
    els.chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });

    // Send button
    els.sendBtn.addEventListener('click', handleSendMessage);

    // Upload button
    els.uploadBtn.addEventListener('click', openUploadModal);

    // Quick action cards
    $$('.quick-action-card').forEach((card) => {
        card.addEventListener('click', () => {
            const action = card.dataset.action;
            if (action === 'upload') openUploadModal();
            if (action === 'ask') els.chatInput.focus();
            if (action === 'analyze') {
                els.chatInput.value = 'Provide a comprehensive analysis of ';
                els.chatInput.focus();
                onInputChange();
            }
        });
    });

    // Modal
    els.closeModalBtn.addEventListener('click', closeUploadModal);
    els.uploadModal.addEventListener('click', (e) => {
        if (e.target === els.uploadModal) closeUploadModal();
    });

    // Drag & drop
    els.dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        els.dropZone.classList.add('drag-over');
    });
    els.dropZone.addEventListener('dragleave', () => {
        els.dropZone.classList.remove('drag-over');
    });
    els.dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        els.dropZone.classList.remove('drag-over');
        handleFileSelection(e.dataTransfer.files);
    });
    els.dropZone.addEventListener('click', () => els.fileInput.click());

    // File input
    els.fileInput.addEventListener('change', (e) => {
        handleFileSelection(e.target.files);
    });

    // Upload submit
    els.uploadSubmitBtn.addEventListener('click', handleUpload);

    // Clear storage
    els.clearStorageBtn.addEventListener('click', handleClearStorage);
}

// ─── Input Handling ──────────────────────────────────────
function onInputChange() {
    const input = els.chatInput;
    // Auto-resize
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 120) + 'px';
    // Enable/disable send
    els.sendBtn.disabled = !input.value.trim() || state.isProcessing;
}

// --- Chat ------------------------------------------------
async function handleSendMessage() {
    const query = els.chatInput.value.trim();
    if (!query || state.isProcessing) return;

    state.isProcessing = true;
    els.sendBtn.disabled = true;
    els.chatInput.value = '';
    els.chatInput.style.height = 'auto';

    // Hide welcome, show messages
    showChatView();

    // Add user message
    appendMessage('user', query);

    // Placeholder for assistant message
    const msgEl = appendStreamingMessage();
    const contentEl = msgEl.querySelector('.message-content');
    const senderEl = msgEl.querySelector('.message-sender');
    let fullContent = '';

    try {
        setStatus('Processing...', 'loading');

        await api.sendMessage(query, state.currentSessionId, {
            onChunk: (token) => {
                fullContent += token;
                contentEl.innerHTML = marked.parse(fullContent);
                scrollToBottom();
            },
            onMetadata: async (data) => {
                if (data.session_id && data.session_id !== state.currentSessionId) {
                    state.currentSessionId = data.session_id;
                    await loadSessions();
                }
                if (data.sources && data.sources.length > 0) {
                    renderSources(msgEl, data.sources);
                }
                if (data.query_type) {
                    const badge = document.createElement('span');
                    badge.className = 'query-type-badge';
                    badge.textContent = data.query_type;
                    contentEl.before(badge);
                }
            },
            onStep: (stepContent) => {
                senderEl.textContent = `RAG AI — ${stepContent}`;
            },
            onError: (err) => {
                contentEl.innerHTML = `<span class="error-text">❌ Error: ${err}</span>`;
            }
        });

        senderEl.textContent = 'RAG AI';
        setStatus('Ready', 'ready');
    } catch (err) {
        setStatus('Error', 'error');
        showToast(err.message, 'error');
    }

    state.isProcessing = false;
    onInputChange();
}

function appendStreamingMessage() {
    const msgEl = document.createElement('div');
    msgEl.className = 'message assistant';
    msgEl.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="message-body">
            <div class="message-sender">RAG AI</div>
            <div class="message-content"><div class="thinking-dots"><span></span><span></span><span></span></div></div>
        </div>
    `;
    els.messagesContainer.appendChild(msgEl);
    scrollToBottom();
    return msgEl;
}

function renderSources(msgEl, sources) {
    const bodyEl = msgEl.querySelector('.message-body');
    const sourcesContainer = document.createElement('div');
    sourcesContainer.className = 'sources-container';

    const chips = sources.map((s, i) =>
        `<span class="source-chip" data-idx="${i}" title="Click to expand">${truncate(s.content || s, 80)}</span>`
    ).join('');

    sourcesContainer.innerHTML = `
        <div class="sources-title">📎 Sources (${sources.length})</div>
        ${chips}
        <div class="source-expanded"></div>
    `;

    const detailEl = sourcesContainer.querySelector('.source-expanded');
    sourcesContainer.querySelectorAll('.source-chip').forEach((chip) => {
        chip.addEventListener('click', () => {
            const idx = parseInt(chip.dataset.idx);
            const content = sources[idx].content || sources[idx];
            if (detailEl.classList.contains('show') && detailEl.dataset.activeIdx === String(idx)) {
                detailEl.classList.remove('show');
            } else {
                detailEl.textContent = content;
                detailEl.dataset.activeIdx = String(idx);
                detailEl.classList.add('show');
            }
        });
    });

    bodyEl.appendChild(sourcesContainer);
    scrollToBottom();
}

function showChatView() {
    els.welcomeScreen.style.display = 'none';
    els.messagesContainer.style.display = 'flex';
}

function showWelcomeView() {
    els.welcomeScreen.style.display = 'flex';
    els.messagesContainer.style.display = 'none';
    els.messagesContainer.innerHTML = '';
}

function appendMessage(role, content, queryType = null, sources = []) {
    const msgEl = document.createElement('div');
    msgEl.className = `message ${role}`;

    const avatar = role === 'user' ? 'Y' : 'AI';
    const sender = role === 'user' ? 'You' : 'RAG AI';

    let sourcesHtml = '';
    if (sources && sources.length > 0) {
        const chips = sources.map((s, i) =>
            `<span class="source-chip" data-idx="${i}" title="Click to expand">${truncate(s, 80)}</span>`
        ).join('');
        sourcesHtml = `
      <div class="sources-container">
        <div class="sources-title">📎 Sources (${sources.length})</div>
        ${chips}
        <div class="source-expanded" id="source-detail-${Date.now()}"></div>
      </div>
    `;
    }

    let queryBadge = '';
    if (queryType && queryType !== 'error') {
        queryBadge = `<span class="query-type-badge">${queryType}</span>`;
    }

    const renderedContent = role === 'assistant' ? marked.parse(content) : escapeHtml(content);

    msgEl.innerHTML = `
    <div class="message-avatar">${avatar}</div>
    <div class="message-body">
      <div class="message-sender">${sender}</div>
      ${queryBadge}
      <div class="message-content">${renderedContent}</div>
      ${sourcesHtml}
    </div>
  `;

    // Add source chip click handlers
    if (sources && sources.length > 0) {
        const detailEl = msgEl.querySelector('.source-expanded');
        msgEl.querySelectorAll('.source-chip').forEach((chip) => {
            chip.addEventListener('click', () => {
                const idx = parseInt(chip.dataset.idx);
                if (detailEl.classList.contains('show') && detailEl.dataset.activeIdx === String(idx)) {
                    detailEl.classList.remove('show');
                } else {
                    detailEl.textContent = sources[idx];
                    detailEl.dataset.activeIdx = String(idx);
                    detailEl.classList.add('show');
                }
            });
        });
    }

    els.messagesContainer.appendChild(msgEl);
    scrollToBottom();
}

function appendThinking() {
    const el = document.createElement('div');
    el.className = 'message assistant';
    el.innerHTML = `
    <div class="message-avatar">AI</div>
    <div class="message-body">
      <div class="message-sender">RAG AI</div>
      <div class="message-content">
        <div class="thinking-dots">
          <span></span><span></span><span></span>
        </div>
      </div>
    </div>
  `;
    els.messagesContainer.appendChild(el);
    scrollToBottom();
    return el;
}

function scrollToBottom() {
    els.chatArea.scrollTop = els.chatArea.scrollHeight;
}

// ─── Sessions ────────────────────────────────────────────
async function loadSessions() {
    try {
        const data = await api.getSessions();
        state.sessions = data.sessions || [];
        renderSessions();
    } catch (err) {
        console.error('Failed to load sessions:', err);
    }
}

function renderSessions() {
    if (state.sessions.length === 0) {
        els.sessionsList.innerHTML = '<div class="empty-state">No conversations yet</div>';
        return;
    }

    els.sessionsList.innerHTML = state.sessions.map((s) => `
    <div class="session-item ${s.id === state.currentSessionId ? 'active' : ''}" data-id="${s.id}">
      <span class="session-item-title">${escapeHtml(s.title)}</span>
      <button class="delete-btn" data-id="${s.id}" title="Delete">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
      </button>
    </div>
  `).join('');

    // Click handlers
    els.sessionsList.querySelectorAll('.session-item').forEach((el) => {
        el.addEventListener('click', (e) => {
            if (e.target.closest('.delete-btn')) return;
            loadSession(el.dataset.id);
        });
    });

    els.sessionsList.querySelectorAll('.delete-btn').forEach((btn) => {
        btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            await handleDeleteSession(btn.dataset.id);
        });
    });
}

async function loadSession(sessionId) {
    try {
        const session = await api.getSession(sessionId);
        state.currentSessionId = sessionId;
        els.currentSessionTitle.textContent = session.title;

        // Clear and rebuild messages
        els.messagesContainer.innerHTML = '';

        if (session.messages && session.messages.length > 0) {
            showChatView();
            for (const msg of session.messages) {
                appendMessage(msg.role, msg.content, null, msg.sources);
            }
        } else {
            showWelcomeView();
        }

        renderSessions();
        els.sidebar.classList.remove('open');
    } catch (err) {
        showToast('Failed to load conversation', 'error');
    }
}

async function startNewChat() {
    state.currentSessionId = null;
    els.currentSessionTitle.textContent = 'New Chat';
    showWelcomeView();
    renderSessions();
    els.sidebar.classList.remove('open');
}

async function handleDeleteSession(sessionId) {
    try {
        await api.deleteSession(sessionId);
        if (state.currentSessionId === sessionId) {
            startNewChat();
        }
        await loadSessions();
        showToast('Conversation deleted', 'success');
    } catch (err) {
        showToast('Failed to delete conversation', 'error');
    }
}

// ─── Documents ───────────────────────────────────────────
async function loadDocuments() {
    try {
        const data = await api.getDocuments();
        state.documents = data.documents || [];
        renderDocuments();
    } catch (err) {
        console.error('Failed to load documents:', err);
    }
}

function renderDocuments() {
    if (state.documents.length === 0) {
        els.documentsList.innerHTML = '<div class="empty-state">No documents uploaded</div>';
        return;
    }

    els.documentsList.innerHTML = state.documents.map((d) => `
    <div class="document-item" data-id="${d.id}">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="min-width:14px; color: var(--accent-primary); margin-right: 8px;">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>
      </svg>
      <span class="document-item-name">${escapeHtml(d.filename)}</span>
      <span class="document-item-size">${formatSize(d.file_size)}</span>
      <button class="delete-btn" data-id="${d.id}" title="Delete">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
      </button>
    </div>
  `).join('');

    els.documentsList.querySelectorAll('.delete-btn').forEach((btn) => {
        btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            try {
                await api.deleteDocument(btn.dataset.id);
                await loadDocuments();
                showToast('Document removed', 'success');
            } catch (err) {
                showToast('Failed to delete document', 'error');
            }
        });
    });
}

// ─── Upload ──────────────────────────────────────────────
function openUploadModal() {
    state.selectedFiles = [];
    els.fileList.innerHTML = '';
    els.uploadSubmitBtn.disabled = true;
    els.uploadProgress.hidden = true;
    els.uploadModal.hidden = false;
}

function closeUploadModal() {
    els.uploadModal.hidden = true;
    state.selectedFiles = [];
}

function handleFileSelection(fileListObj) {
    const files = Array.from(fileListObj).filter((f) => f.name.toLowerCase().endsWith('.pdf'));
    if (files.length === 0) {
        showToast('Please select PDF files', 'error');
        return;
    }

    state.selectedFiles = [...state.selectedFiles, ...files];
    renderFileList();
}

function renderFileList() {
    els.fileList.innerHTML = state.selectedFiles.map((f, i) => `
    <div class="file-list-item">
      <span class="file-name">📄 ${escapeHtml(f.name)}</span>
      <span class="file-size">${formatSize(f.size)}</span>
      <button class="remove-file-btn" data-idx="${i}">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
      </button>
    </div>
  `).join('');

    els.uploadSubmitBtn.disabled = state.selectedFiles.length === 0;

    els.fileList.querySelectorAll('.remove-file-btn').forEach((btn) => {
        btn.addEventListener('click', () => {
            state.selectedFiles.splice(parseInt(btn.dataset.idx), 1);
            renderFileList();
        });
    });
}

async function handleUpload() {
    if (state.selectedFiles.length === 0) return;

    els.uploadSubmitBtn.disabled = true;
    els.uploadProgress.hidden = false;
    els.progressFill.style.width = '10%';
    els.progressText.textContent = 'Uploading files...';

    try {
        // Simulate progress
        const progressInterval = setInterval(() => {
            const current = parseFloat(els.progressFill.style.width);
            if (current < 85) {
                els.progressFill.style.width = (current + Math.random() * 8) + '%';
            }
        }, 500);

        els.progressText.textContent = 'Processing & embedding documents...';

        const result = await api.uploadDocuments(state.selectedFiles);

        clearInterval(progressInterval);
        els.progressFill.style.width = '100%';
        els.progressText.textContent = `✅ ${result.message} (${result.total_chunks} chunks)`;

        showToast(`${result.message}`, 'success');

        // Refresh documents list
        await loadDocuments();

        // Close modal after a brief delay
        setTimeout(() => {
            closeUploadModal();
        }, 1500);

    } catch (err) {
        els.progressFill.style.width = '0%';
        els.progressText.textContent = `❌ Error: ${err.message}`;
        showToast(`Upload failed: ${err.message}`, 'error');
        els.uploadSubmitBtn.disabled = false;
    }
}

// ─── Clear Storage ───────────────────────────────────────
async function handleClearStorage() {
    if (!confirm('Are you sure you want to clear all data? This will delete all uploaded documents and embeddings.')) {
        return;
    }

    try {
        await api.clearStorage();
        showToast('All data cleared', 'success');
        await Promise.all([loadSessions(), loadDocuments()]);
        startNewChat();
        setStatus('Ready', 'ready');
    } catch (err) {
        showToast(`Failed to clear data: ${err.message}`, 'error');
    }
}

// ─── Status Badge ────────────────────────────────────────
function setStatus(text, type) {
    els.statusBadge.textContent = text;
    els.statusBadge.className = 'status-badge';
    if (type === 'loading') els.statusBadge.classList.add('loading');
    if (type === 'error') els.statusBadge.classList.add('error');
}

// ─── Toast Notifications ─────────────────────────────────
function createToastContainer() {
    if ($('.toast-container')) return;
    const container = document.createElement('div');
    container.className = 'toast-container';
    document.body.appendChild(container);
}

function showToast(message, type = 'info') {
    const container = $('.toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(20px)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ─── Utilities ───────────────────────────────────────────
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function truncate(text, maxLen) {
    return text.length > maxLen ? text.slice(0, maxLen) + '...' : text;
}

function formatSize(bytes) {
    if (!bytes) return '';
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// ─── Boot ────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', init);
