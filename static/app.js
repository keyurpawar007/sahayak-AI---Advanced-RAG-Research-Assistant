/**
 * RAG AI — Main Application Logic
 * Self-contained vanilla JS, no build tools needed.
 * Uses the global `marked` from CDN.
 */

(function () {
    'use strict';

    // ─── API Client ──────────────────────────────────────
    const API_BASE = '/api';

    async function apiRequest(endpoint, options = {}) {
        const url = API_BASE + endpoint;
        const config = { headers: { 'Content-Type': 'application/json' }, ...options };
        if (options.body instanceof FormData) delete config.headers['Content-Type'];
        const res = await fetch(url, config);
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || 'Request failed');
        }
        return res.json();
    }

    const api = {
        health: () => apiRequest('/health'),
        upload: (files) => { const fd = new FormData(); files.forEach(f => fd.append('files', f)); return apiRequest('/upload', { method: 'POST', body: fd }); },
        chat: (query, sid) => apiRequest('/chat', { method: 'POST', body: JSON.stringify({ query, session_id: sid }) }),
        getDocuments: () => apiRequest('/documents'),
        deleteDocument: (id) => apiRequest('/documents/' + id, { method: 'DELETE' }),
        getSessions: () => apiRequest('/sessions'),
        createSession: (title) => apiRequest('/sessions', { method: 'POST', body: JSON.stringify({ title: title || 'New Chat' }) }),
        getSession: (id) => apiRequest('/sessions/' + id),
        deleteSession: (id) => apiRequest('/sessions/' + id, { method: 'DELETE' }),
        clearStorage: () => apiRequest('/clear-storage', { method: 'DELETE' }),
    };

    // ─── State ───────────────────────────────────────────
    const state = {
        currentSessionId: null,
        sessions: [],
        documents: [],
        selectedFiles: [],
        isProcessing: false,
    };

    // ─── DOM ─────────────────────────────────────────────
    const $ = (s) => document.querySelector(s);
    const $$ = (s) => document.querySelectorAll(s);

    const el = {};

    function cacheDom() {
        el.sidebar = $('#sidebar');
        el.sidebarToggle = $('#sidebar-toggle');
        el.newChatBtn = $('#new-chat-btn');
        el.sessionsList = $('#sessions-list');
        el.documentsList = $('#documents-list');
        el.chatArea = $('#chat-area');
        el.welcomeScreen = $('#welcome-screen');
        el.messagesContainer = $('#messages-container');
        el.chatInput = $('#chat-input');
        el.sendBtn = $('#send-btn');
        el.uploadBtn = $('#upload-btn');
        el.uploadModal = $('#upload-modal');
        el.closeModalBtn = $('#close-modal-btn');
        el.dropZone = $('#drop-zone');
        el.fileInput = $('#file-input');
        el.fileList = $('#file-list');
        el.uploadSubmitBtn = $('#upload-submit-btn');
        el.uploadProgress = $('#upload-progress');
        el.progressFill = $('#progress-fill');
        el.progressText = $('#progress-text');
        el.clearStorageBtn = $('#clear-storage-btn');
        el.statusBadge = $('#status-badge');
        el.currentSessionTitle = $('#current-session-title');
    }

    // ─── Init ────────────────────────────────────────────
    async function init() {
        cacheDom();
        marked.setOptions({ breaks: true, gfm: true });
        bindEvents();

        try {
            const h = await api.health();
            setStatus(h.initialized ? 'Ready' : 'Initializing...', h.initialized ? 'ready' : 'loading');
            if (h.documents_loaded) toast('Documents loaded from previous session', 'info');
        } catch (_) {
            setStatus('Backend offline', 'error');
            toast('Cannot reach backend. Start the server with: python app.py', 'error');
        }

        await Promise.all([loadSessions(), loadDocuments()]);
    }

    // ─── Events ──────────────────────────────────────────
    function bindEvents() {
        el.sidebarToggle.addEventListener('click', () => el.sidebar.classList.toggle('open'));
        el.chatArea.addEventListener('click', () => el.sidebar.classList.remove('open'));
        el.newChatBtn.addEventListener('click', startNewChat);

        el.chatInput.addEventListener('input', onInput);
        el.chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
        });
        el.sendBtn.addEventListener('click', sendMessage);

        el.uploadBtn.addEventListener('click', openUpload);

        $$('.quick-action-card').forEach((c) => c.addEventListener('click', () => {
            const a = c.dataset.action;
            if (a === 'upload') openUpload();
            else if (a === 'ask') el.chatInput.focus();
            else if (a === 'analyze') { el.chatInput.value = 'Provide a comprehensive analysis of '; el.chatInput.focus(); onInput(); }
        }));

        el.closeModalBtn.addEventListener('click', closeUpload);
        el.uploadModal.addEventListener('click', (e) => { if (e.target === el.uploadModal) closeUpload(); });

        el.dropZone.addEventListener('dragover', (e) => { e.preventDefault(); el.dropZone.classList.add('drag-over'); });
        el.dropZone.addEventListener('dragleave', () => el.dropZone.classList.remove('drag-over'));
        el.dropZone.addEventListener('drop', (e) => { e.preventDefault(); el.dropZone.classList.remove('drag-over'); pickFiles(e.dataTransfer.files); });
        el.dropZone.addEventListener('click', () => el.fileInput.click());
        el.fileInput.addEventListener('change', (e) => pickFiles(e.target.files));

        el.uploadSubmitBtn.addEventListener('click', submitUpload);
        el.clearStorageBtn.addEventListener('click', clearAll);
    }

    // ─── Input ───────────────────────────────────────────
    function onInput() {
        const t = el.chatInput;
        t.style.height = 'auto';
        t.style.height = Math.min(t.scrollHeight, 120) + 'px';
        el.sendBtn.disabled = !t.value.trim() || state.isProcessing;
    }

    // ─── Chat ────────────────────────────────────────────
    async function sendMessage() {
        const q = el.chatInput.value.trim();
        if (!q || state.isProcessing) return;

        state.isProcessing = true;
        el.sendBtn.disabled = true;
        el.chatInput.value = '';
        el.chatInput.style.height = 'auto';

        showChat();
        addMsg('user', q);
        const thinkingDots = addThinking();

        try {
            setStatus('Thinking...', 'loading');

            const response = await fetch(API_BASE + '/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: q, session_id: state.currentSessionId })
            });

            if (!response.ok) throw new Error('Streaming request failed');

            thinkingDots.remove();

            // Create a placeholder message for the streaming content
            const aiMsgEl = createStreamMsgEl();
            el.messagesContainer.appendChild(aiMsgEl);
            const contentEl = aiMsgEl.querySelector('.message-content');

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let fullAnswer = '';
            let metadataReceived = false;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const jsonStr = line.slice(6);
                        try {
                            const data = JSON.parse(jsonStr);
                            if (data.type === 'metadata' && !metadataReceived) {
                                metadataReceived = true;
                                if (data.query_type) {
                                    const badge = document.createElement('span');
                                    badge.className = 'query-type-badge';
                                    badge.textContent = data.query_type;
                                    aiMsgEl.querySelector('.message-sender').after(badge);
                                }
                                if (data.sources && data.sources.length) {
                                    renderSources(aiMsgEl, data.sources);
                                }
                            } else if (data.type === 'token') {
                                fullAnswer += data.content;
                                contentEl.innerHTML = marked.parse(fullAnswer);
                                scrollEnd();
                            } else if (data.type === 'done') {
                                // Finalize
                            }
                        } catch (e) {
                            console.warn('Failed to parse SSE chunk', e);
                        }
                    }
                }
            }

            setStatus('Ready', 'ready');
            // Refresh sessions if it was a new chat
            if (!state.currentSessionId) {
                await loadSessions();
                // Find the newest session and set it
                if (state.sessions.length > 0) state.currentSessionId = state.sessions[0].id;
            }

        } catch (err) {
            thinkingDots.remove();
            addMsg('assistant', '❌ Error: ' + err.message, 'error');
            setStatus('Error', 'error');
            toast(err.message, 'error');
        }

        state.isProcessing = false;
        onInput();
    }

    function createStreamMsgEl() {
        const d = document.createElement('div');
        d.className = 'message assistant';
        d.innerHTML =
            '<div class="message-avatar">AI</div>' +
            '<div class="message-body">' +
            '<div class="message-sender">RAG AI</div>' +
            '<div class="message-content"></div>' +
            '</div>';
        return d;
    }

    function renderSources(msgEl, sources) {
        if (!sources || !sources.length) return;
        const body = msgEl.querySelector('.message-body');
        const container = document.createElement('div');
        container.className = 'sources-container';
        const title = document.createElement('div');
        title.className = 'sources-title';
        title.textContent = '📎 Sources (' + sources.length + ')';
        container.appendChild(title);

        const uid = Date.now();
        const expanded = document.createElement('div');
        expanded.className = 'source-expanded';
        expanded.id = 'sd-' + uid;

        sources.forEach((s, i) => {
            const chip = document.createElement('span');
            chip.className = 'source-chip';
            chip.textContent = trunc(s.content, 80);
            chip.addEventListener('click', () => {
                if (expanded.classList.contains('show') && expanded.dataset.i === '' + i) {
                    expanded.classList.remove('show');
                } else {
                    expanded.textContent = s.content;
                    expanded.dataset.i = '' + i;
                    expanded.classList.add('show');
                }
            });
            container.appendChild(chip);
        });

        container.appendChild(expanded);
        body.appendChild(container);
    }

    function showChat() {
        el.welcomeScreen.style.display = 'none';
        el.messagesContainer.style.display = 'flex';
    }

    function showWelcome() {
        el.welcomeScreen.style.display = 'flex';
        el.messagesContainer.style.display = 'none';
        el.messagesContainer.innerHTML = '';
    }

    function addMsg(role, content, qtype, sources) {
        const d = document.createElement('div');
        d.className = 'message ' + role;

        const avatar = role === 'user' ? 'Y' : 'AI';
        const sender = role === 'user' ? 'You' : 'RAG AI';
        const rendered = role === 'assistant' ? marked.parse(content) : esc(content);
        const badge = qtype && qtype !== 'error' ? '<span class="query-type-badge">' + qtype + '</span>' : '';

        let srcHtml = '';
        if (sources && sources.length) {
            const uid = Date.now();
            const chips = sources.map((s, i) => '<span class="source-chip" data-i="' + i + '">' + esc(trunc(s, 80)) + '</span>').join('');
            srcHtml = '<div class="sources-container"><div class="sources-title">📎 Sources (' + sources.length + ')</div>' + chips + '<div class="source-expanded" id="sd-' + uid + '"></div></div>';
        }

        d.innerHTML =
            '<div class="message-avatar">' + avatar + '</div>' +
            '<div class="message-body">' +
            '<div class="message-sender">' + sender + '</div>' +
            badge +
            '<div class="message-content">' + rendered + '</div>' +
            srcHtml +
            '</div>';

        if (sources && sources.length) {
            const det = d.querySelector('.source-expanded');
            d.querySelectorAll('.source-chip').forEach((c) => {
                c.addEventListener('click', () => {
                    const i = parseInt(c.dataset.i);
                    if (det.classList.contains('show') && det.dataset.ai === '' + i) det.classList.remove('show');
                    else { det.textContent = sources[i]; det.dataset.ai = '' + i; det.classList.add('show'); }
                });
            });
        }

        el.messagesContainer.appendChild(d);
        scrollEnd();
    }

    function addThinking() {
        const d = document.createElement('div');
        d.className = 'message assistant';
        d.innerHTML =
            '<div class="message-avatar">AI</div>' +
            '<div class="message-body"><div class="message-sender">RAG AI</div>' +
            '<div class="message-content"><div class="thinking-dots"><span></span><span></span><span></span></div></div></div>';
        el.messagesContainer.appendChild(d);
        scrollEnd();
        return d;
    }

    function scrollEnd() { el.chatArea.scrollTop = el.chatArea.scrollHeight; }

    // ─── Sessions ────────────────────────────────────────
    async function loadSessions() {
        try {
            const d = await api.getSessions();
            state.sessions = d.sessions || [];
            renderSessions();
        } catch (_) { }
    }

    function renderSessions() {
        if (!state.sessions.length) { el.sessionsList.innerHTML = '<div class="empty-state">No conversations yet</div>'; return; }
        el.sessionsList.innerHTML = state.sessions.map((s) =>
            '<div class="session-item ' + (s.id === state.currentSessionId ? 'active' : '') + '" data-id="' + s.id + '">' +
            '<span class="session-item-title">' + esc(s.title) + '</span>' +
            '<button class="delete-btn" data-id="' + s.id + '" title="Delete"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg></button>' +
            '</div>'
        ).join('');

        el.sessionsList.querySelectorAll('.session-item').forEach((e) => {
            e.addEventListener('click', (ev) => { if (!ev.target.closest('.delete-btn')) loadSession(e.dataset.id); });
        });
        el.sessionsList.querySelectorAll('.delete-btn').forEach((b) => {
            b.addEventListener('click', (ev) => { ev.stopPropagation(); delSession(b.dataset.id); });
        });
    }

    async function loadSession(id) {
        try {
            const s = await api.getSession(id);
            state.currentSessionId = id;
            el.currentSessionTitle.textContent = s.title;
            el.messagesContainer.innerHTML = '';
            if (s.messages && s.messages.length) {
                showChat();
                s.messages.forEach((m) => addMsg(m.role, m.content, null, m.sources));
            } else showWelcome();
            renderSessions();
            el.sidebar.classList.remove('open');
        } catch (_) { toast('Failed to load conversation', 'error'); }
    }

    function startNewChat() {
        state.currentSessionId = null;
        el.currentSessionTitle.textContent = 'New Chat';
        showWelcome();
        renderSessions();
        el.sidebar.classList.remove('open');
    }

    async function delSession(id) {
        try {
            await api.deleteSession(id);
            if (state.currentSessionId === id) startNewChat();
            await loadSessions();
            toast('Conversation deleted', 'success');
        } catch (_) { toast('Failed to delete', 'error'); }
    }

    // ─── Documents ───────────────────────────────────────
    async function loadDocuments() {
        try {
            const d = await api.getDocuments();
            state.documents = d.documents || [];
            renderDocuments();
        } catch (_) { }
    }

    function renderDocuments() {
        if (!state.documents.length) { el.documentsList.innerHTML = '<div class="empty-state">No documents uploaded</div>'; return; }
        el.documentsList.innerHTML = state.documents.map((d) =>
            '<div class="document-item" data-id="' + d.id + '">' +
            '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="min-width:14px;color:var(--accent-primary);margin-right:8px"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>' +
            '<span class="document-item-name">' + esc(d.filename) + '</span>' +
            '<span class="document-item-size">' + fmtSize(d.file_size) + '</span>' +
            '<button class="delete-btn" data-id="' + d.id + '" title="Delete"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg></button>' +
            '</div>'
        ).join('');

        el.documentsList.querySelectorAll('.delete-btn').forEach((b) => {
            b.addEventListener('click', async (e) => {
                e.stopPropagation();
                try { await api.deleteDocument(b.dataset.id); await loadDocuments(); toast('Document removed', 'success'); }
                catch (_) { toast('Failed to delete document', 'error'); }
            });
        });
    }

    // ─── Upload ──────────────────────────────────────────
    function openUpload() {
        state.selectedFiles = [];
        el.fileList.innerHTML = '';
        el.uploadSubmitBtn.disabled = true;
        el.uploadProgress.hidden = true;
        el.uploadModal.hidden = false;
    }

    function closeUpload() { el.uploadModal.hidden = true; state.selectedFiles = []; }

    function pickFiles(fl) {
        const files = Array.from(fl).filter((f) => f.name.toLowerCase().endsWith('.pdf'));
        if (!files.length) { toast('Please select PDF files', 'error'); return; }
        state.selectedFiles.push(...files);
        renderFiles();
    }

    function renderFiles() {
        el.fileList.innerHTML = state.selectedFiles.map((f, i) =>
            '<div class="file-list-item">' +
            '<span class="file-name">📄 ' + esc(f.name) + '</span>' +
            '<span class="file-size">' + fmtSize(f.size) + '</span>' +
            '<button class="remove-file-btn" data-i="' + i + '"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg></button>' +
            '</div>'
        ).join('');
        el.uploadSubmitBtn.disabled = !state.selectedFiles.length;
        el.fileList.querySelectorAll('.remove-file-btn').forEach((b) => {
            b.addEventListener('click', () => { state.selectedFiles.splice(parseInt(b.dataset.i), 1); renderFiles(); });
        });
    }

    async function submitUpload() {
        if (!state.selectedFiles.length) return;
        el.uploadSubmitBtn.disabled = true;
        el.uploadProgress.hidden = false;
        el.progressFill.style.width = '10%';
        el.progressText.textContent = 'Uploading files...';

        const iv = setInterval(() => {
            const w = parseFloat(el.progressFill.style.width);
            if (w < 85) el.progressFill.style.width = (w + Math.random() * 8) + '%';
        }, 500);

        try {
            el.progressText.textContent = 'Processing & embedding documents...';
            const r = await api.upload(state.selectedFiles);
            clearInterval(iv);
            el.progressFill.style.width = '100%';
            el.progressText.textContent = '✅ ' + r.message + ' (' + r.total_chunks + ' chunks)';
            toast(r.message, 'success');
            await loadDocuments();
            setTimeout(closeUpload, 1500);
        } catch (err) {
            clearInterval(iv);
            el.progressFill.style.width = '0%';
            el.progressText.textContent = '❌ Error: ' + err.message;
            toast('Upload failed: ' + err.message, 'error');
            el.uploadSubmitBtn.disabled = false;
        }
    }

    // ─── Clear ───────────────────────────────────────────
    async function clearAll() {
        if (!confirm('Clear ALL data? This deletes all documents and embeddings.')) return;
        try {
            await api.clearStorage();
            toast('All data cleared', 'success');
            await Promise.all([loadSessions(), loadDocuments()]);
            startNewChat();
            setStatus('Ready', 'ready');
        } catch (err) { toast('Failed: ' + err.message, 'error'); }
    }

    // ─── Status ──────────────────────────────────────────
    function setStatus(t, type) {
        el.statusBadge.textContent = t;
        el.statusBadge.className = 'status-badge' + (type === 'loading' ? ' loading' : type === 'error' ? ' error' : '');
    }

    // ─── Toast ───────────────────────────────────────────
    function toast(msg, type) {
        const c = $('.toast-container');
        const t = document.createElement('div');
        t.className = 'toast ' + (type || 'info');
        t.textContent = msg;
        c.appendChild(t);
        setTimeout(() => { t.style.opacity = '0'; t.style.transform = 'translateX(20px)'; t.style.transition = 'all .3s ease'; setTimeout(() => t.remove(), 300); }, 4000);
    }

    // ─── Utils ───────────────────────────────────────────
    function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
    function trunc(s, n) { return s.length > n ? s.slice(0, n) + '...' : s; }
    function fmtSize(b) { if (!b) return ''; if (b < 1024) return b + ' B'; if (b < 1048576) return (b / 1024).toFixed(1) + ' KB'; return (b / 1048576).toFixed(1) + ' MB'; }

    // ─── Boot ────────────────────────────────────────────
    document.addEventListener('DOMContentLoaded', init);
})();
