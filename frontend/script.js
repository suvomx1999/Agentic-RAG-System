document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const queryInput = document.getElementById('query-input');
    const chatHistory = document.getElementById('chat-history');
    const sendBtn = document.getElementById('send-btn');
    const traceContainer = document.getElementById('trace-container');
    const sourcesContainer = document.getElementById('sources-container');
    const maxIterations = document.getElementById('max-iterations');
    const streamToggle = document.getElementById('stream-toggle');
    const vectorStatus = document.getElementById('vector-status');

    // Auto-resize textarea
    queryInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if (this.value.trim() === '') {
            sendBtn.disabled = true;
        } else {
            sendBtn.disabled = false;
        }
    });

    // Handle Enter key (Shift+Enter for new line)
    queryInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (this.value.trim() !== '') {
                chatForm.dispatchEvent(new Event('submit'));
            }
        }
    });

    // Check backend health
    async function checkHealth() {
        try {
            const res = await fetch('/health');
            const data = await res.json();
            const dot = vectorStatus.querySelector('.pulse-dot');
            const span = vectorStatus.querySelector('span');
            
            if (data.status === 'ok') {
                dot.className = 'pulse-dot green';
                span.textContent = 'System Ready';
            } else {
                dot.className = 'pulse-dot yellow';
                span.textContent = 'System Degraded';
            }
        } catch (err) {
            const dot = vectorStatus.querySelector('.pulse-dot');
            const span = vectorStatus.querySelector('span');
            dot.className = 'pulse-dot red';
            span.textContent = 'Server Offline';
        }
    }
    checkHealth();
    setInterval(checkHealth, 30000); // Check every 30s

    // Add message to chat
    function appendMessage(role, content, isMarkdown = false) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}-message`;
        
        let avatar = role === 'user' ? '👤' : '✨';
        
        // Parse markdown if needed
        let htmlContent = content;
        if (isMarkdown && role === 'assistant') {
            htmlContent = DOMPurify.sanitize(marked.parse(content));
        }

        msgDiv.innerHTML = `
            <div class="avatar">${avatar}</div>
            <div class="message-content">${htmlContent}</div>
        `;
        
        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        return msgDiv;
    }

    function addTypingIndicator() {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message assistant-message typing-indicator-msg';
        msgDiv.innerHTML = `
            <div class="avatar">✨</div>
            <div class="message-content" style="background: transparent; border: none; box-shadow: none;">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;
        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        return msgDiv;
    }

    // Node Icons mapping
    const nodeIcons = {
        'analyze_query': '🔍',
        'rewrite_query': '✍️',
        'retrieve': '📚',
        'rerank': '⚖️',
        'grade_documents': '🧪',
        'web_search_fallback': '🌐',
        'merge_context': '🧩',
        'generate': '🧠',
        'grade_answer': '✅',
        'handle_error': '⚠️'
    };

    function formatNodeName(name) {
        return name.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
    }

    function addTrace(nodeName, status, data = null) {
        // Clear empty state if first trace
        const emptyState = traceContainer.querySelector('.empty-state');
        if (emptyState) emptyState.remove();

        const icon = nodeIcons[nodeName] || '⚙️';
        const formattedName = formatNodeName(nodeName);

        // Check if item exists
        let item = document.getElementById(`trace-${nodeName}`);
        
        if (!item) {
            item = document.createElement('div');
            item.id = `trace-${nodeName}`;
            item.className = 'trace-item active';
            item.innerHTML = `
                <div class="trace-header">
                    <span class="trace-icon">${icon}</span>
                    <span>${formattedName}</span>
                    <span style="margin-left: auto; font-size: 10px; color: var(--accent);">Running...</span>
                </div>
            `;
            traceContainer.appendChild(item);
        }

        if (status === 'complete') {
            item.classList.remove('active');
            item.querySelector('.trace-header span:last-child').innerHTML = '<span style="color:var(--success)">Done</span>';
            
            if (data && Object.keys(data).length > 0) {
                const dataDiv = document.createElement('div');
                dataDiv.className = 'trace-data';
                dataDiv.textContent = JSON.stringify(data, null, 2);
                item.appendChild(dataDiv);
            }
        }
        
        traceContainer.scrollTop = traceContainer.scrollHeight;
    }

    function renderSources(sources) {
        sourcesContainer.innerHTML = '';
        if (!sources || sources.length === 0) {
            sourcesContainer.innerHTML = '<div class="empty-state">No sources used.</div>';
            return;
        }

        sources.forEach((src, idx) => {
            const card = document.createElement('div');
            card.className = 'source-card';
            
            // Clean up source path for display
            let displaySource = src.source || src.url || 'Unknown';
            if (displaySource.includes('/')) {
                displaySource = displaySource.split('/').pop();
            }

            card.innerHTML = `
                <div class="source-title">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-top:2px"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                    ${displaySource}
                </div>
                <div class="source-meta">
                    <span>Chunk: ${src.chunk_index || 'N/A'}</span>
                    <span class="badge">[${idx + 1}]</span>
                </div>
            `;
            sourcesContainer.appendChild(card);
        });
    }

    // Submit form handler
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query) return;

        // Reset UI
        queryInput.value = '';
        queryInput.style.height = 'auto';
        sendBtn.disabled = true;
        traceContainer.innerHTML = '<div class="empty-state">Processing...</div>';
        sourcesContainer.innerHTML = '<div class="empty-state">Waiting for retrieval...</div>';

        // Add user message
        appendMessage('user', query);
        
        const isStream = streamToggle.checked;
        const maxIter = parseInt(maxIterations.value) || 3;

        const requestBody = {
            query: query,
            max_iterations: maxIter,
            stream: isStream
        };

        if (isStream) {
            // Streaming via SSE
            const typingMsg = addTypingIndicator();
            let finalAnswer = "";
            let sources = [];

            try {
                const response = await fetch('/query/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestBody)
                });

                if (!response.ok) throw new Error("Server error");

                const reader = response.body.getReader();
                const decoder = new TextDecoder("utf-8");

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value, { stream: true });
                    const lines = chunk.split('\n\n');
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const dataStr = line.replace('data: ', '');
                            try {
                                const data = JSON.parse(dataStr);
                                
                                if (data.status === 'done') {
                                    typingMsg.remove();
                                    appendMessage('assistant', finalAnswer, true);
                                    renderSources(sources);
                                    break;
                                }
                                
                                if (data.error) {
                                    typingMsg.remove();
                                    appendMessage('assistant', `⚠️ **Error:** ${data.error}`, true);
                                    break;
                                }

                                if (data.node) {
                                    addTrace(data.node, data.status, data.data);
                                    
                                    // Extract answer from generate node
                                    if (data.node === 'generate' && data.status === 'complete') {
                                        if (data.data && data.data.answer) {
                                            finalAnswer = data.data.answer;
                                        }
                                    }
                                    // Extract context from merge_context
                                    if (data.node === 'merge_context' && data.status === 'complete') {
                                        // The backend streams partial updates, so we can't get the full objects easily here
                                        // But we update sources text to let user know
                                        sourcesContainer.innerHTML = '<div class="empty-state" style="color:var(--success)">Context Merged. Generating answer...</div>';
                                    }
                                }
                            } catch (e) {
                                console.error("Error parsing SSE data", e);
                            }
                        }
                    }
                }
                
                // After stream ends, we can do a quick fetch to get structured sources if needed, 
                // but since the endpoint returns final state, we might need a separate endpoint. 
                // For now, we rely on inline citations.

            } catch (err) {
                typingMsg.remove();
                appendMessage('assistant', `⚠️ **Failed to connect:** ${err.message}`, true);
                traceContainer.innerHTML = '<div class="empty-state" style="color:var(--danger)">Connection failed.</div>';
            }

        } else {
            // Synchronous request
            const typingMsg = addTypingIndicator();
            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestBody)
                });
                
                const data = await response.json();
                typingMsg.remove();
                
                if (response.ok) {
                    appendMessage('assistant', data.answer, true);
                    renderSources(data.sources);
                    traceContainer.innerHTML = `
                        <div class="trace-item">
                            <div class="trace-header"><span style="color:var(--success)">✅</span> Execution Complete</div>
                            <div class="trace-data">
                                Iterations: ${data.iterations}
                                <br>Retrieval Type: ${data.retrieval_type}
                                <br>Addressed Query: ${data.grade?.addresses_query}
                                <br>Confidence: ${data.grade?.confidence}
                            </div>
                        </div>
                    `;
                } else {
                    appendMessage('assistant', `⚠️ **Error:** ${data.error || 'Unknown error'}`, true);
                    traceContainer.innerHTML = `<div class="empty-state" style="color:var(--danger)">${data.error}</div>`;
                }
            } catch (err) {
                typingMsg.remove();
                appendMessage('assistant', `⚠️ **Failed to connect:** ${err.message}`, true);
                traceContainer.innerHTML = '<div class="empty-state" style="color:var(--danger)">Connection failed.</div>';
            }
        }
    });
});
