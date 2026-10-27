const $ = id => document.getElementById(id);
let history = [], imageData = null, recorder = null, audioChunks = [], cameraStream = null, lastAnswer = '', conversationId = null;

const headers = () => {
  const h = {};
  const token = $('token').value.trim();
  if (token) h.Authorization = `Bearer ${token}`;
  if (conversationId) h['X-Conversation-ID'] = conversationId;
  return h;
};

// Configure DOMPurify to open all links in a new tab securely
if (typeof DOMPurify !== 'undefined' && typeof DOMPurify.addHook === 'function') {
  DOMPurify.addHook('afterSanitizeAttributes', function(node) {
    if (node.tagName === 'A' && 'target' in node) {
      node.setAttribute('target', '_blank');
      node.setAttribute('rel', 'noopener noreferrer');
    }
  });
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function formatTime(isoString) {
  if (!isoString) return '';
  try {
    const date = new Date(isoString);
    const now = new Date();
    const diffSec = Math.floor((now - date) / 1000);
    if (diffSec < 60) return 'Just now';
    if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
    if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
    if (diffSec < 172800) return 'Yesterday';
    return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  } catch {
    return '';
  }
}

function renderMarkdown(content) {
  if (typeof content !== 'string') content = String(content || '');

  // Pre-process inline citations like [PDF p.5] or [PDF page 5] to styled badges
  content = content.replace(/\[PDF (?:p\.|page) *(\d+)\]/gi, (match, page) => {
    return ` <span class="source-tag inline-cite">📄 Gale Encyclopedia p.${page}</span> `;
  });

  if (typeof marked !== 'undefined') {
    try {
      const parseFn = typeof marked.parse === 'function' ? marked.parse : (typeof marked === 'function' ? marked : null);
      if (parseFn) {
        const rawHtml = parseFn(content, { breaks: true, gfm: true });
        return typeof DOMPurify !== 'undefined' && typeof DOMPurify.sanitize === 'function'
          ? DOMPurify.sanitize(rawHtml)
          : rawHtml;
      }
    } catch (err) {
      console.error('Markdown parse error:', err);
    }
  }

  const div = document.createElement('div');
  div.textContent = content;
  return div.innerHTML;
}

function formatSources(sources = []) {
  if (!sources || !sources.length) return null;
  const container = document.createElement('div');
  container.className = 'sources';
  
  const label = document.createElement('strong');
  label.textContent = 'Verified Sources:';
  container.appendChild(label);

  const seen = new Set();
  sources.slice(0, 10).forEach(s => {
    if (s.type === 'pdf') {
      const key = `pdf-${s.page}`;
      if (seen.has(key)) return;
      seen.add(key);
      const tag = document.createElement('span');
      tag.className = 'source-tag';
      tag.textContent = `📄 Gale Encyclopedia p.${s.page}`;
      container.appendChild(tag);
    } else if (s.url) {
      if (seen.has(s.url)) return;
      seen.add(s.url);
      const link = document.createElement('a');
      link.className = 'source-link';
      link.href = s.url;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      let title = s.title;
      if (!title) {
        try { title = (new URL(s.url)).hostname.replace(/^www\./, ''); } catch { title = s.url; }
      }
      link.textContent = `🔗 ${title}`;
      container.appendChild(link);
    }
  });

  return container;
}

function bubble(text, kind = 'bot', sources = []) {
  const el = document.createElement('article');
  el.className = kind;

  if (kind === 'bot') {
    const contentDiv = document.createElement('div');
    contentDiv.className = 'content';
    contentDiv.innerHTML = renderMarkdown(text);
    el.appendChild(contentDiv);
  } else {
    el.textContent = text;
  }

  const sourcesEl = formatSources(sources);
  if (sourcesEl) el.appendChild(sourcesEl);

  $('messages').appendChild(el);
  $('messages').scrollTop = $('messages').scrollHeight;
  return el;
}

function renderWelcomeCard() {
  const article = document.createElement('article');
  article.className = 'bot welcome-card';
  article.innerHTML = `
    <div class="content">
      <h3>Welcome to MedGuide AI</h3>
      <p>Ask a medical question, upload a skin or report photo, or use voice input. I verify findings against the Gale Encyclopedia and current medical clinical guidelines.</p>
      <div class="starter-prompts">
        <button class="starter-btn" type="button" data-prompt="What are the early warning signs and symptoms of bronchitis?">
          📋 Early signs of bronchitis
        </button>
        <button class="starter-btn" type="button" data-prompt="How to manage asthma attacks and when to seek emergency care?">
          🫁 Asthma emergency guidelines
        </button>
        <button class="starter-btn" type="button" data-prompt="What is the difference between viral and bacterial throat infections?">
          🩺 Viral vs bacterial infection
        </button>
      </div>
    </div>
  `;
  return article;
}

// Sidebar toggle for mobile/responsive
function toggleSidebar(forceOpen) {
  const sidebar = $('sidebar');
  const backdrop = $('sidebarBackdrop');
  const shouldOpen = forceOpen !== undefined ? forceOpen : !sidebar.classList.contains('open');
  if (shouldOpen) {
    sidebar.classList.add('open');
    backdrop.hidden = false;
  } else {
    sidebar.classList.remove('open');
    backdrop.hidden = true;
  }
}

if ($('toggleSidebar')) $('toggleSidebar').onclick = () => toggleSidebar(true);
if ($('closeSidebar')) $('closeSidebar').onclick = () => toggleSidebar(false);
if ($('sidebarBackdrop')) $('sidebarBackdrop').onclick = () => toggleSidebar(false);

// Load and manage Conversations
async function loadConversations() {
  try {
    const r = await fetch('/api/conversations', { headers: headers() });
    if (!r.ok) return;
    const list = await r.json();

    const ul = $('chatList');
    ul.innerHTML = '';
    const countEl = $('chatCount');
    if (countEl) countEl.textContent = list.length;

    const empty = $('emptyHistory');
    if (empty) empty.hidden = list.length > 0;

    list.forEach(conv => {
      const li = document.createElement('li');
      li.className = 'chat-item' + (conv.id === conversationId ? ' active' : '');
      li.dataset.id = conv.id;
      
      const main = document.createElement('div');
      main.className = 'chat-item-main';
      main.innerHTML = `
        <span class="chat-item-icon">💬</span>
        <div class="chat-item-text">
          <span class="chat-item-title" title="${escapeHtml(conv.title)}">${escapeHtml(conv.title)}</span>
          <span class="chat-item-time">${formatTime(conv.updated_at)}</span>
        </div>
      `;
      main.onclick = () => openConversation(conv.id);

      const delBtn = document.createElement('button');
      delBtn.className = 'chat-delete-btn';
      delBtn.type = 'button';
      delBtn.title = 'Delete consultation';
      delBtn.setAttribute('aria-label', 'Delete');
      delBtn.innerHTML = '🗑';
      delBtn.onclick = (e) => {
        e.stopPropagation();
        deleteConversation(conv.id);
      };

      li.appendChild(main);
      li.appendChild(delBtn);
      ul.appendChild(li);
    });
  } catch (err) {
    console.warn('Failed to load conversations:', err);
  }
}

async function openConversation(id) {
  if (conversationId === id && history.length > 0) {
    toggleSidebar(false);
    return;
  }

  try {
    const r = await fetch(`/api/conversations/${id}`, { headers: headers() });
    if (!r.ok) throw new Error('Could not retrieve conversation');
    const data = await r.json();

    conversationId = data.id;
    history = [];
    lastAnswer = '';

    // Update UI headers
    $('currentChatTitle').textContent = data.title;
    $('chatSubtitle').textContent = 'Active consultation • Resumed from history';

    // Clear messages pane and render restored messages
    $('messages').innerHTML = '';
    data.messages.forEach(m => {
      if (m.role === 'user') {
        bubble(m.content, 'user');
        history.push({ role: 'user', content: m.content });
      } else {
        bubble(m.content, 'bot', m.sources || []);
        history.push({ role: 'assistant', content: m.content });
        lastAnswer = m.content;
      }
    });

    history = history.slice(-8);

    // Update active highlight in sidebar
    document.querySelectorAll('.chat-item').forEach(el => {
      el.classList.toggle('active', el.dataset.id === id);
    });

    toggleSidebar(false);
    $('messages').scrollTop = $('messages').scrollHeight;
  } catch (err) {
    alert('Failed to load consultation: ' + err.message);
  }
}

function startNewChat() {
  conversationId = null;
  history = [];
  lastAnswer = '';
  imageData = null;

  $('currentChatTitle').textContent = '✚ MedGuide AI';
  $('chatSubtitle').textContent = 'Medical education • Knowledge base + web research';

  $('messages').innerHTML = '';
  $('messages').appendChild(renderWelcomeCard());

  $('preview').hidden = true;
  $('image').value = '';
  $('message').value = '';

  document.querySelectorAll('.chat-item').forEach(el => el.classList.remove('active'));

  toggleSidebar(false);
  $('message').focus();
}

async function deleteConversation(id) {
  if (!confirm('Delete this consultation from history?')) return;
  try {
    const r = await fetch(`/api/conversations/${id}`, { method: 'DELETE', headers: headers() });
    if (!r.ok) throw new Error('Failed to delete');

    if (conversationId === id) {
      startNewChat();
    }
    loadConversations();
  } catch (err) {
    alert('Could not delete consultation: ' + err.message);
  }
}

if ($('newChatBtn')) $('newChatBtn').onclick = () => startNewChat();
if ($('headerNewChat')) $('headerNewChat').onclick = () => startNewChat();

// Delegate clicks on starter prompt buttons
$('messages').addEventListener('click', e => {
  const btn = e.target.closest('.starter-btn');
  if (btn && btn.dataset.prompt) {
    $('message').value = btn.dataset.prompt;
    $('message').focus();
  }
});

async function check() {
  try {
    const r = await fetch('/api/health');
    const d = await r.json();
    const statusText = d.configured ? '● Connected' : '⚠ Configure API keys';
    $('status').textContent = statusText;
    if ($('sidebarStatus')) $('sidebarStatus').textContent = d.configured ? 'System Ready' : 'API Keys Needed';
  } catch {
    $('status').textContent = '● Server offline';
    if ($('sidebarStatus')) $('sidebarStatus').textContent = 'Server Offline';
  }
}

check();
loadConversations();

function setImage(file) {
  if (!file) return;
  if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type) || file.size > 5e6) {
    alert('Use PNG/JPEG/WebP under 5MB');
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    imageData = reader.result;
    $('thumbnail').src = imageData;
    $('preview').hidden = false;
  };
  reader.readAsDataURL(file);
}

$('image').addEventListener('change', e => setImage(e.target.files[0]));
$('remove').onclick = () => {
  imageData = null;
  $('preview').hidden = true;
  $('image').value = '';
};

$('chatForm').addEventListener('submit', async e => {
  e.preventDefault();
  const message = $('message').value.trim();
  if (!message) return;

  const prior = history.slice(-8);
  bubble(message, 'user');
  $('message').value = '';
  $('send').disabled = true;

  const pending = document.createElement('article');
  pending.className = 'bot';

  const statusBadge = document.createElement('div');
  statusBadge.className = 'stream-status';
  statusBadge.innerHTML = '<span class="typing-indicator"><span></span><span></span><span></span></span><span class="status-text">Analyzing medical query & structuring criteria…</span>';
  pending.appendChild(statusBadge);

  const contentDiv = document.createElement('div');
  contentDiv.className = 'content';
  pending.appendChild(contentDiv);

  const sourcesWrapper = document.createElement('div');
  sourcesWrapper.className = 'sources-wrapper';
  pending.appendChild(sourcesWrapper);

  $('messages').appendChild(pending);
  $('messages').scrollTop = $('messages').scrollHeight;

  const reqImageData = imageData;
  // Clear image thumbnail preview immediately
  $('remove').click();

  let accumulatedAnswer = '';
  let receivedSources = [];

  try {
    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { ...headers(), 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        history: prior,
        image_data: reqImageData,
        conversation_id: conversationId
      })
    });

    if (!response.ok) {
      let errText = 'Consultation request failed';
      try {
        const errJson = await response.json();
        errText = errJson.detail || errText;
      } catch {}
      throw new Error(errText);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    const handleEvent = (ev) => {
      if (!ev || !ev.type) return;

      if (ev.type === 'status') {
        const statusText = statusBadge.querySelector('.status-text');
        if (statusText) statusText.textContent = ev.message;
      } else if (ev.type === 'sources') {
        receivedSources = ev.sources || [];
        sourcesWrapper.innerHTML = '';
        const sourcesEl = formatSources(receivedSources);
        if (sourcesEl) sourcesWrapper.appendChild(sourcesEl);
      } else if (ev.type === 'token') {
        const chunk = ev.token !== undefined ? ev.token : (ev.content || '');
        accumulatedAnswer += chunk;
        contentDiv.innerHTML = renderMarkdown(accumulatedAnswer);
        $('messages').scrollTop = $('messages').scrollHeight;
      } else if (ev.type === 'replace') {
        accumulatedAnswer = ev.answer || ev.content || '';
        contentDiv.innerHTML = renderMarkdown(accumulatedAnswer);
        $('messages').scrollTop = $('messages').scrollHeight;
      } else if (ev.type === 'done') {
        if (ev.conversation_id) {
          conversationId = ev.conversation_id;
        }
        if (ev.title) {
          $('currentChatTitle').textContent = ev.title;
          $('chatSubtitle').textContent = 'Active consultation';
        }
        lastAnswer = ev.answer || accumulatedAnswer;
        contentDiv.innerHTML = renderMarkdown(lastAnswer);

        // Ensure sources are shown
        const finalSources = (ev.sources && ev.sources.length) ? ev.sources : receivedSources;
        if (finalSources.length) {
          sourcesWrapper.innerHTML = '';
          const sourcesEl = formatSources(finalSources);
          if (sourcesEl) sourcesWrapper.appendChild(sourcesEl);
        }

        // Remove status badge
        if (statusBadge && statusBadge.parentNode) {
          statusBadge.remove();
        }

        history.push({ role: 'user', content: message }, { role: 'assistant', content: lastAnswer });
        loadConversations();
      }
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const blocks = buffer.split('\n\n');
      buffer = blocks.pop(); // Keep incomplete block in buffer

      for (const block of blocks) {
        const trimmed = block.trim();
        if (!trimmed) continue;
        const lines = trimmed.split('\n');
        for (const line of lines) {
          if (line.startsWith('data:')) {
            const jsonStr = line.slice(5).trim();
            if (jsonStr) {
              try {
                const ev = JSON.parse(jsonStr);
                handleEvent(ev);
              } catch (parseErr) {
                console.warn('SSE JSON parse error:', parseErr, jsonStr);
              }
            }
          }
        }
      }
    }

    // Flush any leftover in buffer
    if (buffer.trim()) {
      const lines = buffer.trim().split('\n');
      for (const line of lines) {
        if (line.startsWith('data:')) {
          const jsonStr = line.slice(5).trim();
          if (jsonStr) {
            try {
              const ev = JSON.parse(jsonStr);
              handleEvent(ev);
            } catch {}
          }
        }
      }
    }

    // Final safety cleanup if 'done' wasn't sent or caught
    if (statusBadge && statusBadge.parentNode) {
      statusBadge.remove();
    }
    if (!lastAnswer && accumulatedAnswer) {
      lastAnswer = accumulatedAnswer;
      history.push({ role: 'user', content: message }, { role: 'assistant', content: lastAnswer });
      loadConversations();
    }
  } catch (err) {
    if (statusBadge && statusBadge.parentNode) {
      statusBadge.remove();
    }
    contentDiv.innerHTML = `<p style="color:#b3261e;margin:0;">⚠️ <strong>Service issue:</strong> ${escapeHtml(err.message)}</p>`;
  } finally {
    $('send').disabled = false;
    $('messages').scrollTop = $('messages').scrollHeight;
  }
});

$('mic').onclick = async () => {
  if (recorder && recorder.state === 'recording') {
    recorder.stop();
    $('mic').textContent = '🎤 Voice';
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = e => audioChunks.push(e.data);
    recorder.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      const blob = new Blob(audioChunks, { type: recorder.mimeType });
      const data = new FormData();
      data.append('file', blob, 'voice.webm');
      $('mic').disabled = true;
      try {
        const r = await fetch('/api/transcribe', { method: 'POST', headers: headers(), body: data });
        const out = await r.json();
        if (!r.ok) throw Error(out.detail);
        $('message').value = out.text;
      } catch (e) {
        alert(e.message);
      } finally {
        $('mic').disabled = false;
      }
    };
    recorder.start();
    $('mic').textContent = '⏹ Stop recording';
  } catch (e) {
    alert('Microphone unavailable: ' + e.message);
  }
};

$('listen').onclick = () => {
  if (!lastAnswer) return alert('Ask a question first');
  if (!('speechSynthesis' in window)) {
    alert('Speech synthesis is not supported in this browser.');
    return;
  }
  if (window.speechSynthesis.speaking) {
    window.speechSynthesis.cancel();
    $('listen').textContent = '🔊 Read answer';
    return;
  }
  const utterance = new SpeechSynthesisUtterance(lastAnswer);
  utterance.onstart = () => { $('listen').textContent = '⏹ Stop speaking'; };
  utterance.onend = () => { $('listen').textContent = '🔊 Read answer'; };
  utterance.onerror = () => { $('listen').textContent = '🔊 Read answer'; };
  window.speechSynthesis.speak(utterance);
};

$('camera').onclick = async () => {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
    $('video').srcObject = cameraStream;
    $('video').hidden = false;
    $('cameraActions').hidden = false;
  } catch (e) {
    alert('Camera unavailable: ' + e.message);
  }
};

$('closeCamera').onclick = () => {
  cameraStream?.getTracks().forEach(t => t.stop());
  cameraStream = null;
  $('video').hidden = true;
  $('cameraActions').hidden = true;
};

$('snap').onclick = () => {
  const v = $('video'), c = $('canvas');
  c.width = v.videoWidth;
  c.height = v.videoHeight;
  c.getContext('2d').drawImage(v, 0, 0);
  c.toBlob(b => setImage(new File([b], 'camera.jpg', { type: 'image/jpeg' })), 'image/jpeg', 0.85);
  $('closeCamera').click();
};