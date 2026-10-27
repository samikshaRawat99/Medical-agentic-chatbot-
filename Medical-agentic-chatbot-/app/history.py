from __future__ import annotations
import sqlite3
import json
from typing import Optional
from datetime import datetime
from groq import Groq
from app.config import settings

DB_PATH = "medical_memory.db"

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_history_db():
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT DEFAULT '[]',
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id)")
        conn.commit()

# Initialize tables immediately on module load
init_history_db()

def generate_title(question: str) -> str:
    """Generate a concise 3-5 word topic title using Groq chat model, with smart fallback."""
    try:
        if settings.groq_api_key and settings.groq_chat_model:
            client = Groq(api_key=settings.groq_api_key, timeout=10)
            prompt = (
                "Generate a concise 3 to 5 word topic title summarizing this medical topic/question. "
                "Do not use quotes, punctuation, or generic words like 'Question about' or 'Inquiry'. "
                "Capitalize each word.\n\n"
                f"Medical question: {question[:300]}"
            )
            res = client.chat.completions.create(
                model=settings.groq_chat_model,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=20,
                temperature=0.2
            )
            raw = res.choices[0].message.content.strip().strip('"\'')
            if raw:
                # Take first line and remove trailing punctuation
                title = raw.split('\n')[0].strip(' .:-')
                if 2 <= len(title) <= 60:
                    return title
    except Exception as exc:
        print("Topic title generation error:", repr(exc), flush=True)

    # Fallback to key words from user question
    words = [w for w in question.replace('?', '').replace('!', '').split() if w.lower() not in ('what', 'is', 'the', 'a', 'an', 'how', 'to', 'can', 'i', 'do', 'are')]
    if words:
        clean = ' '.join(words[:4]).title()
        return clean[:50]
    return "Medical Consultation"

def list_conversations() -> list[dict]:
    """Return all conversations ordered by latest update."""
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC")
        rows = c.fetchall()
        return [dict(row) for row in rows]

def get_conversation(conv_id: str) -> Optional[dict]:
    """Return metadata and full message history for a conversation."""
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?", (conv_id,))
        conv_row = c.fetchone()
        if not conv_row:
            return None
        c.execute(
            "SELECT role, content, sources, created_at FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conv_id,)
        )
        msg_rows = c.fetchall()
        messages = []
        for r in msg_rows:
            sources = []
            if r["sources"]:
                try:
                    sources = json.loads(r["sources"])
                except Exception:
                    sources = []
            messages.append({
                "role": r["role"],
                "content": r["content"],
                "sources": sources,
                "created_at": r["created_at"]
            })
        data = dict(conv_row)
        data["messages"] = messages
        return data

def save_conversation_turn(conv_id: str, user_question: str, bot_answer: str, sources: list = None, title: str = None) -> str:
    """Save a turn (user + bot) into SQLite, creating conversation with topic title if new."""
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id, title FROM conversations WHERE id = ?", (conv_id,))
        existing = c.fetchone()
        if not existing:
            if not title:
                title = generate_title(user_question)
            c.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conv_id, title, now, now)
            )
        else:
            title = existing["title"]
            c.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id))

        # Insert user message
        c.execute(
            "INSERT INTO messages (conversation_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
            (conv_id, "user", user_question, "[]", now)
        )
        # Insert assistant message
        c.execute(
            "INSERT INTO messages (conversation_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
            (conv_id, "assistant", bot_answer, json.dumps(sources or []), now)
        )
        conn.commit()
    return title

def delete_conversation(conv_id: str) -> bool:
    """Delete a conversation and all its messages."""
    with get_db() as conn:
        c = conn.cursor()
        c.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
        c.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        conn.commit()
        return c.rowcount > 0
