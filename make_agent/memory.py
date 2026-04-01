"""Persistent conversation memory backed by SQLite with FTS5 full-text search.

The database lives at ``~/.make-agent/<project-slug>/memory.db``.

Schema overview:
- ``messages``      — base table (id, created_at, sender, message)
- ``messages_fts``  — FTS5 content table over ``messages``
- ``user_memory``   — view: messages WHERE sender = 'user'
- ``agent_memory``  — view: messages WHERE sender = 'agent'
- ``token_usage``   — per-LLM-call token counts (session_id, agent, model, input/output tokens)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

_SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS messages (
        id         INTEGER PRIMARY KEY,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
        sender     TEXT NOT NULL CHECK(sender IN ('user', 'agent')),
        message    TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS token_usage (
        id            INTEGER PRIMARY KEY,
        created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
        session_id    TEXT NOT NULL,
        agent         TEXT NOT NULL,
        model         TEXT NOT NULL,
        input_tokens  INTEGER NOT NULL DEFAULT 0,
        output_tokens INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
        message,
        content='messages',
        content_rowid='id'
    )
    """,
    """
    CREATE VIEW IF NOT EXISTS user_memory AS
        SELECT * FROM messages WHERE sender = 'user'
    """,
    """
    CREATE VIEW IF NOT EXISTS agent_memory AS
        SELECT * FROM messages WHERE sender = 'agent'
    """,
    """
    CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts(rowid, message) VALUES (new.id, new.message);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, message)
            VALUES ('delete', old.id, old.message);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, message)
            VALUES ('delete', old.id, old.message);
        INSERT INTO messages_fts(rowid, message) VALUES (new.id, new.message);
    END
    """,
]


class Memory:
    """Persistent agent memory stored in a SQLite database.

    The database and schema are created lazily on first use.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
            for stmt in _SCHEMA_STATEMENTS:
                self._conn.execute(stmt)
            self._conn.commit()
        return self._conn

    def store(self, sender: str, message: str) -> None:
        """Store a message from *sender* (``'user'`` or ``'agent'``)."""
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO messages (sender, message) VALUES (?, ?)",
            (sender, message),
        )
        conn.commit()

    def _search(
        self,
        view: str,
        query: str,
        limit: int = 10,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> str:
        conn = self._get_conn()
        sql = f"""
            SELECT v.created_at, v.message
            FROM {view} v
            JOIN messages_fts ON v.id = messages_fts.rowid
            WHERE messages_fts MATCH ?
        """
        params: list = [query]
        if from_date:
            sql += " AND v.created_at >= ?"
            params.append(from_date)
        if to_date:
            sql += " AND v.created_at <= ?"
            params.append(to_date)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)

        rows = conn.execute(sql, params).fetchall()
        if not rows:
            return "No results found."
        return "\n".join(f"[{row['created_at']}] {row['message']}" for row in rows)

    def search_user(
        self,
        query: str,
        limit: int = 10,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> str:
        """Search past user messages using FTS5 via the ``user_memory`` view."""
        return self._search("user_memory", query, limit, from_date, to_date)

    def search_agent(
        self,
        query: str,
        limit: int = 10,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> str:
        """Search past agent replies using FTS5 via the ``agent_memory`` view."""
        return self._search("agent_memory", query, limit, from_date, to_date)

    def recent(
        self,
        limit: int = 10,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> str:
        """Return the *limit* most recent messages, optionally filtered by date range."""
        conn = self._get_conn()
        sql = "SELECT created_at, sender, message FROM messages WHERE 1=1"
        params: list = []
        if from_date:
            sql += " AND created_at >= ?"
            params.append(from_date)
        if to_date:
            sql += " AND created_at <= ?"
            params.append(to_date)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(sql, params).fetchall()
        if not rows:
            return "No messages found."
        rows = list(reversed(rows))
        return "\n".join(f"[{row['created_at']}] {row['sender']}: {row['message']}" for row in rows)

    def record_token_usage(
        self,
        session_id: str,
        agent: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Insert one row into ``token_usage`` for a single LLM API call."""
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO token_usage (session_id, agent, model, input_tokens, output_tokens)"
            " VALUES (?, ?, ?, ?, ?)",
            (session_id, agent, model, input_tokens, output_tokens),
        )
        conn.commit()

    def get_session_stats(self, session_id: str) -> dict:
        """Return aggregated token usage totals for *session_id*.

        Returns a dict with keys ``input_tokens``, ``output_tokens``,
        ``total_tokens``, and ``models`` (list of distinct model names used),
        or an empty dict when no rows exist for that session.
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens"
            " FROM token_usage WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None or row["input_tokens"] is None:
            return {}
        models = [
            r["model"]
            for r in conn.execute(
                "SELECT DISTINCT model FROM token_usage WHERE session_id = ? ORDER BY model",
                (session_id,),
            ).fetchall()
        ]
        return {
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "total_tokens": row["input_tokens"] + row["output_tokens"],
            "models": models,
        }

    def close(self) -> None:
        """Close the underlying database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
