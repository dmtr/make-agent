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
from typing import Any

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
        ``total_tokens``, ``models`` (list of distinct model names used),
        and ``agents`` (dict mapping agent name to per-agent stats),
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

        # Per-agent breakdown
        agent_rows = conn.execute(
            "SELECT agent, SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens"
            " FROM token_usage WHERE session_id = ? GROUP BY agent ORDER BY agent",
            (session_id,),
        ).fetchall()

        agents = {}
        for arow in agent_rows:
            agent_name = arow["agent"]
            input_tok = arow["input_tokens"] or 0
            output_tok = arow["output_tokens"] or 0
            agents[agent_name] = {
                "input_tokens": input_tok,
                "output_tokens": output_tok,
                "total_tokens": input_tok + output_tok,
            }

        return {
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "total_tokens": row["input_tokens"] + row["output_tokens"],
            "models": models,
            "agents": agents,
        }

    def close(self) -> None:
        """Close the underlying database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# ── memory tool schemas ───────────────────────────────────────────────────────

_MEMORY_SEARCH_PARAMS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "FTS5 keyword query. Use individual keywords rather than full sentences — "
                "FTS5 matches on exact tokens, not phrases or semantics. "
                "For example, to find 'the goal of this project', use 'goal project' or 'goal'. "
                "Combine keywords with OR for broader recall: 'goal OR objective OR purpose'. "
                "Avoid stop words (the, of, is, a) as they are not indexed."
            ),
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of results to return (default: 10).",
        },
        "from_date": {
            "type": "string",
            "description": "ISO 8601 date string to filter results on or after (e.g. '2026-03-01').",
        },
        "to_date": {
            "type": "string",
            "description": "ISO 8601 date string to filter results on or before (e.g. '2026-03-31').",
        },
    },
    "required": ["query"],
}

MEMORY_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_user_memory",
            "description": (
                "Search past user messages using keyword-based full-text search (FTS5). "
                "Use this proactively to recall context from earlier in the conversation or past sessions. "
                "Query with short keywords — FTS5 does not match full sentences. "
                "If the first query returns no results, retry with broader or alternative keywords."
            ),
            "parameters": _MEMORY_SEARCH_PARAMS,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_agent_memory",
            "description": (
                "Search past agent replies using keyword-based full-text search (FTS5). "
                "Use this to recall what you previously told the user or decisions you made. "
                "Query with short keywords — FTS5 does not match full sentences. "
                "If the first query returns no results, retry with broader or alternative keywords."
            ),
            "parameters": _MEMORY_SEARCH_PARAMS,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_messages",
            "description": (
                "Fetch the N most recent messages from memory, in chronological order. "
                "Each entry shows the timestamp, sender (user or agent), and message text. "
                "Use this to quickly recall recent conversation context without needing keywords."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of recent messages to return (default: 10).",
                    },
                    "from_date": {
                        "type": "string",
                        "description": "ISO 8601 date string to filter results on or after (e.g. '2026-03-01').",
                    },
                    "to_date": {
                        "type": "string",
                        "description": "ISO 8601 date string to filter results on or before (e.g. '2026-03-31').",
                    },
                },
                "required": [],
            },
        },
    },
]


def get_memory_executors(memory: Memory) -> dict[str, Any]:
    """Return a name → callable executor map for memory tools bound to *memory*."""
    return {
        "search_user_memory": lambda query, limit=10, from_date=None, to_date=None, **_kw: memory.search_user(query, limit, from_date, to_date),
        "search_agent_memory": lambda query, limit=10, from_date=None, to_date=None, **_kw: memory.search_agent(query, limit, from_date, to_date),
        "get_recent_messages": lambda limit=10, from_date=None, to_date=None, **_kw: memory.recent(limit, from_date, to_date),
    }
