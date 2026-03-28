"""Persistent conversation memory backed by SQLite with FTS5 full-text search.

The database lives at ``~/.make-agent/<project-slug>/memory.db``.

Schema overview:
- ``messages``      — base table (id, created_at, sender, message)
- ``messages_fts``  — FTS5 content table over ``messages``
- ``user_memory``   — view: messages WHERE sender = 'user'
- ``agent_memory``  — view: messages WHERE sender = 'agent'
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

    def close(self) -> None:
        """Close the underlying database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
