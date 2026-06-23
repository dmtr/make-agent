"""UserSessionManager — persist and recall session startup parameters in memory.db.

Schema addition (alongside existing ``messages`` and ``token_usage`` tables)::

    CREATE TABLE sessions (
        id          INTEGER PRIMARY KEY,
        session_id  TEXT NOT NULL UNIQUE,
        started_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
        ended_at    TEXT,           -- NULL until the session ends cleanly
        model       TEXT NOT NULL,
        params_json TEXT NOT NULL   -- JSON blob of all other CLI / AgentConfig params
    )

The ``params_json`` column stores every parameter that is not the model string:
``reasoning_effort``, ``max_tokens``, ``max_tool_output``, ``tool_timeout``,
``use_prompt_cache``, ``skills_dir``, ``enabled_skills``, ``trusted_skills``,
``disabled_builtin_tools``.  Unknown keys are ignored on read, so the schema
stays forward-compatible as new flags are added.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY,
    session_id  TEXT    NOT NULL UNIQUE,
    started_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    ended_at    TEXT,
    model       TEXT    NOT NULL,
    params_json TEXT    NOT NULL
)
"""


class UserSessionManager:
    """Persist and recall agent session startup parameters using a SQLite database.

    The database is shared with :class:`~make_agent.memory.Memory`; pass the
    same ``db_path`` so both classes operate on a single file.

    The connection is opened lazily on the first call and kept open for the
    lifetime of the instance.  Call :meth:`close` when you are done (or use the
    instance in a ``try/finally`` block).
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    # ── internal ──────────────────────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute(_SESSIONS_TABLE)
            self._conn.commit()
        return self._conn

    # ── public API ────────────────────────────────────────────────────────────

    def save_session_params(
        self,
        session_id: str,
        model: str,
        params: dict[str, Any],
    ) -> None:
        """Insert a new row for *session_id* with *model* and *params*.

        *params* is an arbitrary dict of CLI / ``AgentConfig`` values that will
        be JSON-serialised and stored in the ``params_json`` column.  The row's
        ``ended_at`` is left ``NULL`` until :meth:`update_session_ended` is
        called.

        If a row with the same *session_id* already exists it is replaced so
        that duplicate start calls are idempotent.
        """
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO sessions (session_id, model, params_json)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                model       = excluded.model,
                params_json = excluded.params_json,
                started_at  = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'),
                ended_at    = NULL
            """,
            (session_id, model, json.dumps(params)),
        )
        conn.commit()

    def update_session_ended(self, session_id: str) -> None:
        """Stamp the current UTC time into ``ended_at`` for *session_id*.

        Safe to call even when the row does not exist (no-op in that case).
        """
        conn = self._get_conn()
        conn.execute(
            """
            UPDATE sessions
            SET ended_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
            WHERE session_id = ?
            """,
            (session_id,),
        )
        conn.commit()

    def get_last_session_params(self) -> dict[str, Any] | None:
        """Return the most recently *started* session's parameters, or ``None``.

        The returned dict always contains a ``"model"`` key plus whatever was
        stored in ``params_json``.  Returns ``None`` when the ``sessions`` table
        is empty.
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT model, params_json FROM sessions ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        result: dict[str, Any] = {"model": row["model"]}
        try:
            extra = json.loads(row["params_json"])
            if isinstance(extra, dict):
                result.update(extra)
        except (json.JSONDecodeError, TypeError):
            pass
        return result

    def close(self) -> None:
        """Close the underlying database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
