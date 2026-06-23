"""Tests for UserSessionManager — session parameter persistence."""

from __future__ import annotations

import json
import time

import pytest

from make_agent.memory import UserSessionManager


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mgr(tmp_path):
    """Return a fresh UserSessionManager backed by a temp DB."""
    m = UserSessionManager(tmp_path / "memory.db")
    yield m
    m.close()


# ── Schema & initialisation ───────────────────────────────────────────────────


class TestUserSessionManagerSchema:
    def test_db_file_created_on_first_use(self, tmp_path):
        db_path = tmp_path / "memory.db"
        assert not db_path.exists()
        m = UserSessionManager(db_path)
        m.save_session_params("s1", "model-x", {})
        assert db_path.exists()
        m.close()

    def test_parent_dirs_created(self, tmp_path):
        db_path = tmp_path / "deep" / "nested" / "memory.db"
        m = UserSessionManager(db_path)
        m.save_session_params("s1", "model-x", {})
        assert db_path.exists()
        m.close()

    def test_sessions_table_columns(self, mgr):
        conn = mgr._get_conn()
        info = conn.execute("PRAGMA table_info(sessions)").fetchall()
        col_names = [row[1] for row in info]
        assert "id" in col_names
        assert "session_id" in col_names
        assert "started_at" in col_names
        assert "ended_at" in col_names
        assert "model" in col_names
        assert "params_json" in col_names

    def test_idempotent_init(self, tmp_path):
        """Opening the same DB twice does not raise."""
        db_path = tmp_path / "memory.db"
        m1 = UserSessionManager(db_path)
        m1.save_session_params("s1", "model-a", {})
        m1.close()
        m2 = UserSessionManager(db_path)
        m2.save_session_params("s2", "model-b", {})
        conn = m2._get_conn()
        count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        assert count == 2
        m2.close()


# ── save_session_params ───────────────────────────────────────────────────────


class TestSaveSessionParams:
    def test_row_is_inserted(self, mgr):
        mgr.save_session_params("sid-1", "claude-3", {"max_tokens": 4096})
        conn = mgr._get_conn()
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = 'sid-1'"
        ).fetchone()
        assert row is not None
        assert row["model"] == "claude-3"

    def test_params_json_is_stored(self, mgr):
        params = {
            "reasoning_effort": "high",
            "max_tokens": 8192,
            "use_prompt_cache": True,
        }
        mgr.save_session_params("sid-2", "gpt-4o", params)
        conn = mgr._get_conn()
        row = conn.execute(
            "SELECT params_json FROM sessions WHERE session_id = 'sid-2'"
        ).fetchone()
        stored = json.loads(row["params_json"])
        assert stored == params

    def test_started_at_is_populated(self, mgr):
        mgr.save_session_params("sid-3", "model-x", {})
        conn = mgr._get_conn()
        row = conn.execute(
            "SELECT started_at FROM sessions WHERE session_id = 'sid-3'"
        ).fetchone()
        assert row["started_at"] is not None
        assert "T" in row["started_at"]  # ISO 8601

    def test_ended_at_is_null_on_insert(self, mgr):
        mgr.save_session_params("sid-4", "model-x", {})
        conn = mgr._get_conn()
        row = conn.execute(
            "SELECT ended_at FROM sessions WHERE session_id = 'sid-4'"
        ).fetchone()
        assert row["ended_at"] is None

    def test_duplicate_session_id_is_replaced(self, mgr):
        mgr.save_session_params("sid-dup", "model-a", {"max_tokens": 1024})
        mgr.save_session_params("sid-dup", "model-b", {"max_tokens": 2048})
        conn = mgr._get_conn()
        rows = conn.execute(
            "SELECT * FROM sessions WHERE session_id = 'sid-dup'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["model"] == "model-b"
        assert json.loads(rows[0]["params_json"])["max_tokens"] == 2048

    def test_duplicate_resets_ended_at_to_null(self, mgr):
        mgr.save_session_params("sid-re", "model-a", {})
        mgr.update_session_ended("sid-re")
        # Re-save same session_id (e.g. restart scenario)
        mgr.save_session_params("sid-re", "model-a", {})
        conn = mgr._get_conn()
        row = conn.execute(
            "SELECT ended_at FROM sessions WHERE session_id = 'sid-re'"
        ).fetchone()
        assert row["ended_at"] is None

    def test_empty_params_stored_as_empty_object(self, mgr):
        mgr.save_session_params("sid-empty", "model-x", {})
        conn = mgr._get_conn()
        row = conn.execute(
            "SELECT params_json FROM sessions WHERE session_id = 'sid-empty'"
        ).fetchone()
        assert json.loads(row["params_json"]) == {}

    def test_multiple_sessions_stored_independently(self, mgr):
        mgr.save_session_params("sid-a", "model-a", {"max_tokens": 100})
        mgr.save_session_params("sid-b", "model-b", {"max_tokens": 200})
        conn = mgr._get_conn()
        count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        assert count == 2


# ── update_session_ended ──────────────────────────────────────────────────────


class TestUpdateSessionEnded:
    def test_stamps_ended_at(self, mgr):
        mgr.save_session_params("sid-end", "model-x", {})
        mgr.update_session_ended("sid-end")
        conn = mgr._get_conn()
        row = conn.execute(
            "SELECT ended_at FROM sessions WHERE session_id = 'sid-end'"
        ).fetchone()
        assert row["ended_at"] is not None
        assert "T" in row["ended_at"]

    def test_ended_at_is_after_started_at(self, mgr):
        mgr.save_session_params("sid-timing", "model-x", {})
        time.sleep(0.01)  # ensure a measurable gap
        mgr.update_session_ended("sid-timing")
        conn = mgr._get_conn()
        row = conn.execute(
            "SELECT started_at, ended_at FROM sessions WHERE session_id = 'sid-timing'"
        ).fetchone()
        assert row["ended_at"] >= row["started_at"]

    def test_noop_for_unknown_session_id(self, mgr):
        """Calling update_session_ended for a non-existent session must not raise."""
        mgr.update_session_ended("ghost-session-id")  # should not raise

    def test_only_target_session_is_updated(self, mgr):
        mgr.save_session_params("sid-x", "model-x", {})
        mgr.save_session_params("sid-y", "model-y", {})
        mgr.update_session_ended("sid-x")
        conn = mgr._get_conn()
        row_y = conn.execute(
            "SELECT ended_at FROM sessions WHERE session_id = 'sid-y'"
        ).fetchone()
        assert row_y["ended_at"] is None


# ── get_last_session_params ───────────────────────────────────────────────────


class TestGetLastSessionParams:
    def test_returns_none_when_empty(self, mgr):
        assert mgr.get_last_session_params() is None

    def test_returns_model(self, mgr):
        mgr.save_session_params("sid-1", "claude-opus", {})
        result = mgr.get_last_session_params()
        assert result is not None
        assert result["model"] == "claude-opus"

    def test_returns_extra_params(self, mgr):
        params = {"reasoning_effort": "high", "max_tokens": 8192}
        mgr.save_session_params("sid-2", "gpt-4o", params)
        result = mgr.get_last_session_params()
        assert result["reasoning_effort"] == "high"
        assert result["max_tokens"] == 8192

    def test_returns_most_recent_session(self, mgr):
        mgr.save_session_params("sid-old", "model-old", {"max_tokens": 1000})
        mgr.save_session_params("sid-new", "model-new", {"max_tokens": 9999})
        result = mgr.get_last_session_params()
        assert result["model"] == "model-new"
        assert result["max_tokens"] == 9999

    def test_model_key_always_present(self, mgr):
        mgr.save_session_params("sid-1", "model-x", {"reasoning_effort": "low"})
        result = mgr.get_last_session_params()
        assert "model" in result

    def test_unknown_keys_in_params_json_included(self, mgr):
        """Extra/future keys stored in params_json should be returned as-is."""
        mgr.save_session_params("sid-future", "model-x", {"future_flag": True})
        result = mgr.get_last_session_params()
        assert result.get("future_flag") is True

    def test_corrupt_params_json_returns_model_only(self, tmp_path):
        """Gracefully handle a corrupted params_json column."""
        db_path = tmp_path / "memory.db"
        m = UserSessionManager(db_path)
        m.save_session_params("sid-corrupt", "model-x", {})
        # Manually corrupt the JSON
        conn = m._get_conn()
        conn.execute(
            "UPDATE sessions SET params_json = 'NOT_JSON' WHERE session_id = 'sid-corrupt'"
        )
        conn.commit()
        result = m.get_last_session_params()
        assert result is not None
        assert result["model"] == "model-x"
        assert len(result) == 1  # only model key, no extra params
        m.close()

    def test_session_ordering_by_insertion(self, mgr):
        """Last inserted row is returned regardless of session_id sort order."""
        mgr.save_session_params("zzz", "model-zzz", {})
        mgr.save_session_params("aaa", "model-aaa", {})
        result = mgr.get_last_session_params()
        assert result["model"] == "model-aaa"


# ── Shared DB with Memory ─────────────────────────────────────────────────────


class TestSharedDatabase:
    def test_sessions_table_coexists_with_messages_table(self, tmp_path):
        """UserSessionManager and Memory can share the same db_path."""
        from make_agent.memory import Memory

        db_path = tmp_path / "memory.db"
        mem = Memory(db_path)
        mem.store("user", "hello")

        mgr = UserSessionManager(db_path)
        mgr.save_session_params("sid-shared", "model-x", {})

        # Both tables should be queryable from Memory's connection
        conn = mem._get_conn()
        msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        assert msg_count == 1

        # And from UserSessionManager's connection
        conn2 = mgr._get_conn()
        sess_count = conn2.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        assert sess_count == 1

        mem.close()
        mgr.close()

    def test_session_table_does_not_interfere_with_memory_fts(self, tmp_path):
        """Adding session rows must not corrupt FTS5 indexes on messages."""
        from make_agent.memory import Memory

        db_path = tmp_path / "memory.db"
        mem = Memory(db_path)
        mem.store("user", "unique keyword wombat")

        mgr = UserSessionManager(db_path)
        mgr.save_session_params("sid-1", "model-x", {})

        result = mem.search_user("wombat")
        assert "unique keyword wombat" in result

        mem.close()
        mgr.close()


# ── _apply_last_session_defaults (main.py helper) ────────────────────────────


class TestApplyLastSessionDefaults:
    def test_fills_missing_model(self):
        import argparse
        from make_agent.main import _apply_last_session_defaults

        args = argparse.Namespace(
            model=None,
            reasoning_effort="medium",
            max_tokens=4096,
            max_tool_output=32000,
            tool_timeout=600,
            prompt_cache=False,
        )
        last = {"model": "claude-opus", "reasoning_effort": "high"}
        _apply_last_session_defaults(args, last, provided=frozenset())
        assert args.model == "claude-opus"

    def test_does_not_overwrite_explicitly_provided_arg(self):
        import argparse
        from make_agent.main import _apply_last_session_defaults

        args = argparse.Namespace(
            model="gpt-4o",
            reasoning_effort="low",
            max_tokens=4096,
            max_tool_output=32000,
            tool_timeout=600,
            prompt_cache=False,
        )
        last = {"model": "claude-opus", "reasoning_effort": "high"}
        _apply_last_session_defaults(args, last, provided=frozenset({"model"}))
        assert args.model == "gpt-4o"  # unchanged — was explicitly provided

    def test_fills_reasoning_effort_when_not_provided(self):
        import argparse
        from make_agent.main import _apply_last_session_defaults

        args = argparse.Namespace(
            model="gpt-4o",
            reasoning_effort="medium",
            max_tokens=4096,
            max_tool_output=32000,
            tool_timeout=600,
            prompt_cache=False,
        )
        last = {"model": "gpt-4o", "reasoning_effort": "xhigh"}
        _apply_last_session_defaults(args, last, provided=frozenset())
        assert args.reasoning_effort == "xhigh"

    def test_fills_numeric_params(self):
        import argparse
        from make_agent.main import _apply_last_session_defaults

        args = argparse.Namespace(
            model=None,
            reasoning_effort="medium",
            max_tokens=4096,
            max_tool_output=32000,
            tool_timeout=600,
            prompt_cache=False,
        )
        last = {
            "model": "m",
            "max_tokens": 8192,
            "max_tool_output": 64000,
            "tool_timeout": 300,
        }
        _apply_last_session_defaults(args, last, provided=frozenset())
        assert args.max_tokens == 8192
        assert args.max_tool_output == 64000
        assert args.tool_timeout == 300

    def test_fills_prompt_cache_from_use_prompt_cache_key(self):
        import argparse
        from make_agent.main import _apply_last_session_defaults

        args = argparse.Namespace(
            model=None,
            reasoning_effort="medium",
            max_tokens=4096,
            max_tool_output=32000,
            tool_timeout=600,
            prompt_cache=False,
        )
        last = {"model": "m", "use_prompt_cache": True}
        _apply_last_session_defaults(args, last, provided=frozenset())
        assert args.prompt_cache is True

    def test_missing_key_in_last_leaves_arg_unchanged(self):
        import argparse
        from make_agent.main import _apply_last_session_defaults

        args = argparse.Namespace(
            model=None,
            reasoning_effort="medium",
            max_tokens=4096,
            max_tool_output=32000,
            tool_timeout=600,
            prompt_cache=False,
        )
        last = {"model": "m"}  # no reasoning_effort
        _apply_last_session_defaults(args, last, provided=frozenset())
        assert args.reasoning_effort == "medium"  # default kept


# ── _dest_from_option (main.py helper) ───────────────────────────────────────


class TestDestFromOption:
    def test_long_flag_with_dashes(self):
        from make_agent.main import _dest_from_option

        assert _dest_from_option("--max-tokens") == "max_tokens"

    def test_long_flag_equals_form(self):
        from make_agent.main import _dest_from_option

        assert _dest_from_option("--max-tokens=8192") == "max_tokens"

    def test_short_flag(self):
        from make_agent.main import _dest_from_option

        assert _dest_from_option("-v") == "v"

    def test_no_dashes_in_name(self):
        from make_agent.main import _dest_from_option

        assert _dest_from_option("--model") == "model"

    def test_prompt_cache_flag(self):
        from make_agent.main import _dest_from_option

        assert _dest_from_option("--prompt-cache") == "prompt_cache"
