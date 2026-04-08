"""Tests for the Memory class and memory search built-in tools."""

from __future__ import annotations

import argparse
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from make_agent.memory import Memory
from make_agent.builtin_tools import get_builtin_tools, get_memory_schemas


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mem(tmp_path):
    """Return a fresh Memory instance backed by a temp DB."""
    m = Memory(tmp_path / "memory.db")
    yield m
    m.close()


# ── Schema & initialisation ───────────────────────────────────────────────────

class TestMemorySchema:
    def test_db_file_created_on_first_use(self, tmp_path):
        db_path = tmp_path / "memory.db"
        assert not db_path.exists()
        m = Memory(db_path)
        m.store("user", "hello")
        assert db_path.exists()
        m.close()

    def test_parent_dirs_created(self, tmp_path):
        db_path = tmp_path / "deep" / "nested" / "memory.db"
        m = Memory(db_path)
        m.store("user", "x")
        assert db_path.exists()
        m.close()

    def test_messages_table_columns(self, mem):
        conn = mem._get_conn()
        info = conn.execute("PRAGMA table_info(messages)").fetchall()
        col_names = [row[1] for row in info]
        assert "id" in col_names
        assert "created_at" in col_names
        assert "sender" in col_names
        assert "message" in col_names

    def test_fts_table_exists(self, mem):
        conn = mem._get_conn()
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "messages_fts" in tables

    def test_user_memory_view_exists(self, mem):
        conn = mem._get_conn()
        views = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='view'")}
        assert "user_memory" in views

    def test_agent_memory_view_exists(self, mem):
        conn = mem._get_conn()
        views = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='view'")}
        assert "agent_memory" in views

    def test_idempotent_init(self, tmp_path):
        """Opening the same DB twice does not raise."""
        db_path = tmp_path / "memory.db"
        m1 = Memory(db_path)
        m1.store("user", "first")
        m1.close()
        m2 = Memory(db_path)
        m2.store("agent", "second")
        conn = m2._get_conn()
        count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        assert count == 2
        m2.close()


# ── store() ───────────────────────────────────────────────────────────────────

class TestMemoryStore:
    def test_store_user_message(self, mem):
        mem.store("user", "hello world")
        conn = mem._get_conn()
        row = conn.execute("SELECT sender, message FROM messages").fetchone()
        assert row["sender"] == "user"
        assert row["message"] == "hello world"

    def test_store_agent_message(self, mem):
        mem.store("agent", "I can help")
        conn = mem._get_conn()
        row = conn.execute("SELECT sender FROM messages").fetchone()
        assert row["sender"] == "agent"

    def test_store_multiple_messages(self, mem):
        mem.store("user", "first")
        mem.store("agent", "second")
        mem.store("user", "third")
        conn = mem._get_conn()
        count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        assert count == 3

    def test_created_at_auto_populated(self, mem):
        mem.store("user", "test")
        conn = mem._get_conn()
        row = conn.execute("SELECT created_at FROM messages").fetchone()
        assert row["created_at"] is not None
        assert "T" in row["created_at"]  # ISO 8601 format

    def test_invalid_sender_raises(self, mem):
        with pytest.raises(sqlite3.IntegrityError):
            mem._get_conn().execute(
                "INSERT INTO messages (sender, message) VALUES (?, ?)",
                ("robot", "hello"),
            )


# ── Views ─────────────────────────────────────────────────────────────────────

class TestMemoryViews:
    def test_user_memory_filters_user(self, mem):
        mem.store("user", "user msg")
        mem.store("agent", "agent msg")
        conn = mem._get_conn()
        rows = conn.execute("SELECT * FROM user_memory").fetchall()
        assert len(rows) == 1
        assert rows[0]["sender"] == "user"

    def test_agent_memory_filters_agent(self, mem):
        mem.store("user", "user msg")
        mem.store("agent", "agent msg")
        conn = mem._get_conn()
        rows = conn.execute("SELECT * FROM agent_memory").fetchall()
        assert len(rows) == 1
        assert rows[0]["sender"] == "agent"


# ── FTS5 triggers ─────────────────────────────────────────────────────────────

class TestFTSTriggers:
    def test_insert_trigger_indexes_message(self, mem):
        mem.store("user", "unique phrase xyzzy")
        conn = mem._get_conn()
        rows = conn.execute(
            "SELECT rowid FROM messages_fts WHERE messages_fts MATCH ?",
            ("xyzzy",),
        ).fetchall()
        assert len(rows) == 1

    def test_delete_trigger_removes_from_index(self, mem):
        mem.store("user", "unique phrase xyzzy")
        conn = mem._get_conn()
        conn.execute("DELETE FROM messages WHERE message = 'unique phrase xyzzy'")
        conn.commit()
        rows = conn.execute(
            "SELECT rowid FROM messages_fts WHERE messages_fts MATCH ?",
            ("xyzzy",),
        ).fetchall()
        assert len(rows) == 0

    def test_update_trigger_reindexes(self, mem):
        mem.store("user", "old phrase aaa")
        conn = mem._get_conn()
        row_id = conn.execute("SELECT id FROM messages").fetchone()[0]
        conn.execute("UPDATE messages SET message = 'new phrase bbb' WHERE id = ?", (row_id,))
        conn.commit()
        old_hits = conn.execute(
            "SELECT rowid FROM messages_fts WHERE messages_fts MATCH ?", ("aaa",)
        ).fetchall()
        new_hits = conn.execute(
            "SELECT rowid FROM messages_fts WHERE messages_fts MATCH ?", ("bbb",)
        ).fetchall()
        assert len(old_hits) == 0
        assert len(new_hits) == 1


# ── search_user / search_agent ────────────────────────────────────────────────

class TestMemorySearch:
    def test_search_user_finds_match(self, mem):
        mem.store("user", "how do I list files")
        mem.store("agent", "use ls command")
        result = mem.search_user("list files")
        assert "how do I list files" in result

    def test_search_user_excludes_agent(self, mem):
        mem.store("user", "how do I list files")
        mem.store("agent", "use ls command")
        result = mem.search_user("ls command")
        assert "No results found" in result

    def test_search_agent_finds_match(self, mem):
        mem.store("user", "how do I list files")
        mem.store("agent", "use ls command")
        result = mem.search_agent("ls command")
        assert "use ls command" in result

    def test_search_agent_excludes_user(self, mem):
        mem.store("user", "how do I list files")
        mem.store("agent", "use ls command")
        result = mem.search_agent("list files")
        assert "No results found" in result

    def test_search_no_results(self, mem):
        mem.store("user", "something unrelated")
        result = mem.search_user("nonexistent_term_xyz")
        assert result == "No results found."

    def test_search_limit(self, mem):
        for i in range(10):
            mem.store("user", f"make agent test message {i}")
        result = mem.search_user("make agent", limit=3)
        lines = [l for l in result.splitlines() if l.strip()]
        assert len(lines) <= 3

    def test_search_from_date_filter(self, mem):
        conn = mem._get_conn()
        conn.execute(
            "INSERT INTO messages (created_at, sender, message) VALUES (?, ?, ?)",
            ("2026-01-01T00:00:00Z", "user", "old message about python"),
        )
        conn.execute(
            "INSERT INTO messages (created_at, sender, message) VALUES (?, ?, ?)",
            ("2026-03-01T00:00:00Z", "user", "new message about python"),
        )
        conn.commit()
        result = mem.search_user("python", from_date="2026-02-01")
        assert "new message about python" in result
        assert "old message about python" not in result

    def test_search_to_date_filter(self, mem):
        conn = mem._get_conn()
        conn.execute(
            "INSERT INTO messages (created_at, sender, message) VALUES (?, ?, ?)",
            ("2026-01-01T00:00:00Z", "user", "old message about python"),
        )
        conn.execute(
            "INSERT INTO messages (created_at, sender, message) VALUES (?, ?, ?)",
            ("2026-03-01T00:00:00Z", "user", "new message about python"),
        )
        conn.commit()
        result = mem.search_user("python", to_date="2026-02-01")
        assert "old message about python" in result
        assert "new message about python" not in result

    def test_search_date_range_filter(self, mem):
        conn = mem._get_conn()
        for date, msg in [
            ("2026-01-01T00:00:00Z", "january message"),
            ("2026-02-15T00:00:00Z", "february message"),
            ("2026-03-20T00:00:00Z", "march message"),
        ]:
            conn.execute(
                "INSERT INTO messages (created_at, sender, message) VALUES (?, ?, ?)",
                (date, "user", msg),
            )
        conn.commit()
        result = mem.search_user("message", from_date="2026-02-01", to_date="2026-03-01")
        assert "february message" in result
        assert "january message" not in result
        assert "march message" not in result

    def test_result_format_includes_timestamp(self, mem):
        mem.store("user", "test query message")
        result = mem.search_user("test query")
        assert "[" in result and "]" in result  # [created_at] message format


# ── recent() ──────────────────────────────────────────────────────────────────

class TestMemoryRecent:
    def test_returns_messages_in_chronological_order(self, mem):
        mem.store("user", "first")
        mem.store("agent", "second")
        mem.store("user", "third")
        result = mem.recent(10)
        assert result.index("first") < result.index("second") < result.index("third")

    def test_limit_respected(self, mem):
        for i in range(5):
            mem.store("user", f"msg {i}")
        result = mem.recent(3)
        lines = [l for l in result.splitlines() if l.strip()]
        assert len(lines) == 3

    def test_returns_most_recent_when_limited(self, mem):
        for i in range(5):
            mem.store("user", f"msg {i}")
        result = mem.recent(2)
        assert "msg 4" in result
        assert "msg 3" in result
        assert "msg 0" not in result

    def test_includes_sender_in_output(self, mem):
        mem.store("user", "hello")
        mem.store("agent", "hi back")
        result = mem.recent()
        assert "user" in result
        assert "agent" in result

    def test_empty_memory_returns_message(self, mem):
        result = mem.recent()
        assert result == "No messages found."

    def test_default_limit_is_ten(self, mem):
        for i in range(15):
            mem.store("user", f"msg {i}")
        result = mem.recent()
        lines = [l for l in result.splitlines() if l.strip()]
        assert len(lines) == 10

    def test_from_date_filter(self, mem):
        conn = mem._get_conn()
        for date, msg in [
            ("2026-01-01T00:00:00Z", "january message"),
            ("2026-03-01T00:00:00Z", "march message"),
        ]:
            conn.execute(
                "INSERT INTO messages (created_at, sender, message) VALUES (?, ?, ?)",
                (date, "user", msg),
            )
        conn.commit()
        result = mem.recent(10, from_date="2026-02-01")
        assert "march message" in result
        assert "january message" not in result

    def test_to_date_filter(self, mem):
        conn = mem._get_conn()
        for date, msg in [
            ("2026-01-01T00:00:00Z", "january message"),
            ("2026-03-01T00:00:00Z", "march message"),
        ]:
            conn.execute(
                "INSERT INTO messages (created_at, sender, message) VALUES (?, ?, ?)",
                (date, "user", msg),
            )
        conn.commit()
        result = mem.recent(10, to_date="2026-02-01")
        assert "january message" in result
        assert "march message" not in result

    def test_date_range_and_limit_combined(self, mem):
        conn = mem._get_conn()
        for i, date in enumerate([
            "2026-02-01T00:00:00Z",
            "2026-02-02T00:00:00Z",
            "2026-02-03T00:00:00Z",
            "2026-02-04T00:00:00Z",
        ]):
            conn.execute(
                "INSERT INTO messages (created_at, sender, message) VALUES (?, ?, ?)",
                (date, "user", f"feb msg {i}"),
            )
        conn.commit()
        result = mem.recent(2, from_date="2026-02-01", to_date="2026-02-28")
        lines = [l for l in result.splitlines() if l.strip()]
        assert len(lines) == 2
        # Should be the 2 most recent within range
        assert "feb msg 3" in result
        assert "feb msg 2" in result
        assert "feb msg 0" not in result


# ── Built-in tools integration ────────────────────────────────────────────────

class TestMemoryBuiltinTools:
    def test_memory_schemas_returned(self):
        schemas = get_memory_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert "search_user_memory" in names
        assert "search_agent_memory" in names
        assert "get_recent_messages" in names

    def test_memory_schemas_have_required_query(self):
        schemas = get_memory_schemas()
        for schema in schemas:
            params = schema["function"]["parameters"]
            if schema["function"]["name"] in ("search_user_memory", "search_agent_memory"):
                assert "query" in params["required"]

    def test_memory_schemas_have_optional_params(self):
        schemas = get_memory_schemas()
        for schema in schemas:
            props = schema["function"]["parameters"]["properties"]
            if schema["function"]["name"] in ("search_user_memory", "search_agent_memory"):
                assert "limit" in props
                assert "from_date" in props
                assert "to_date" in props

    def test_search_user_memory_tool_callable(self, mem):
        mem.store("user", "remember this phrase")
        tools = get_builtin_tools("agents_dir", memory=mem)
        assert "search_user_memory" in tools
        result = tools["search_user_memory"](query="remember this phrase")
        assert "remember this phrase" in result

    def test_search_agent_memory_tool_callable(self, mem):
        mem.store("agent", "I can recall things")
        tools = get_builtin_tools("agents_dir", memory=mem)
        assert "search_agent_memory" in tools
        result = tools["search_agent_memory"](query="recall things")
        assert "I can recall things" in result

    def test_no_memory_tools_without_memory(self):
        tools = get_builtin_tools("agents_dir")
        assert "search_user_memory" not in tools
        assert "search_agent_memory" not in tools
        assert "get_recent_messages" not in tools

    def test_no_memory_schemas_not_injected_without_memory(self):
        schemas = get_memory_schemas()
        assert len(schemas) == 3

    def test_get_recent_messages_tool_callable(self, mem):
        mem.store("user", "first message")
        mem.store("agent", "first reply")
        tools = get_builtin_tools("agents_dir", memory=mem)
        assert "get_recent_messages" in tools
        result = tools["get_recent_messages"](limit=5)
        assert "first message" in result
        assert "first reply" in result

    def test_get_recent_messages_schema_has_date_params(self):
        schemas = get_memory_schemas()
        schema = next(s for s in schemas if s["function"]["name"] == "get_recent_messages")
        props = schema["function"]["parameters"]["properties"]
        assert "limit" in props
        assert "from_date" in props
        assert "to_date" in props


# ── Agent auto-storage ────────────────────────────────────────────────────────

class TestAgentAutoStorage:
    """Verify Agent.__call__ writes to memory automatically."""

    def _make_agent(self, tmp_path, mem):
        from make_agent.agent import Agent, AgentConfig
        mf_path = tmp_path / "Makefile"
        mf_path.write_text("noop:\n\t@echo ok\n")
        config = AgentConfig(makefile_path=mf_path, memory=mem)
        return Agent(config)

    def test_user_message_stored(self, tmp_path, mem):
        agent = self._make_agent(tmp_path, mem)
        fake_response = MagicMock()
        fake_response.choices[0].message.tool_calls = None
        fake_response.choices[0].message.content = "the reply"

        with patch("make_agent.agent._completion_with_retry", return_value=fake_response):
            agent("hello from user")

        conn = mem._get_conn()
        row = conn.execute("SELECT sender, message FROM messages WHERE sender='user'").fetchone()
        assert row is not None
        assert row["message"] == "hello from user"

    def test_agent_reply_stored(self, tmp_path, mem):
        agent = self._make_agent(tmp_path, mem)
        fake_response = MagicMock()
        fake_response.choices[0].message.tool_calls = None
        fake_response.choices[0].message.content = "the reply"

        with patch("make_agent.agent._completion_with_retry", return_value=fake_response):
            agent("hello from user")

        conn = mem._get_conn()
        row = conn.execute("SELECT sender, message FROM messages WHERE sender='agent'").fetchone()
        assert row is not None
        assert row["message"] == "the reply"

    def test_no_storage_without_memory(self, tmp_path):
        from make_agent.agent import Agent, AgentConfig
        mf_path = tmp_path / "Makefile"
        mf_path.write_text("noop:\n\t@echo ok\n")
        config = AgentConfig(makefile_path=mf_path, memory=None)
        agent = Agent(config)

        fake_response = MagicMock()
        fake_response.choices[0].message.tool_calls = None
        fake_response.choices[0].message.content = "the reply"

        with patch("make_agent.agent._completion_with_retry", return_value=fake_response):
            agent("hello")
        # No exception — memory is simply not used


# ── CLI flag wiring ───────────────────────────────────────────────────────────

class TestWithMemoryFlag:
    def test_with_memory_flag_creates_memory_instance(self, tmp_path):
        import make_agent.main as main_module

        mf = tmp_path / "Makefile"
        mf.write_text("noop:\n\t@echo ok\n")

        args = argparse.Namespace(
            file=str(mf),
            model="model-x",
            prompt="hello",
            prompt_file=None,
            debug=False,
            max_retries=5,
            tool_timeout=600,
            max_tool_output=20000,
            max_tokens=4096,
            agents_dir=None,
            with_memory=True,
            disable_builtin_tools=None,
            reasoning_effort=None,
        )

        captured: dict = {}

        def _fake_run(**kwargs):
            captured.update(kwargs)

        original = main_module.run
        main_module.run = _fake_run
        try:
            with patch("make_agent.main.project_dir", return_value=tmp_path):
                main_module._cmd_run(args)
        finally:
            main_module.run = original

        assert captured.get("memory") is not None
        from make_agent.memory import Memory
        assert isinstance(captured["memory"], Memory)

    def test_without_memory_flag_passes_none(self, tmp_path):
        import make_agent.main as main_module

        mf = tmp_path / "Makefile"
        mf.write_text("noop:\n\t@echo ok\n")

        args = argparse.Namespace(
            file=str(mf),
            model="model-x",
            prompt="hello",
            prompt_file=None,
            debug=False,
            max_retries=5,
            tool_timeout=600,
            max_tool_output=20000,
            max_tokens=4096,
            agents_dir=None,
            with_memory=False,
            disable_builtin_tools=None,
            reasoning_effort=None,
        )

        captured: dict = {}

        def _fake_run(**kwargs):
            captured.update(kwargs)

        original = main_module.run
        main_module.run = _fake_run
        try:
            main_module._cmd_run(args)
        finally:
            main_module.run = original

        assert captured.get("memory") is None

    def test_settings_memory_true_enables_memory(self, tmp_path):
        import make_agent.main as main_module

        mf = tmp_path / "Makefile"
        mf.write_text("noop:\n\t@echo ok\n")

        args = argparse.Namespace(
            file=str(mf),
            model="model-x",
            prompt="hello",
            prompt_file=None,
            debug=False,
            max_retries=5,
            tool_timeout=600,
            max_tool_output=20000,
            max_tokens=4096,
            agents_dir=None,
            with_memory=False,  # not set via CLI
            disable_builtin_tools=None,
            reasoning_effort=None,
        )

        captured: dict = {}

        def _fake_run(**kwargs):
            captured.update(kwargs)

        original = main_module.run
        main_module.run = _fake_run
        try:
            with patch("make_agent.main.load_settings", return_value={"model": "x", "memory": True}), \
                 patch("make_agent.main.project_dir", return_value=tmp_path):
                main_module._cmd_run(args)
        finally:
            main_module.run = original

        from make_agent.memory import Memory
        assert isinstance(captured.get("memory"), Memory)


# ── Token usage ───────────────────────────────────────────────────────────────

class TestTokenUsage:
    def test_token_usage_table_columns(self, mem):
        conn = mem._get_conn()
        info = conn.execute("PRAGMA table_info(token_usage)").fetchall()
        col_names = [row[1] for row in info]
        assert "id" in col_names
        assert "created_at" in col_names
        assert "session_id" in col_names
        assert "agent" in col_names
        assert "model" in col_names
        assert "input_tokens" in col_names
        assert "output_tokens" in col_names

    def test_record_token_usage_inserts_row(self, mem):
        mem.record_token_usage("sess-1", "agent.mk", "openai/gpt-4o", 100, 50)
        conn = mem._get_conn()
        rows = conn.execute("SELECT * FROM token_usage").fetchall()
        assert len(rows) == 1
        row = rows[0]
        assert row["session_id"] == "sess-1"
        assert row["agent"] == "agent.mk"
        assert row["model"] == "openai/gpt-4o"
        assert row["input_tokens"] == 100
        assert row["output_tokens"] == 50

    def test_record_token_usage_multiple_rows(self, mem):
        mem.record_token_usage("sess-1", "a.mk", "model-a", 10, 5)
        mem.record_token_usage("sess-1", "a.mk", "model-a", 20, 10)
        conn = mem._get_conn()
        count = conn.execute("SELECT COUNT(*) FROM token_usage WHERE session_id='sess-1'").fetchone()[0]
        assert count == 2

    def test_get_session_stats_empty_when_no_rows(self, mem):
        assert mem.get_session_stats("nonexistent-session") == {}

    def test_get_session_stats_aggregates_totals(self, mem):
        mem.record_token_usage("sess-1", "a.mk", "model-a", 100, 40)
        mem.record_token_usage("sess-1", "a.mk", "model-a", 200, 60)
        stats = mem.get_session_stats("sess-1")
        assert stats["input_tokens"] == 300
        assert stats["output_tokens"] == 100
        assert stats["total_tokens"] == 400

    def test_get_session_stats_includes_model(self, mem):
        mem.record_token_usage("sess-1", "a.mk", "openai/gpt-4o", 10, 5)
        stats = mem.get_session_stats("sess-1")
        assert stats["models"] == ["openai/gpt-4o"]

    def test_get_session_stats_multiple_models(self, mem):
        mem.record_token_usage("sess-1", "a.mk", "model-a", 10, 5)
        mem.record_token_usage("sess-1", "b.mk", "model-b", 20, 10)
        stats = mem.get_session_stats("sess-1")
        assert sorted(stats["models"]) == ["model-a", "model-b"]

    def test_get_session_stats_isolates_sessions(self, mem):
        mem.record_token_usage("sess-1", "a.mk", "model-a", 100, 50)
        mem.record_token_usage("sess-2", "a.mk", "model-a", 999, 999)
        stats = mem.get_session_stats("sess-1")
        assert stats["input_tokens"] == 100
        assert stats["output_tokens"] == 50
        assert stats["total_tokens"] == 150

    def test_get_session_stats_deduplicates_model_names(self, mem):
        mem.record_token_usage("sess-1", "a.mk", "model-a", 10, 5)
        mem.record_token_usage("sess-1", "a.mk", "model-a", 20, 10)
        stats = mem.get_session_stats("sess-1")
        assert stats["models"] == ["model-a"]
