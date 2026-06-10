"""Tests for UserMessagesManager and Memory.recent_user."""

from __future__ import annotations

from pathlib import Path

import pytest
from make_agent.agent_shell.user_messages import UserMessagesManager
from make_agent.memory import Memory


@pytest.fixture
def memory(tmp_path: Path) -> Memory:
    """Create an in-memory Memory instance with some test data."""
    db_path = tmp_path / "test_memory.db"
    mem = Memory(db_path)
    # Insert test user messages
    mem.store("user", "first message")
    mem.store("user", "second message")
    mem.store("agent", "agent reply")
    mem.store("user", "third message")
    mem.store("user", "fourth message")
    return mem


class TestMemoryRecentUser:
    """Tests for Memory.recent_user()."""

    def test_returns_user_messages_newest_first(self, memory: Memory) -> None:
        result = memory.recent_user(10)
        assert result == [
            "fourth message",
            "third message",
            "second message",
            "first message",
        ]

    def test_respects_limit(self, memory: Memory) -> None:
        result = memory.recent_user(2)
        assert result == ["fourth message", "third message"]

    def test_excludes_agent_messages(self, memory: Memory) -> None:
        result = memory.recent_user(10)
        assert "agent reply" not in result

    def test_empty_database(self, tmp_path: Path) -> None:
        mem = Memory(tmp_path / "empty.db")
        result = mem.recent_user(10)
        assert result == []


class TestUserMessagesManager:
    """Tests for UserMessagesManager navigation."""

    def test_previous_navigates_oldest_first(self, memory: Memory) -> None:
        mgr = UserMessagesManager(memory)
        # First UP press
        assert mgr.previous("current text") == "fourth message"
        assert mgr._index == 0
        # Second UP press
        assert mgr.previous() == "third message"
        assert mgr._index == 1
        # Third UP press
        assert mgr.previous() == "second message"
        assert mgr._index == 2
        # Fourth UP press
        assert mgr.previous() == "first message"
        assert mgr._index == 3
        # Fifth UP press — already at oldest
        assert mgr.previous() is None

    def test_next_navigates_back_and_restores_original(self, memory: Memory) -> None:
        mgr = UserMessagesManager(memory)
        # Navigate up twice
        assert mgr.previous("my current text") == "fourth message"
        assert mgr.previous() == "third message"
        # Navigate down once
        assert mgr.next() == "fourth message"
        assert mgr._index == 0
        # Navigate down again — should restore original text
        assert mgr.next() == "my current text"
        assert mgr._index == -1

    def test_submit_resets_navigation(self, memory: Memory) -> None:
        mgr = UserMessagesManager(memory)
        mgr.previous("some text")
        assert mgr.is_navigating()
        mgr.submit()
        assert not mgr.is_navigating()

    def test_refreshes_on_each_nav_start(self, memory: Memory) -> None:
        mgr = UserMessagesManager(memory)
        # Start navigating
        mgr.previous("text")
        assert len(mgr._messages) == 4
        # Add a new user message after navigation started
        memory.store("user", "fifth message")
        # Start a new navigation session — should pick up the new message
        mgr.submit()
        result = mgr.previous("new text")
        assert result == "fifth message"
        assert len(mgr._messages) == 5

    def test_empty_history_returns_none(self, tmp_path: Path) -> None:
        mem = Memory(tmp_path / "empty.db")
        mgr = UserMessagesManager(mem)
        assert mgr.previous("text") is None

    def test_next_when_not_navigating_returns_none(self, memory: Memory) -> None:
        mgr = UserMessagesManager(memory)
        assert mgr.next() is None

    def test_saves_original_text_on_first_up(self, memory: Memory) -> None:
        mgr = UserMessagesManager(memory)
        mgr.previous("original typed text")
        assert mgr._original_text == "original typed text"

    def test_limit_caps_history(self, memory: Memory) -> None:
        mgr = UserMessagesManager(memory, limit=2)
        assert mgr.previous("") == "fourth message"
        assert mgr.previous() == "third message"
        assert mgr.previous() is None  # Limit reached

    def test_cancel_when_not_navigating(self, memory: Memory) -> None:
        mgr = UserMessagesManager(memory)
        mgr.cancel()
        assert not mgr.is_navigating()
