"""Tests for AgenticLoop.truncate_to_turn()."""

from __future__ import annotations

import pytest
from make_agent.agent_core import AgentConfig, AgenticLoop
from make_agent.memory import Memory
from make_agent.tool_handler import ToolHandler


def _make_loop(tmp_path, system_prompt: str = "You are a helper.") -> AgenticLoop:
    from unittest.mock import MagicMock

    memory = Memory(tmp_path / "memory.db")
    tool_handler = ToolHandler(str(tmp_path), memory, base_dir=tmp_path)
    mock_provider = MagicMock()
    return AgenticLoop(
        AgentConfig(
            system_prompt=system_prompt,
            model="anthropic/claude-3-5-haiku-20241022",
            skills_dir=str(tmp_path),
            provider=mock_provider,
        ),
        tool_handler,
    )


class TestTruncateToTurn:
    def test_basic_three_turn_truncate_to_two(self, tmp_path):
        loop = _make_loop(tmp_path)
        # Simulate 3 turns
        loop._messages = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "reply 1"},
            {"role": "user", "content": "turn 2"},
            {"role": "assistant", "content": "reply 2"},
            {"role": "user", "content": "turn 3"},
            {"role": "assistant", "content": "reply 3"},
        ]
        removed = loop.truncate_to_turn(2)
        assert removed == 2
        assert loop._messages == [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "reply 1"},
            {"role": "user", "content": "turn 2"},
            {"role": "assistant", "content": "reply 2"},
        ]

    def test_truncate_preserves_tool_calls_and_results(self, tmp_path):
        """Tool call messages and tool result messages must stay with their turn."""
        loop = _make_loop(tmp_path)
        loop._messages = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "read", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "tc1", "content": "file contents"},
            {"role": "assistant", "content": "here's the file"},
            {"role": "user", "content": "turn 2"},
            {"role": "assistant", "content": "reply 2"},
        ]
        removed = loop.truncate_to_turn(1)
        assert removed == 2  # turn 2: user + assistant = 2 messages
        assert loop._messages == [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "read", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "tc1", "content": "file contents"},
            {"role": "assistant", "content": "here's the file"},
        ]

    def test_truncate_to_one(self, tmp_path):
        loop = _make_loop(tmp_path)
        loop._messages = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "reply 1"},
            {"role": "user", "content": "turn 2"},
            {"role": "assistant", "content": "reply 2"},
        ]
        removed = loop.truncate_to_turn(1)
        assert removed == 2
        assert loop._messages == [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "reply 1"},
        ]

    def test_truncate_to_all_turns_is_noop(self, tmp_path):
        loop = _make_loop(tmp_path)
        original = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "reply 1"},
        ]
        loop._messages = list(original)
        removed = loop.truncate_to_turn(1)
        assert removed == 0
        assert loop._messages == original

    def test_truncate_to_zero_raises(self, tmp_path):
        loop = _make_loop(tmp_path)
        loop._messages = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "reply 1"},
        ]
        with pytest.raises(ValueError, match="out of range"):
            loop.truncate_to_turn(0)

    def test_truncate_beyond_range_raises(self, tmp_path):
        loop = _make_loop(tmp_path)
        loop._messages = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "reply 1"},
        ]
        with pytest.raises(ValueError, match="out of range"):
            loop.truncate_to_turn(5)

    def test_truncate_preserves_multiple_system_messages(self, tmp_path):
        """Compact may inject system messages; they must all be preserved."""
        loop = _make_loop(tmp_path)
        loop._messages = [
            {"role": "system", "content": "You are a helper."},
            {"role": "system", "content": "Prior conversation summary: stuff"},
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "reply 1"},
            {"role": "user", "content": "turn 2"},
            {"role": "assistant", "content": "reply 2"},
        ]
        removed = loop.truncate_to_turn(1)
        assert removed == 2
        assert loop._messages == [
            {"role": "system", "content": "You are a helper."},
            {"role": "system", "content": "Prior conversation summary: stuff"},
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "reply 1"},
        ]

    def test_truncate_with_no_system_messages(self, tmp_path):
        loop = _make_loop(tmp_path, system_prompt="")
        loop._messages = [
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "reply 1"},
            {"role": "user", "content": "turn 2"},
            {"role": "assistant", "content": "reply 2"},
        ]
        removed = loop.truncate_to_turn(1)
        assert removed == 2
        assert loop._messages == [
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "reply 1"},
        ]

    def test_truncate_empty_messages_raises(self, tmp_path):
        loop = _make_loop(tmp_path)
        loop._messages = []
        with pytest.raises(ValueError, match="out of range"):
            loop.truncate_to_turn(1)

    def test_truncate_system_only_raises(self, tmp_path):
        """Only system messages, no turns — truncation is invalid."""
        loop = _make_loop(tmp_path)
        loop._messages = [
            {"role": "system", "content": "You are a helper."},
        ]
        with pytest.raises(ValueError, match="out of range"):
            loop.truncate_to_turn(1)
