"""Tests for auto-compact: AgenticLoop.compact_history() and AgentManager retry."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from make_agent.agent_core import (
    AgentConfig,
    AgentManager,
    AgenticLoop,
    CompactEvent,
    DoneEvent,
    MessageCallback,
    TokenCallback,
)
from make_agent.provider import ContextExceededChunk, TextDelta, UsageDelta
from make_agent.tool_handler.runner import get_tool_result


# ── helpers ───────────────────────────────────────────────────────────────────


def _sys(content: str) -> dict:
    return {"role": "system", "content": content}


def _user(content: str) -> dict:
    return {"role": "user", "content": content}


def _assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}


def _make_loop(messages: list[dict]) -> AgenticLoop:
    """Create an AgenticLoop with pre-seeded messages (bypasses __init__ system prompt)."""
    tool_handler = MagicMock()
    tool_handler.tool_names = []
    tool_handler.llm_tool_kwargs = {"tools": []}
    tool_handler.schemas = []

    class _FakeProvider:
        async def astream(self, *a, **kw):
            if False:
                yield  # make it an async generator

    config = AgentConfig(system_prompt="", model="claude-3-5-haiku-20241022", provider=_FakeProvider())
    loop = AgenticLoop(config, tool_handler)
    loop._messages = list(messages)
    return loop


def _make_tool_handler() -> MagicMock:
    th = MagicMock()
    th.is_skill_trusted = MagicMock(return_value=True)
    th.tool_names = []
    th.llm_tool_kwargs = {"tools": []}
    th.schemas = []
    th.execute = AsyncMock(return_value=get_tool_result("ok", "", 0))
    return th


def _make_manager_with_error(success_text: str) -> tuple[AgentManager, str]:
    """Manager whose first LLM call yields ContextExceededChunk, then succeeds returning *success_text*.

    Seeds the session with prior conversation history so there is something to drop.
    """
    call_count = 0

    class _FakeProvider:
        async def astream(self, *a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield ContextExceededChunk()
                return
            yield TextDelta(text=success_text)

    config = AgentConfig(
        system_prompt="system",
        model="claude-3-5-haiku-20241022",
        provider=_FakeProvider(),
    )
    manager = AgentManager(_make_tool_handler())
    session_id = manager.create_session(config)
    loop = manager.get_agent(session_id)
    loop._messages.extend([
        _user("a"), _assistant("b"),
        _user("c"), _assistant("d"),
    ])
    return manager, session_id


def _context_exceeded_error() -> Exception:
    exc = Exception("context window exceeded the limit")
    exc.status_code = 400  # type: ignore[attr-defined]
    return exc


# ── compact_history() unit tests ──────────────────────────────────────────────


class TestCompactHistory:
    def test_system_messages_preserved(self):
        loop = _make_loop([
            _sys("system prompt"),
            _user("hello"), _assistant("hi"),
            _user("bye"), _assistant("see ya"),
        ])
        loop.compact_history()
        assert loop._messages[0] == _sys("system prompt")

    def test_returns_zero_when_only_one_turn(self):
        loop = _make_loop([_sys("sys"), _user("only"), _assistant("one")])
        result = loop.compact_history()
        assert result == 0

    def test_returns_zero_when_no_turns(self):
        loop = _make_loop([_sys("sys")])
        result = loop.compact_history()
        assert result == 0

    def test_drops_oldest_half_of_turns(self):
        # 4 turns → keep ceil(4/2) = 2 most recent
        loop = _make_loop([
            _sys("sys"),
            _user("t1"), _assistant("r1"),
            _user("t2"), _assistant("r2"),
            _user("t3"), _assistant("r3"),
            _user("t4"), _assistant("r4"),
        ])
        dropped = loop.compact_history()
        assert dropped == 4  # t1+r1 and t2+r2 removed
        non_sys = [m for m in loop._messages if m.get("role") != "system"]
        assert non_sys[0] == _user("t3")
        assert non_sys[-1] == _assistant("r4")

    def test_two_turns_drops_oldest_one(self):
        loop = _make_loop([
            _sys("sys"),
            _user("first"), _assistant("first reply"),
            _user("second"), _assistant("second reply"),
        ])
        dropped = loop.compact_history()
        assert dropped == 2
        non_sys = [m for m in loop._messages if m.get("role") != "system"]
        assert non_sys[0] == _user("second")

    def test_odd_turns_keeps_ceiling(self):
        # 3 turns → keep ceil(3/2) = 2
        loop = _make_loop([
            _sys("sys"),
            _user("1"), _assistant("a"),
            _user("2"), _assistant("b"),
            _user("3"), _assistant("c"),
        ])
        dropped = loop.compact_history()
        assert dropped == 2
        non_sys = [m for m in loop._messages if m.get("role") != "system"]
        assert non_sys[0] == _user("2")

    def test_returns_messages_removed_count(self):
        loop = _make_loop([
            _sys("sys"),
            _user("a"), _assistant("b"),
            _user("c"), _assistant("d"),
        ])
        before = len(loop._messages)
        dropped = loop.compact_history()
        assert dropped == before - len(loop._messages)

    def test_multiple_system_messages_all_preserved(self):
        loop = _make_loop([
            _sys("sys1"),
            _sys("sys2"),
            _user("a"), _assistant("b"),
            _user("c"), _assistant("d"),
        ])
        loop.compact_history()
        sys_msgs = [m for m in loop._messages if m.get("role") == "system"]
        assert len(sys_msgs) == 2


# ── AgentManager compact-and-retry tests ─────────────────────────────────────


class TestCompactRetry:
    @pytest.mark.asyncio
    async def test_retries_after_context_exceeded(self):
        manager, sid = _make_manager_with_error("recovered")
        events = []
        async for event in manager.astream_events(sid, "hello"):
            events.append(event)

        compact_events = [e for e in events if isinstance(e, CompactEvent)]
        done_events = [e for e in events if isinstance(e, DoneEvent)]
        assert len(compact_events) == 1
        assert compact_events[0].attempt == 1
        assert compact_events[0].messages_dropped > 0
        assert len(done_events) == 1
        assert done_events[0].content == "recovered"

    @pytest.mark.asyncio
    async def test_non_context_error_propagates_immediately(self):
        """A plain exception from the provider still propagates to the caller."""
        call_count = 0

        class _FakeProvider:
            async def astream(self, *a, **kw):
                nonlocal call_count
                call_count += 1
                raise RuntimeError("some other error")
                yield  # noqa: unreachable

        config = AgentConfig(
            system_prompt="system",
            model="claude-3-5-haiku-20241022",
            provider=_FakeProvider(),
        )
        manager = AgentManager(_make_tool_handler())
        sid = manager.create_session(config)
        with pytest.raises(RuntimeError, match="some other error"):
            async for _ in manager.astream_events(sid, "hello"):
                pass

    @pytest.mark.asyncio
    async def test_compact_event_has_correct_messages_dropped(self):
        manager, sid = _make_manager_with_error("ok")
        events = []
        async for event in manager.astream_events(sid, "hello"):
            events.append(event)
        compact_events = [e for e in events if isinstance(e, CompactEvent)]
        assert compact_events[0].messages_dropped > 0

    @pytest.mark.asyncio
    async def test_loop_reused_after_compact(self):
        """After compaction the session reuses the same AgenticLoop instance."""
        manager, sid = _make_manager_with_error("ok")
        original_loop = manager.get_agent(sid)
        async for _ in manager.astream_events(sid, "hello"):
            pass
        assert manager.get_agent(sid) is original_loop

    @pytest.mark.asyncio
    async def test_reraises_when_compact_drops_nothing(self):
        """If compact_history() returns 0 (no compactable turns), a RuntimeError is raised."""

        class _FakeProvider:
            async def astream(self, *a, **kw):
                yield ContextExceededChunk()

        config = AgentConfig(
            system_prompt="sys",
            model="claude-3-5-haiku-20241022",
            provider=_FakeProvider(),
        )
        manager = AgentManager(_make_tool_handler())
        session_id = manager.create_session(config)
        # No extra history — only the system prompt in _messages

        with pytest.raises(RuntimeError, match="no messages can be compacted"):
            async for _ in manager.astream_events(session_id, "hello"):
                pass

