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
from make_agent.provider import TextDelta, UsageDelta
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


def _make_manager_with_error(exc: Exception, then_cbs: list) -> tuple[AgentManager, str]:
    """Manager whose loop raises *exc* on the first call, then succeeds with *then_cbs*."""
    tool_handler = MagicMock()
    tool_handler.is_skill_trusted = MagicMock(return_value=True)
    tool_handler.execute = AsyncMock(return_value=get_tool_result("ok", "", 0))

    manager = AgentManager(tool_handler)
    session_id = manager.get_session_id()

    call_count = 0

    async def _fake_astream(msg: str):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise exc
        for cb in then_cbs:
            yield cb

    loop_mock = MagicMock(spec=AgenticLoop)
    loop_mock.astream = _fake_astream
    loop_mock._max_tool_output = 0
    # Give the mock a real compact_history implementation
    loop_mock._messages = [_sys("system"), _user("a"), _assistant("b"), _user("c"), _assistant("d")]
    loop_mock.compact_history = lambda: _compact(loop_mock)
    manager._sessions[session_id] = loop_mock
    return manager, session_id


def _compact(loop_mock) -> int:
    """Standalone compact_history() applied to loop_mock._messages."""
    msgs = loop_mock._messages
    system = [m for m in msgs if m.get("role") == "system"]
    non_system = [m for m in msgs if m.get("role") != "system"]
    turns: list[list[dict]] = []
    for msg in non_system:
        if msg.get("role") == "user":
            turns.append([msg])
        elif turns:
            turns[-1].append(msg)
    if len(turns) <= 1:
        return 0
    import math
    keep = math.ceil(len(turns) / 2)
    kept = [m for turn in turns[-keep:] for m in turn]
    old_len = len(loop_mock._messages)
    loop_mock._messages = system + kept
    return old_len - len(loop_mock._messages)


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
        exc = _context_exceeded_error()
        manager, sid = _make_manager_with_error(exc, [MessageCallback("recovered")])
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
        exc = RuntimeError("some other error")
        manager, sid = _make_manager_with_error(exc, [])
        with pytest.raises(RuntimeError, match="some other error"):
            async for _ in manager.astream_events(sid, "hello"):
                pass

    @pytest.mark.asyncio
    async def test_compact_event_has_correct_messages_dropped(self):
        exc = _context_exceeded_error()
        manager, sid = _make_manager_with_error(exc, [MessageCallback("ok")])
        events = []
        async for event in manager.astream_events(sid, "hello"):
            events.append(event)
        compact_events = [e for e in events if isinstance(e, CompactEvent)]
        assert compact_events[0].messages_dropped > 0

    @pytest.mark.asyncio
    async def test_reraises_when_compact_drops_nothing(self):
        """If compact_history() returns 0, the error is re-raised."""
        exc = _context_exceeded_error()
        tool_handler = MagicMock()
        tool_handler.is_skill_trusted = MagicMock(return_value=True)
        tool_handler.execute = AsyncMock(return_value=get_tool_result("ok", "", 0))

        manager = AgentManager(tool_handler)
        sid = manager.get_session_id()

        async def _always_raises(msg: str):
            raise exc
            yield  # make it an async generator  # noqa: unreachable

        loop_mock = MagicMock(spec=AgenticLoop)
        loop_mock.astream = _always_raises
        loop_mock._max_tool_output = 0
        # Only one turn → compact_history returns 0
        loop_mock._messages = [_sys("sys"), _user("only"), _assistant("reply")]
        loop_mock.compact_history = lambda: 0
        manager._sessions[sid] = loop_mock

        from make_agent.provider import is_context_exceeded
        with pytest.raises(Exception) as exc_info:
            async for _ in manager.astream_events(sid, "hello"):
                pass
        assert is_context_exceeded(exc_info.value)
