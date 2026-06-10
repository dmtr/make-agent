"""Tests for the trusted-skill confirmation mechanism."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from make_agent.tool_handler.handler import ToolHandler
from make_agent.tool_handler.runner import get_tool_result
from make_agent.main import _parse_trusted_skills
from make_agent.agent_core import (
    AgentManager,
    AgenticLoop,
    CallBack,
    ConfirmEvent,
    MessageCallback,
    TokenCallback,
    ToolCallback,
    ToolDoneEvent,
    ToolStartEvent,
)


# ── _parse_trusted_skills ──────────────────────────────────────────────────────


def test_parse_trusted_skills_none():
    assert _parse_trusted_skills(None) == frozenset()


def test_parse_trusted_skills_empty_string():
    assert _parse_trusted_skills("") == frozenset()


def test_parse_trusted_skills_single():
    assert _parse_trusted_skills("web-fetch") == frozenset(["web-fetch"])


def test_parse_trusted_skills_multiple():
    result = _parse_trusted_skills("web-fetch,search,summarise")
    assert result == frozenset(["web-fetch", "search", "summarise"])


def test_parse_trusted_skills_strips_whitespace():
    result = _parse_trusted_skills(" web-fetch , search ")
    assert result == frozenset(["web-fetch", "search"])


def test_parse_trusted_skills_all():
    assert _parse_trusted_skills("all") == frozenset(["*"])


def test_parse_trusted_skills_all_case_insensitive():
    assert _parse_trusted_skills("ALL") == frozenset(["*"])


# ── ToolHandler.is_skill_trusted ───────────────────────────────────────────────


def _make_handler(trusted_skills: frozenset[str] = frozenset()) -> ToolHandler:
    backend = MagicMock()
    backend.schemas = []
    backend.executors = {}
    memory = MagicMock()
    memory.store = MagicMock()
    backend.schemas = []
    backend.executors = {"execute_skill": AsyncMock(return_value="ok")}
    backend.get_skill_trusted = MagicMock(return_value=None)
    return ToolHandler(backend, memory, trusted_skills=trusted_skills)


def test_is_trusted_empty_set():
    h = _make_handler(frozenset())
    assert not h.is_skill_trusted("web-fetch", "fetch")


def test_is_trusted_specific_match():
    h = _make_handler(frozenset(["web-fetch"]))
    assert h.is_skill_trusted("web-fetch", "fetch")
    assert not h.is_skill_trusted("search", "query")


def test_is_trusted_wildcard():
    h = _make_handler(frozenset(["*"]))
    assert h.is_skill_trusted("web-fetch", "fetch")
    assert h.is_skill_trusted("any-skill", "run")


def test_is_trusted_dot_notation_specific_target():
    h = _make_handler(frozenset(["web.fetch"]))
    assert h.is_skill_trusted("web", "fetch")
    assert not h.is_skill_trusted("web", "search")
    assert not h.is_skill_trusted("other", "fetch")


def test_is_trusted_dot_notation_does_not_trust_whole_skill():
    h = _make_handler(frozenset(["web.fetch"]))
    assert not h.is_skill_trusted("web", "search")


def test_is_trusted_skill_level_trusts_all_targets():
    h = _make_handler(frozenset(["web"]))
    assert h.is_skill_trusted("web", "fetch")
    assert h.is_skill_trusted("web", "search")
    assert not h.is_skill_trusted("other", "fetch")


def test_is_trusted_backend_ast_trust():
    """Backend returning trusted=True from AST grants trust without CLI override."""
    backend = MagicMock()
    backend.schemas = []
    backend.executors = {"execute_skill": AsyncMock(return_value="ok")}
    backend.get_skill_trusted = MagicMock(return_value=True)
    memory = MagicMock()
    memory.store = MagicMock()
    h = ToolHandler(backend, memory, trusted_skills=frozenset())
    assert h.is_skill_trusted("web", "fetch")


def test_is_trusted_backend_ast_untrusted():
    """Backend returning trusted=False (AST untrusted) requires CLI override."""
    backend = MagicMock()
    backend.schemas = []
    backend.executors = {"execute_skill": AsyncMock(return_value="ok")}
    backend.get_skill_trusted = MagicMock(return_value=False)
    memory = MagicMock()
    memory.store = MagicMock()
    h = ToolHandler(backend, memory, trusted_skills=frozenset())
    assert not h.is_skill_trusted("web", "fetch")


def test_is_trusted_backend_ast_none():
    """Backend returning None (Makefile backend) requires CLI override."""
    backend = MagicMock()
    backend.schemas = []
    backend.executors = {"execute_skill": AsyncMock(return_value="ok")}
    backend.get_skill_trusted = MagicMock(return_value=None)
    memory = MagicMock()
    memory.store = MagicMock()
    h = ToolHandler(backend, memory, trusted_skills=frozenset())
    assert not h.is_skill_trusted("web", "fetch")


# ── AgentManager.astream_events — skill confirmation ──────────────────────────


def _make_tool_callback(
    tool_name: str, tool_args: dict, tool_call_id: str = "tc-1"
) -> ToolCallback:
    return ToolCallback(
        message="{}",
        tool_name=tool_name,
        tool_args=tool_args,
        tool_call_id=tool_call_id,
        description="",
    )


async def _drain_events(manager: AgentManager, session_id: str, message: str) -> list:
    """Collect all AgentEvents from one turn, auto-allowing any ConfirmEvent."""
    events = []
    async for event in manager.astream_events(session_id, message):
        events.append(event)
        if isinstance(event, ConfirmEvent):
            event.allow()
    return events


def _make_manager_with_loop(
    cbs: list[CallBack], trusted_skills: frozenset[str] = frozenset()
) -> tuple[AgentManager, str]:
    """Create an AgentManager whose AgenticLoop yields *cbs* in order."""
    tool_handler = MagicMock()
    tool_handler.is_skill_trusted = MagicMock(
        side_effect=lambda skill, target: (
            "*" in trusted_skills or skill in trusted_skills
        )
    )
    tool_handler.execute = AsyncMock(
        return_value=get_tool_result("result-output", "", 0)
    )

    manager = AgentManager(tool_handler)
    session_id = manager.get_session_id()

    async def _fake_astream(msg: str):
        for cb in cbs:
            if isinstance(cb, ToolCallback):
                yield cb
                await cb.wait()
            else:
                yield cb

    loop_mock = MagicMock(spec=AgenticLoop)
    loop_mock.astream = _fake_astream
    loop_mock._max_tool_output = 0
    loop_mock._messages = []
    loop_mock.compact_history = MagicMock(return_value=(0, 0))
    manager._sessions[session_id] = loop_mock
    manager._tool_handler = tool_handler
    return manager, session_id


async def test_astream_events_token_and_done():
    """TokenCallback → TokenEvent; MessageCallback → DoneEvent."""
    cbs = [
        TokenCallback("hello "),
        TokenCallback("world"),
        MessageCallback("hello world"),
    ]
    manager, sid = _make_manager_with_loop(cbs)
    events = await _drain_events(manager, sid, "hi")
    assert [type(e).__name__ for e in events] == [
        "TokenEvent",
        "TokenEvent",
        "DoneEvent",
    ]
    assert events[0].text == "hello "
    assert events[2].content == "hello world"


async def test_astream_events_trusted_skill_no_confirm():
    """Trusted skill executes without emitting ConfirmEvent."""
    cb = _make_tool_callback(
        "execute_skill", {"name": "web-fetch", "target": "fetch", "kwargs": {}}
    )
    cbs = [cb, MessageCallback("done")]
    manager, sid = _make_manager_with_loop(cbs, trusted_skills=frozenset(["web-fetch"]))
    events = await _drain_events(manager, sid, "go")
    assert not any(isinstance(e, ConfirmEvent) for e in events)
    assert any(isinstance(e, ToolStartEvent) for e in events)
    assert any(isinstance(e, ToolDoneEvent) for e in events)


async def test_astream_events_untrusted_skill_confirm_allowed():
    """Untrusted skill emits ConfirmEvent; if allowed, tool executes."""
    cb = _make_tool_callback(
        "execute_skill", {"name": "web-fetch", "target": "fetch", "kwargs": {}}
    )
    cbs = [cb, MessageCallback("done")]
    manager, sid = _make_manager_with_loop(cbs, trusted_skills=frozenset())

    events: list = []
    async for event in manager.astream_events(sid, "go"):
        events.append(event)
        if isinstance(event, ConfirmEvent):
            event.allow()

    assert any(isinstance(e, ConfirmEvent) for e in events)
    assert any(isinstance(e, ToolStartEvent) for e in events)
    assert any(isinstance(e, ToolDoneEvent) for e in events)
    manager._tool_handler.execute.assert_awaited_once()


async def test_astream_events_untrusted_skill_confirm_denied():
    """Untrusted skill denied: tool is skipped, no ToolStartEvent/ToolDoneEvent."""
    cb = _make_tool_callback(
        "execute_skill", {"name": "web-fetch", "target": "fetch", "kwargs": {}}
    )
    cbs = [cb, MessageCallback("done")]
    manager, sid = _make_manager_with_loop(cbs, trusted_skills=frozenset())

    events: list = []
    async for event in manager.astream_events(sid, "go"):
        events.append(event)
        if isinstance(event, ConfirmEvent):
            event.deny()

    assert any(isinstance(e, ConfirmEvent) for e in events)
    assert not any(isinstance(e, ToolStartEvent) for e in events)
    assert not any(isinstance(e, ToolDoneEvent) for e in events)
    manager._tool_handler.execute.assert_not_awaited()


async def test_astream_events_non_skill_tool_no_confirm():
    """Non-execute_skill tools are never gated by the trust mechanism."""
    cb = _make_tool_callback("list_skills", {"filter": ""})
    cbs = [cb, MessageCallback("done")]
    manager, sid = _make_manager_with_loop(cbs)
    events = await _drain_events(manager, sid, "go")
    assert not any(isinstance(e, ConfirmEvent) for e in events)
    assert any(isinstance(e, ToolStartEvent) for e in events)
    manager._tool_handler.is_skill_trusted.assert_not_called()
