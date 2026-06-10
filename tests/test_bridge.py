"""Tests for the AgentManager queue bridge (run_shell_bridge / _execute_bridge_turn)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock


from make_agent.agent_core import (
    AgentManager,
    AgenticLoop,
    ApprovalRequested,
    ApproveSkill,
    CancelTurn,
    DenySkill,
    ManagerError,
    MessageCallback,
    ShellCommand,
    ShellEvent,
    Shutdown,
    StartTurn,
    StatusChanged,
    ToolFinished,
    ToolStarted,
    TokenEmitted,
    TurnCancelled,
    TurnFinished,
    ToolCallback,
    TokenCallback,
)
from make_agent.tool_handler.runner import get_tool_result


# ── helpers ────────────────────────────────────────────────────────────────────


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


def _make_manager(
    cbs: list, trusted_skills: frozenset[str] = frozenset()
) -> tuple[AgentManager, str]:
    """Return an AgentManager backed by a fake AgenticLoop that yields *cbs*."""
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


async def _run_bridge_turn(
    cbs: list,
    trusted_skills: frozenset[str] = frozenset(),
    extra_commands: list[ShellCommand] | None = None,
) -> list[ShellEvent]:
    """
    Start the bridge, send one StartTurn, collect all events until
    TurnFinished / TurnCancelled / ManagerError, then shut down.
    Returns the collected ShellEvent list.
    """
    manager, session_id = _make_manager(cbs, trusted_skills)
    cmd_q: asyncio.Queue[ShellCommand] = asyncio.Queue()
    evt_q: asyncio.Queue[ShellEvent] = asyncio.Queue()

    bridge_task = asyncio.create_task(
        manager.run_shell_bridge(session_id, cmd_q, evt_q)
    )

    await cmd_q.put(StartTurn(message="hello", turn_id="t1"))
    if extra_commands:
        for cmd in extra_commands:
            await cmd_q.put(cmd)

    events: list[ShellEvent] = []
    async for event in _drain(evt_q, bridge_task):
        events.append(event)

    return events


async def _drain(
    evt_q: asyncio.Queue[ShellEvent],
    bridge_task: asyncio.Task,
) -> list[ShellEvent]:
    """Yield events until a terminal event arrives, then send Shutdown."""
    events: list[ShellEvent] = []
    while True:
        event = await asyncio.wait_for(evt_q.get(), timeout=5)
        events.append(event)
        yield event
        if isinstance(event, (TurnFinished, TurnCancelled, ManagerError)):
            await evt_q.put(Shutdown())  # will be consumed next loop iteration
            # Actually we need to send Shutdown to bridge
            break
    # drain any StatusChanged that arrives after the terminal event
    try:
        while True:
            event = await asyncio.wait_for(evt_q.get(), timeout=0.1)
            events.append(event)
            yield event
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass
    # tell the bridge to stop and wait for it
    # (Shutdown might already be queued)
    return


async def _collect_events(
    manager: AgentManager,
    session_id: str,
    commands: list[ShellCommand],
    *,
    stop_after: int = 1,
    timeout: float = 5.0,
) -> list[ShellEvent]:
    """Drive the bridge with *commands*, collect events until *stop_after* terminal events."""
    cmd_q: asyncio.Queue[ShellCommand] = asyncio.Queue()
    evt_q: asyncio.Queue[ShellEvent] = asyncio.Queue()

    bridge_task = asyncio.create_task(
        manager.run_shell_bridge(session_id, cmd_q, evt_q)
    )

    for cmd in commands:
        await cmd_q.put(cmd)

    collected: list[ShellEvent] = []
    terminal_count = 0
    while terminal_count < stop_after:
        event = await asyncio.wait_for(evt_q.get(), timeout=timeout)
        collected.append(event)
        if isinstance(event, (TurnFinished, TurnCancelled, ManagerError)):
            terminal_count += 1

    # drain remaining queued events (e.g. StatusChanged after terminal)
    try:
        while True:
            event = await asyncio.wait_for(evt_q.get(), timeout=0.05)
            collected.append(event)
    except asyncio.TimeoutError:
        pass

    await cmd_q.put(Shutdown())
    await asyncio.wait_for(bridge_task, timeout=timeout)
    return collected


# ── normal turn flow ───────────────────────────────────────────────────────────


async def test_bridge_simple_turn_event_sequence():
    """StartTurn -> TurnStarted, tokens, TurnFinished, StatusChanged."""
    cbs = [
        TokenCallback("hello "),
        TokenCallback("world"),
        MessageCallback("hello world"),
    ]
    manager, sid = _make_manager(cbs)
    events = await _collect_events(
        manager, sid, [StartTurn(message="hi", turn_id="t1")]
    )

    types = [type(e).__name__ for e in events]
    assert "TurnStarted" in types
    assert "TokenEmitted" in types
    assert "TurnFinished" in types
    assert "StatusChanged" in types

    token_texts = [e.text for e in events if isinstance(e, TokenEmitted)]
    assert token_texts == ["hello ", "world"]

    done = next(e for e in events if isinstance(e, TurnFinished))
    assert done.content == "hello world"


async def test_bridge_status_busy_idle_sequence():
    """StatusChanged(is_busy=True) comes before TurnStarted; is_busy=False comes last."""
    cbs = [MessageCallback("ok")]
    manager, sid = _make_manager(cbs)
    events = await _collect_events(
        manager, sid, [StartTurn(message="hi", turn_id="t1")]
    )

    status_events = [e for e in events if isinstance(e, StatusChanged)]
    assert status_events[0].is_busy is True
    assert status_events[-1].is_busy is False


async def test_bridge_turn_ids_are_propagated():
    """All turn-scoped events carry the turn_id supplied in StartTurn."""
    cbs = [TokenCallback("x"), MessageCallback("x")]
    manager, sid = _make_manager(cbs)
    events = await _collect_events(
        manager, sid, [StartTurn(message="hi", turn_id="my-turn")]
    )

    for event in events:
        if hasattr(event, "turn_id"):
            assert event.turn_id == "my-turn"


# ── tool events ────────────────────────────────────────────────────────────────


async def test_bridge_tool_events():
    """Tool calls produce ToolStarted then ToolFinished with matching tool_id."""
    cb = _make_tool_callback("list_files", {"path": "."})
    cbs = [cb, MessageCallback("done")]
    manager, sid = _make_manager(cbs)
    events = await _collect_events(
        manager, sid, [StartTurn(message="go", turn_id="t1")]
    )

    started = next(e for e in events if isinstance(e, ToolStarted))
    finished = next(e for e in events if isinstance(e, ToolFinished))
    assert started.name == "list_files"
    assert finished.name == "list_files"
    assert started.tool_id == finished.tool_id
    assert finished.is_error is False
    assert finished.output == "result-output"


# ── single-turn policy ─────────────────────────────────────────────────────────


async def test_bridge_rejects_second_start_turn_while_busy():
    """A second StartTurn while one is active yields ManagerError."""
    # Use a slow-finishing fake: block until we send the second StartTurn
    gate = asyncio.Event()

    async def _slow_astream(msg: str):
        await gate.wait()
        yield TokenCallback("hi")
        yield MessageCallback("hi")

    tool_handler = MagicMock()
    tool_handler.is_skill_trusted = MagicMock(return_value=False)
    tool_handler.execute = AsyncMock(return_value=get_tool_result("ok", "", 0))
    manager = AgentManager(tool_handler)
    session_id = manager.get_session_id()

    loop_mock = MagicMock(spec=AgenticLoop)
    loop_mock.astream = _slow_astream
    loop_mock._max_tool_output = 0
    loop_mock._messages = []
    loop_mock.compact_history = MagicMock(return_value=(0, 0))
    manager._sessions[session_id] = loop_mock

    cmd_q: asyncio.Queue[ShellCommand] = asyncio.Queue()
    evt_q: asyncio.Queue[ShellEvent] = asyncio.Queue()

    bridge_task = asyncio.create_task(
        manager.run_shell_bridge(session_id, cmd_q, evt_q)
    )

    await cmd_q.put(StartTurn(message="first", turn_id="t1"))
    # wait for TurnStarted so the first turn is definitely active
    first_event = await asyncio.wait_for(evt_q.get(), timeout=5)
    assert isinstance(first_event, StatusChanged)

    await cmd_q.put(StartTurn(message="second", turn_id="t2"))

    # Drain until we see the ManagerError (TurnStarted for t1 may arrive first)
    error_event = None
    for _ in range(5):
        ev = await asyncio.wait_for(evt_q.get(), timeout=5)
        if isinstance(ev, ManagerError):
            error_event = ev
            break
    assert error_event is not None, "Expected ManagerError but did not receive one"
    assert "already active" in error_event.message.lower()

    # unblock and clean up
    gate.set()
    await cmd_q.put(CancelTurn())
    await cmd_q.put(Shutdown())
    await asyncio.wait_for(bridge_task, timeout=5)


# ── cancellation ──────────────────────────────────────────────────────────────


async def test_bridge_cancel_turn_emits_turn_cancelled():
    """CancelTurn stops the active turn and emits TurnCancelled + StatusChanged(False)."""
    gate = asyncio.Event()

    async def _blocking_astream(msg: str):
        await gate.wait()
        yield MessageCallback("never")

    tool_handler = MagicMock()
    tool_handler.is_skill_trusted = MagicMock(return_value=False)
    manager = AgentManager(tool_handler)
    session_id = manager.get_session_id()

    loop_mock = MagicMock(spec=AgenticLoop)
    loop_mock.astream = _blocking_astream
    loop_mock._max_tool_output = 0
    loop_mock._messages = []
    loop_mock.compact_history = MagicMock(return_value=(0, 0))
    manager._sessions[session_id] = loop_mock

    cmd_q: asyncio.Queue[ShellCommand] = asyncio.Queue()
    evt_q: asyncio.Queue[ShellEvent] = asyncio.Queue()

    bridge_task = asyncio.create_task(
        manager.run_shell_bridge(session_id, cmd_q, evt_q)
    )

    await cmd_q.put(StartTurn(message="hi", turn_id="t1"))
    # wait for StatusChanged(is_busy=True)
    evt = await asyncio.wait_for(evt_q.get(), timeout=5)
    assert isinstance(evt, StatusChanged) and evt.is_busy is True

    await cmd_q.put(CancelTurn())

    collected: list[ShellEvent] = []
    while True:
        ev = await asyncio.wait_for(evt_q.get(), timeout=5)
        collected.append(ev)
        if isinstance(ev, StatusChanged) and not ev.is_busy:
            break

    assert any(isinstance(e, TurnCancelled) for e in collected)
    idle = [e for e in collected if isinstance(e, StatusChanged)]
    assert idle[-1].is_busy is False

    await cmd_q.put(Shutdown())
    await asyncio.wait_for(bridge_task, timeout=5)


# ── approval flow ──────────────────────────────────────────────────────────────


async def test_bridge_approval_allowed_resumes_turn():
    """ApprovalRequested -> ApproveSkill -> tool executes -> TurnFinished."""
    cb = _make_tool_callback(
        "execute_skill", {"name": "web-fetch", "target": "fetch", "kwargs": {}}
    )
    cbs = [cb, MessageCallback("done")]
    manager, sid = _make_manager(cbs, trusted_skills=frozenset())

    cmd_q: asyncio.Queue[ShellCommand] = asyncio.Queue()
    evt_q: asyncio.Queue[ShellEvent] = asyncio.Queue()

    bridge_task = asyncio.create_task(manager.run_shell_bridge(sid, cmd_q, evt_q))
    await cmd_q.put(StartTurn(message="go", turn_id="t1"))

    collected: list[ShellEvent] = []
    while True:
        ev = await asyncio.wait_for(evt_q.get(), timeout=5)
        collected.append(ev)
        if isinstance(ev, ApprovalRequested):
            await cmd_q.put(ApproveSkill(request_id=ev.request_id))
        if isinstance(ev, TurnFinished):
            break

    assert any(isinstance(e, ApprovalRequested) for e in collected)
    assert any(isinstance(e, ToolStarted) for e in collected)
    assert any(isinstance(e, ToolFinished) for e in collected)

    done = next(e for e in collected if isinstance(e, TurnFinished))
    assert done.content == "done"

    await cmd_q.put(Shutdown())
    await asyncio.wait_for(bridge_task, timeout=5)


async def test_bridge_approval_denied_skips_tool():
    """ApprovalRequested -> DenySkill -> no ToolStarted/ToolFinished -> TurnFinished."""
    cb = _make_tool_callback(
        "execute_skill", {"name": "web-fetch", "target": "fetch", "kwargs": {}}
    )
    cbs = [cb, MessageCallback("done")]
    manager, sid = _make_manager(cbs, trusted_skills=frozenset())

    cmd_q: asyncio.Queue[ShellCommand] = asyncio.Queue()
    evt_q: asyncio.Queue[ShellEvent] = asyncio.Queue()

    bridge_task = asyncio.create_task(manager.run_shell_bridge(sid, cmd_q, evt_q))
    await cmd_q.put(StartTurn(message="go", turn_id="t1"))

    collected: list[ShellEvent] = []
    while True:
        ev = await asyncio.wait_for(evt_q.get(), timeout=5)
        collected.append(ev)
        if isinstance(ev, ApprovalRequested):
            await cmd_q.put(DenySkill(request_id=ev.request_id))
        if isinstance(ev, TurnFinished):
            break

    assert any(isinstance(e, ApprovalRequested) for e in collected)
    assert not any(isinstance(e, ToolStarted) for e in collected)
    assert not any(isinstance(e, ToolFinished) for e in collected)
    manager._tool_handler.execute.assert_not_awaited()

    await cmd_q.put(Shutdown())
    await asyncio.wait_for(bridge_task, timeout=5)


async def test_bridge_unknown_approval_request_id_emits_manager_error():
    """Sending ApproveSkill with an unknown request_id yields a ManagerError."""
    cbs = [MessageCallback("hi")]
    manager, sid = _make_manager(cbs)
    events = await _collect_events(
        manager,
        sid,
        [ApproveSkill(request_id="nonexistent"), StartTurn(message="hi", turn_id="t1")],
    )
    assert any(isinstance(e, ManagerError) for e in events)


# ── shutdown ───────────────────────────────────────────────────────────────────


async def test_bridge_shutdown_exits_cleanly():
    """Shutdown with no active turn causes run_shell_bridge to return."""
    manager, sid = _make_manager([])
    cmd_q: asyncio.Queue[ShellCommand] = asyncio.Queue()
    evt_q: asyncio.Queue[ShellEvent] = asyncio.Queue()

    bridge_task = asyncio.create_task(manager.run_shell_bridge(sid, cmd_q, evt_q))
    await cmd_q.put(Shutdown())
    await asyncio.wait_for(bridge_task, timeout=5)
    assert bridge_task.done()
    assert not bridge_task.cancelled()
