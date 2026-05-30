"""Shell event-handling regressions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from make_agent.agent_core import HistoryCompacted, StatusChanged, TurnCancelled, TurnStarted
from make_agent.agent_shell.shell import AgentStatus, MakeAgentShell


def _make_shell() -> MakeAgentShell:
    return MakeAgentShell(
        agent_manager=MagicMock(),
        session_id="sid",
        model="claude-sonnet-4-5",
        history_path=Path("/tmp/make-agent-history"),
    )


@pytest.mark.asyncio
async def test_compact_event_keeps_shell_streaming_when_turn_active():
    shell = _make_shell()
    turn = shell._state.new_turn("hello")
    shell._state.status = AgentStatus.IDLE
    seen_statuses: list[AgentStatus] = []
    shell._refresh = lambda: seen_statuses.append(shell._state.status)

    await shell._event_queue.put(HistoryCompacted(turn_id="t1", attempt=1, messages_dropped=2))
    await shell._event_queue.put(TurnCancelled(turn_id="t1"))
    await shell._consume_turn_events(turn, "t1")

    assert AgentStatus.STREAMING in seen_statuses


@pytest.mark.asyncio
async def test_turn_started_restores_streaming_if_status_went_idle():
    shell = _make_shell()
    turn = shell._state.new_turn("hello")
    shell._state.status = AgentStatus.IDLE
    seen_statuses: list[AgentStatus] = []
    shell._refresh = lambda: seen_statuses.append(shell._state.status)

    await shell._event_queue.put(TurnStarted(turn_id="t1"))
    await shell._event_queue.put(TurnCancelled(turn_id="t1"))
    await shell._consume_turn_events(turn, "t1")

    assert AgentStatus.STREAMING in seen_statuses


@pytest.mark.asyncio
async def test_busy_status_event_restores_streaming_if_status_went_idle():
    shell = _make_shell()
    turn = shell._state.new_turn("hello")
    shell._state.status = AgentStatus.IDLE
    seen_statuses: list[AgentStatus] = []
    shell._refresh = lambda: seen_statuses.append(shell._state.status)

    await shell._event_queue.put(StatusChanged(is_busy=True))
    await shell._event_queue.put(TurnCancelled(turn_id="t1"))
    await shell._consume_turn_events(turn, "t1")

    assert AgentStatus.STREAMING in seen_statuses
