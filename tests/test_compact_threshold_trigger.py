"""Tests for the auto-compact subsystem improvements (Issues 1–5)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from make_agent.agent_core.agent import (
    COMPACT_SUMMARY_MARKER,
    compact_with_summary,
    _split_into_turns,
)
from make_agent.agent_core.constants import KEEP_RECENT_TURNS
from make_agent.provider import estimate_tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_turn(user_text: str, assistant_text: str) -> list[dict]:
    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]


def _build_messages(n_turns: int, system: str = "Be helpful.") -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": system}]
    for i in range(n_turns):
        msgs.extend(_make_turn(f"User turn {i}", f"Assistant turn {i}"))
    return msgs


# ---------------------------------------------------------------------------
# Issue 1: proactive compact uses peak input_tokens, not cumulative sum
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proactive_uses_peak_not_cumulative_sum():
    """With threshold=100, two UsageEvents of 80 each (sum=160) must NOT trigger
    compaction because the peak per-turn value is 80 (below the threshold)."""
    from make_agent.agent_core.agent import AgentManager
    from make_agent.agent_core.events import DoneEvent, UsageEvent
    from make_agent.agent_core.loop import AgentConfig

    config = AgentConfig(system_prompt="", model="gpt-4o-mini")

    manager = AgentManager(
        tool_handler=MagicMock(),
        compact_threshold=100,
    )
    session_id = manager.create_session(config)

    # Build a fake event stream that emits two UsageEvents of 80 tokens each.
    async def _fake_stream(request):
        yield UsageEvent(model="gpt-4o-mini", input_tokens=80, output_tokens=10)
        yield UsageEvent(model="gpt-4o-mini", input_tokens=80, output_tokens=10)
        yield DoneEvent(content="ok")

    compact_called = False
    original_compact = manager._compact_session

    async def _spy_compact(sid):
        nonlocal compact_called
        compact_called = True
        return await original_compact(sid)

    manager._compact_session = _spy_compact

    with patch.object(manager, "_build_chain", return_value=lambda _req: _fake_stream(None)):
        events = [e async for e in manager.astream_events(session_id, "hi")]

    assert not compact_called, "Compact must NOT fire when peak (80) < threshold (100)"
    from make_agent.agent_core.events import CompactEvent
    assert not any(isinstance(e, CompactEvent) for e in events)


@pytest.mark.asyncio
async def test_proactive_triggers_when_peak_exceeds_threshold():
    """With threshold=100, a single UsageEvent of 120 tokens MUST trigger compaction."""
    from make_agent.agent_core.agent import AgentManager
    from make_agent.agent_core.events import CompactEvent, DoneEvent, UsageEvent
    from make_agent.agent_core.loop import AgentConfig

    config = AgentConfig(system_prompt="", model="gpt-4o-mini")

    manager = AgentManager(
        tool_handler=MagicMock(),
        compact_threshold=100,
    )
    session_id = manager.create_session(config)

    async def _fake_stream(request):
        yield UsageEvent(model="gpt-4o-mini", input_tokens=120, output_tokens=10)
        yield DoneEvent(content="ok")

    async def _spy_compact(_sid):
        return 3  # pretend 3 messages removed

    manager._compact_session = _spy_compact

    with patch.object(manager, "_build_chain", return_value=lambda _req: _fake_stream(None)):
        events = [e async for e in manager.astream_events(session_id, "hi")]

    compact_events = [e for e in events if isinstance(e, CompactEvent)]
    assert len(compact_events) == 1
    assert compact_events[0].messages_removed == 3


# ---------------------------------------------------------------------------
# Issue 2: compact_with_summary keeps the last KEEP_RECENT_TURNS turns intact
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recent_turns_survive_compaction():
    """The last KEEP_RECENT_TURNS turns must appear verbatim in the compacted output."""
    messages = _build_messages(n_turns=6)

    # Grab what the last KEEP_RECENT_TURNS turns look like.
    non_system = [m for m in messages if m.get("role") != "system"]
    turns = _split_into_turns(non_system)
    expected_recent = [msg for turn in turns[-KEEP_RECENT_TURNS:] for msg in turn]

    summary_text = "This is the compact summary."

    with patch(
        "make_agent.agent_core.agent._summarize_messages",
        new=AsyncMock(return_value=summary_text),
    ):
        result = await compact_with_summary(messages, model="gpt-4o-mini")

    non_system_result = [m for m in result if m.get("role") != "system"]
    assert non_system_result == expected_recent, (
        "Last KEEP_RECENT_TURNS turns must survive compaction byte-for-byte"
    )


@pytest.mark.asyncio
async def test_summary_message_inserted_before_recent_turns():
    """A summary system message must appear before the retained recent turns."""
    messages = _build_messages(n_turns=4)
    summary_text = "Summary here."

    with patch(
        "make_agent.agent_core.agent._summarize_messages",
        new=AsyncMock(return_value=summary_text),
    ):
        result = await compact_with_summary(messages, model="gpt-4o-mini")

    summary_msgs = [m for m in result if m.get("role") == "system" and COMPACT_SUMMARY_MARKER in (m.get("content") or "")]
    assert len(summary_msgs) == 1
    assert summary_msgs[0]["content"] == f"{COMPACT_SUMMARY_MARKER}{summary_text}"

    # Summary message must come before any user/assistant messages in the result.
    summary_pos = result.index(summary_msgs[0])
    user_positions = [i for i, m in enumerate(result) if m.get("role") == "user"]
    assert all(summary_pos < p for p in user_positions)


# ---------------------------------------------------------------------------
# Issue 3: no stacked summaries — exactly one summary message at all times
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_second_compaction_replaces_not_stacks_summary():
    """Running compact_with_summary twice must leave exactly one summary message."""
    messages = _build_messages(n_turns=6)

    call_count = 0

    async def _fake_summarize(msgs, model, max_tokens=1024):
        nonlocal call_count
        call_count += 1
        return f"Summary #{call_count}"

    with patch("make_agent.agent_core.agent._summarize_messages", side_effect=_fake_summarize):
        compacted_once = await compact_with_summary(messages, model="gpt-4o-mini")

    # Add a few more turns to make a second compaction worthwhile.
    for i in range(4):
        compacted_once.extend(_make_turn(f"Extra user {i}", f"Extra assistant {i}"))

    with patch("make_agent.agent_core.agent._summarize_messages", side_effect=_fake_summarize):
        compacted_twice = await compact_with_summary(compacted_once, model="gpt-4o-mini")

    summary_msgs = [
        m for m in compacted_twice
        if m.get("role") == "system" and (m.get("content") or "").startswith(COMPACT_SUMMARY_MARKER)
    ]
    assert len(summary_msgs) == 1, (
        f"Expected exactly 1 summary message, got {len(summary_msgs)}"
    )


@pytest.mark.asyncio
async def test_second_compaction_merges_prior_summary():
    """The summarizer must receive the prior summary when compacting a second time."""
    messages = _build_messages(n_turns=6)
    received_inputs: list[list[dict]] = []

    async def _capture_summarize(msgs, model, max_tokens=1024):
        received_inputs.append(list(msgs))
        return "New merged summary."

    with patch("make_agent.agent_core.agent._summarize_messages", side_effect=_capture_summarize):
        compacted_once = await compact_with_summary(messages, model="gpt-4o-mini")

    # Inject the compacted history with a known summary, then compact again.
    for i in range(4):
        compacted_once.extend(_make_turn(f"More user {i}", f"More assistant {i}"))

    received_inputs.clear()
    with patch("make_agent.agent_core.agent._summarize_messages", side_effect=_capture_summarize):
        await compact_with_summary(compacted_once, model="gpt-4o-mini")

    # The second summarization call must include a "Prior summary:" system message.
    assert received_inputs, "Summarizer was not called on the second compaction"
    second_call_msgs = received_inputs[0]
    prior_summary_msgs = [
        m for m in second_call_msgs
        if m.get("role") == "system" and "Prior summary:" in (m.get("content") or "")
    ]
    assert prior_summary_msgs, "Prior summary must be passed to the summarizer on second compaction"


# ---------------------------------------------------------------------------
# Issue 4: token-based gating — fall back when summary grows token count
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_falls_back_when_summary_grows_tokens():
    """When the summary would increase estimated token count, fall back to compact_messages."""
    # Need more than KEEP_COMPACT_TURNS=5 turns so compact_messages can actually prune.
    messages = _build_messages(n_turns=8)
    original_token_count = estimate_tokens(messages, "gpt-4o-mini")

    # Return a very long summary that will exceed the original token count.
    bloated_summary = "word " * (original_token_count * 5)

    with patch(
        "make_agent.agent_core.agent._summarize_messages",
        new=AsyncMock(return_value=bloated_summary),
    ):
        result = await compact_with_summary(messages, model="gpt-4o-mini")

    result_tokens = estimate_tokens(result, "gpt-4o-mini")
    assert result_tokens < original_token_count, (
        "Result must have fewer tokens than the original when summary bloats tokens"
    )

    # Should have fallen back to compact_messages (no COMPACT_SUMMARY_MARKER).
    summary_msgs = [
        m for m in result
        if m.get("role") == "system" and (m.get("content") or "").startswith(COMPACT_SUMMARY_MARKER)
    ]
    assert not summary_msgs, "Bloated summary should not appear in fallback result"


# ---------------------------------------------------------------------------
# Issue 5: hysteresis — no back-to-back proactive compaction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hysteresis_suppresses_proactive_compact_after_first():
    """After a successful proactive compact, a second high-token turn must NOT
    trigger another compact until tokens drop below compact_target."""
    from make_agent.agent_core.agent import AgentManager
    from make_agent.agent_core.events import DoneEvent, UsageEvent
    from make_agent.agent_core.loop import AgentConfig

    config = AgentConfig(system_prompt="", model="gpt-4o-mini")

    # threshold=100, target=50 — after compact, 80 tokens keeps the skip flag set.
    manager = AgentManager(
        tool_handler=MagicMock(),
        compact_threshold=100,
        compact_target=50,
    )
    session_id = manager.create_session(config)

    compact_count = 0

    async def _spy_compact(_sid):
        nonlocal compact_count
        compact_count += 1
        return 3

    manager._compact_session = _spy_compact

    # Turn 1: peak=120 → triggers compact, sets skip flag.
    async def _stream_turn1(request):
        yield UsageEvent(model="gpt-4o-mini", input_tokens=120, output_tokens=10)
        yield DoneEvent(content="turn1")

    with patch.object(manager, "_build_chain", return_value=lambda _req: _stream_turn1(None)):
        await _drain(manager.astream_events(session_id, "turn1"))

    assert compact_count == 1

    # Turn 2: peak=110 (still above threshold, but skip flag is set) → no compact.
    async def _stream_turn2(request):
        yield UsageEvent(model="gpt-4o-mini", input_tokens=110, output_tokens=10)
        yield DoneEvent(content="turn2")

    with patch.object(manager, "_build_chain", return_value=lambda _req: _stream_turn2(None)):
        await _drain(manager.astream_events(session_id, "turn2"))

    assert compact_count == 1, "Compact must not fire again while skip flag is set"


@pytest.mark.asyncio
async def test_hysteresis_rearms_after_tokens_drop_below_target():
    """Once tokens fall below compact_target the skip flag is cleared and
    compaction fires again when tokens cross the threshold."""
    from make_agent.agent_core.agent import AgentManager
    from make_agent.agent_core.events import DoneEvent, UsageEvent
    from make_agent.agent_core.loop import AgentConfig
    config = AgentConfig(system_prompt="", model="gpt-4o-mini")
    manager = AgentManager(
        tool_handler=MagicMock(),
        compact_threshold=100,
        compact_target=50,
    )
    session_id = manager.create_session(config)

    compact_count = 0

    async def _spy_compact(_sid):
        nonlocal compact_count
        compact_count += 1
        return 3

    manager._compact_session = _spy_compact

    # Turn 1: triggers compact → skip flag set.
    async def _stream_high(request):
        yield UsageEvent(model="gpt-4o-mini", input_tokens=120, output_tokens=10)
        yield DoneEvent(content="high")

    with patch.object(manager, "_build_chain", return_value=lambda _req: _stream_high(None)):
        await _drain(manager.astream_events(session_id, "high"))

    assert compact_count == 1

    # Turn 2: peak=30 (below target=50) → clears skip flag.
    async def _stream_low(request):
        yield UsageEvent(model="gpt-4o-mini", input_tokens=30, output_tokens=5)
        yield DoneEvent(content="low")

    with patch.object(manager, "_build_chain", return_value=lambda _req: _stream_low(None)):
        await _drain(manager.astream_events(session_id, "low"))

    assert compact_count == 1, "No compact on low-token turn"
    assert not manager._skip_proactive.get(session_id, False), "Skip flag should be cleared"

    # Turn 3: peak=120 again → compact fires again.
    with patch.object(manager, "_build_chain", return_value=lambda _req: _stream_high(None)):
        await _drain(manager.astream_events(session_id, "high2"))

    assert compact_count == 2, "Compact must re-arm after tokens drop below target"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


async def _drain(ait):
    async for _ in ait:
        pass
