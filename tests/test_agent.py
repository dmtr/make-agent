"""Tests for rate limit retry logic — _parse_retry_after and _acompletion_with_retry."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call, patch

import any_llm
import pytest
from make_agent.agent_core import _acompletion_with_retry, _flatten_messages_for_summary, _parse_retry_after, _prune_skill_messages, CompactEvent


def _make_rate_limit_error(
    retry_after: float | None = None,
    retry_after_ms: float | None = None,
) -> any_llm.RateLimitError:
    headers: dict[str, str] = {}
    if retry_after is not None:
        headers["retry-after"] = str(retry_after)
    if retry_after_ms is not None:
        headers["retry-after-ms"] = str(retry_after_ms)
    fake_response = MagicMock()
    fake_response.headers = headers
    fake_orig = MagicMock()
    fake_orig.response = fake_response
    return any_llm.RateLimitError(
        message="rate limit exceeded",
        original_exception=fake_orig,
        provider_name="anthropic",
    )


def _make_empty_stream():
    """Return an async iterator that yields no chunks (empty stream)."""

    async def _stream():
        return
        yield  # make it an async generator

    return _stream()


def _make_text_stream(content: str):
    """Return an async iterator that yields a single text chunk."""

    async def _stream():
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = content
        chunk.choices[0].delta.tool_calls = None
        chunk.usage = None
        yield chunk

    return _stream()


def _make_tool_call_stream(tool_id: str, tool_name: str, arguments: str):
    """Return an async iterator that yields a single tool-call chunk."""

    async def _stream():
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = None
        tc_delta = MagicMock()
        tc_delta.index = 0
        tc_delta.id = tool_id
        tc_delta.function = MagicMock()
        tc_delta.function.name = tool_name
        tc_delta.function.arguments = arguments
        chunk.choices[0].delta.tool_calls = [tc_delta]
        chunk.usage = None
        yield chunk

    return _stream()


def _make_parallel_tool_calls_stream_anthropic_style(calls: list[tuple[str, str, str]]):
    """Simulate Anthropic-style parallel tool calls where any_llm hardcodes index=0.

    Each call is (tool_id, tool_name, arguments_json).  All start-events and all
    delta-events carry index=0 — exactly the broken output produced by any_llm's
    Anthropic provider — so the fix must be id-based, not index-based.
    """

    async def _stream():
        for tool_id, tool_name, arguments in calls:
            # Start event: has id + name, hardcoded index=0
            start_chunk = MagicMock()
            start_chunk.choices = [MagicMock()]
            start_chunk.choices[0].delta.content = None
            tc_start = MagicMock()
            tc_start.index = 0  # hardcoded by any_llm bug
            tc_start.id = tool_id
            tc_start.function = MagicMock()
            tc_start.function.name = tool_name
            tc_start.function.arguments = ""
            start_chunk.choices[0].delta.tool_calls = [tc_start]
            start_chunk.usage = None
            yield start_chunk

            # Argument delta: no id, hardcoded index=0
            delta_chunk = MagicMock()
            delta_chunk.choices = [MagicMock()]
            delta_chunk.choices[0].delta.content = None
            tc_delta = MagicMock()
            tc_delta.index = 0  # hardcoded by any_llm bug
            tc_delta.id = None
            tc_delta.function = MagicMock()
            tc_delta.function.name = None
            tc_delta.function.arguments = arguments
            delta_chunk.choices[0].delta.tool_calls = [tc_delta]
            delta_chunk.usage = None
            yield delta_chunk

    return _stream()


def _make_tool_call_stream_empty_args(tool_id: str, tool_name: str):
    """Anthropic-style stream where a no-argument tool sends only a start event (no delta).

    This replicates what any_llm emits for tools with empty input: the start
    chunk sets arguments="" and no input_json_delta follows, leaving arguments as "".
    json.loads("") raises JSONDecodeError — the fix should treat "" as "{}".
    """

    async def _stream():
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = None
        tc_delta = MagicMock()
        tc_delta.index = 0
        tc_delta.id = tool_id
        tc_delta.function = MagicMock()
        tc_delta.function.name = tool_name
        tc_delta.function.arguments = ""  # no delta follows for empty-input tools
        chunk.choices[0].delta.tool_calls = [tc_delta]
        chunk.usage = None
        yield chunk

    return _stream()



def _mock_acompletion_with_retry(*streams):
    """Return an async callable that yields successive streams on each call."""
    streams_list = list(streams)
    call_count = 0

    async def _mock(*args, **kwargs):
        nonlocal call_count
        stream = streams_list[call_count % len(streams_list)]
        call_count += 1
        return stream

    return _mock


class TestParseRetryAfter:
    def test_retry_after_seconds(self):
        err = _make_rate_limit_error(retry_after=30)
        assert _parse_retry_after(err) == 30.0

    def test_retry_after_ms(self):
        err = _make_rate_limit_error(retry_after_ms=5000)
        assert _parse_retry_after(err) == 5.0

    def test_retry_after_ms_takes_priority(self):
        err = _make_rate_limit_error(retry_after=60, retry_after_ms=2000)
        assert _parse_retry_after(err) == 2.0

    def test_no_header_returns_none(self):
        err = _make_rate_limit_error()
        assert _parse_retry_after(err) is None

    def test_none_response(self):
        err = any_llm.RateLimitError(
            message="rate limit exceeded",
            original_exception=None,
            provider_name="anthropic",
        )
        assert _parse_retry_after(err) is None


class TestACompletionWithRetry:
    async def test_succeeds_on_first_attempt(self):
        stream = _make_empty_stream()
        with patch("make_agent.agent_core.agent.any_llm.acompletion", AsyncMock(return_value=stream)) as mock_c:
            result = await _acompletion_with_retry("model", [], {}, max_retries=3)
        assert result is stream
        mock_c.assert_called_once()

    async def test_retries_on_rate_limit_then_succeeds(self):
        err = _make_rate_limit_error(retry_after=10)
        stream = _make_empty_stream()
        with patch("make_agent.agent_core.agent.any_llm.acompletion", AsyncMock(side_effect=[err, err, stream])):
            with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
                result = await _acompletion_with_retry("model", [], {}, max_retries=3)
        assert result is stream
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(10.0)

    async def test_exponential_backoff_without_header(self):
        err = _make_rate_limit_error()
        stream = _make_empty_stream()
        with patch("make_agent.agent_core.agent.any_llm.acompletion", AsyncMock(side_effect=[err, err, stream])):
            with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
                await _acompletion_with_retry("model", [], {}, max_retries=3)
        assert mock_sleep.call_args_list == [call(1), call(2)]

    async def test_exponential_backoff_capped_at_60s(self):
        err = _make_rate_limit_error()
        stream = _make_empty_stream()
        side_effects = [err] * 7 + [stream]
        with patch("make_agent.agent_core.agent.any_llm.acompletion", AsyncMock(side_effect=side_effects)):
            with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
                await _acompletion_with_retry("model", [], {}, max_retries=10)
        waits = [c.args[0] for c in mock_sleep.call_args_list]
        assert all(w <= 60 for w in waits)
        assert waits[6] == 60  # 2^6=64 capped to 60

    async def test_raises_after_max_retries_exhausted(self):
        err = _make_rate_limit_error(retry_after=1)
        with patch("make_agent.agent_core.agent.any_llm.acompletion", AsyncMock(side_effect=err)):
            with patch("asyncio.sleep", AsyncMock()):
                with pytest.raises(any_llm.RateLimitError):
                    await _acompletion_with_retry("model", [], {}, max_retries=2)

    async def test_total_calls_equals_max_retries_plus_one(self):
        err = _make_rate_limit_error(retry_after=1)
        with patch("make_agent.agent_core.agent.any_llm.acompletion", AsyncMock(side_effect=err)) as mock_c:
            with patch("asyncio.sleep", AsyncMock()):
                with pytest.raises(any_llm.RateLimitError):
                    await _acompletion_with_retry("model", [], {}, max_retries=3)
        assert mock_c.call_count == 4  # 1 initial + 3 retries

    async def test_zero_max_retries_raises_immediately(self):
        err = _make_rate_limit_error(retry_after=1)
        with patch("make_agent.agent_core.agent.any_llm.acompletion", AsyncMock(side_effect=err)):
            with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
                with pytest.raises(any_llm.RateLimitError):
                    await _acompletion_with_retry("model", [], {}, max_retries=0)
        mock_sleep.assert_not_called()


# ── Agent safety guards ───────────────────────────────────────────────────────


class TestAgentSafetyGuards:
    def _make_agent(self, tmp_path):
        from make_agent.agent_core import Agent, AgentConfig
        from make_agent.memory import Memory
        from make_agent.skill_backend import MakefileSkillBackend
        from make_agent.tool_handler import ToolHandler

        memory = Memory(tmp_path / "memory.db")
        tool_handler = ToolHandler(MakefileSkillBackend(str(tmp_path), base_dir=tmp_path), memory)
        agent = Agent(AgentConfig(system_prompt="You are a helper.", model="openai/gpt-4o-mini", skills_dir=str(tmp_path)), memory, tool_handler)
        # Inject a custom tool to give the agent a known tool set
        agent._tool_handler._schemas.append(  # noqa: SLF001
            {
                "type": "function",
                "function": {"name": "safe", "description": "A safe tool.", "parameters": {"type": "object", "properties": {}, "required": []}},
            }
        )
        agent._tool_handler._executors["safe"] = lambda **_: "ok"  # noqa: SLF001
        return agent

    async def test_unknown_tool_is_rejected_without_running_make(self, tmp_path):
        agent = self._make_agent(tmp_path)

        with patch(
            "make_agent.agent_core.agent._acompletion_with_retry",
            _mock_acompletion_with_retry(
                _make_tool_call_stream("tc1", "hidden", "{}"),
                _make_text_stream("done"),
            ),
        ):
            result = await agent.arun("use hidden target")

        assert result == "done"
        tool_outputs = [m["content"] for m in agent.messages if m.get("role") == "tool"]
        assert any("unknown tool: hidden" in output for output in tool_outputs)

    async def test_model_turn_limit_stops_runaway_tool_loop(self, tmp_path):
        agent = self._make_agent(tmp_path)

        async def _always_tool_call(*args, **kwargs):
            return _make_tool_call_stream("tc1", "hidden", "{}")

        with (
            patch("make_agent.agent_core.agent._MAX_MODEL_TURNS_PER_REQUEST", 2),
            patch("make_agent.agent_core.agent._acompletion_with_retry", _always_tool_call),
        ):
            with pytest.raises(RuntimeError, match="model turns"):
                await agent.arun("loop forever")


class TestAssistantMessageContent:
    """Assistant messages with tool calls must never have content=None (breaks Ollama provider)."""

    async def test_tool_call_without_text_has_empty_string_content(self, tmp_path):
        """When the LLM streams a tool call with no text, the assistant message content must be ''."""
        from make_agent.agent_core import Agent, AgentConfig
        from make_agent.memory import Memory
        from make_agent.skill_backend import MakefileSkillBackend
        from make_agent.tool_handler import ToolHandler

        memory = Memory(tmp_path / "memory.db")
        tool_handler = ToolHandler(MakefileSkillBackend(str(tmp_path), base_dir=tmp_path), memory)
        agent = Agent(AgentConfig(system_prompt="You are a helper.", model="openai/gpt-4o-mini", skills_dir=str(tmp_path)), memory, tool_handler)
        # Inject say_hi as a known builtin tool
        agent._tool_handler._schemas.append(  # noqa: SLF001
            {
                "type": "function",
                "function": {"name": "say_hi", "description": "Say hi.", "parameters": {"type": "object", "properties": {}, "required": []}},
            }
        )
        agent._tool_handler._executors["say_hi"] = lambda **_: "hi"  # noqa: SLF001

        with patch(
            "make_agent.agent_core.agent._acompletion_with_retry",
            _mock_acompletion_with_retry(
                _make_tool_call_stream("tc1", "say_hi", "{}"),
                _make_text_stream("all done"),
            ),
        ):
            result = await agent.arun("call say_hi")

        assert result == "all done"
        assistant_msgs = [m for m in agent.messages if m.get("role") == "assistant" and "tool_calls" in m]
        assert assistant_msgs, "expected at least one assistant message with tool_calls"
        for msg in assistant_msgs:
            assert msg["content"] is not None, "assistant message content must not be None (breaks Ollama)"
            assert isinstance(msg["content"], str)


class TestAnthropicParallelToolCalls:
    """Anthropic (via any_llm) hardcodes index=0 for every tool call in a response.

    When the model returns two parallel tool calls, both start-events and both
    delta-events carry index=0.  The agent loop must use id-based detection to
    keep them separate, otherwise names and arguments get concatenated and the
    next LLM request fails with a JSONDecodeError, breaking the loop.
    """

    def _make_agent(self, tmp_path):
        from make_agent.agent_core import Agent, AgentConfig
        from make_agent.memory import Memory
        from make_agent.skill_backend import MakefileSkillBackend
        from make_agent.tool_handler import ToolHandler

        memory = Memory(tmp_path / "memory.db")
        tool_handler = ToolHandler(MakefileSkillBackend(str(tmp_path), base_dir=tmp_path), memory)
        agent = Agent(
            AgentConfig(system_prompt="You are a helper.", model="anthropic/claude-3-5-sonnet-20241022", skills_dir=str(tmp_path)),
            memory,
            tool_handler,
        )
        for name in ("tool_a", "tool_b"):
            agent._tool_handler._schemas.append(  # noqa: SLF001
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": f"Tool {name}.",
                        "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
                    },
                }
            )
            agent._tool_handler._executors[name] = lambda x, _n=name: f"{_n}_result"  # noqa: SLF001
        return agent

    async def test_parallel_tool_calls_are_kept_separate(self, tmp_path):
        """Two parallel Anthropic tool calls (both index=0) must each be executed."""
        agent = self._make_agent(tmp_path)

        parallel_stream = _make_parallel_tool_calls_stream_anthropic_style(
            [
                ("toolu_A", "tool_a", '{"x": 1}'),
                ("toolu_B", "tool_b", '{"x": 2}'),
            ]
        )

        with patch(
            "make_agent.agent_core.agent._acompletion_with_retry",
            _mock_acompletion_with_retry(parallel_stream, _make_text_stream("done")),
        ):
            result = await agent.arun("run both tools")

        assert result == "done"
        tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 2, f"expected 2 tool results, got {len(tool_msgs)}: {tool_msgs}"

        tool_call_ids = {m["tool_call_id"] for m in tool_msgs}
        assert "toolu_A" in tool_call_ids
        assert "toolu_B" in tool_call_ids

    async def test_parallel_tool_calls_have_correct_arguments(self, tmp_path):
        """Arguments must not be concatenated across parallel tool calls."""
        agent = self._make_agent(tmp_path)

        received: dict[str, dict] = {}

        def capture_a(**kwargs):
            received["tool_a"] = kwargs
            return "result_a"

        def capture_b(**kwargs):
            received["tool_b"] = kwargs
            return "result_b"

        agent._tool_handler._executors["tool_a"] = capture_a  # noqa: SLF001
        agent._tool_handler._executors["tool_b"] = capture_b  # noqa: SLF001

        parallel_stream = _make_parallel_tool_calls_stream_anthropic_style(
            [
                ("toolu_A", "tool_a", '{"x": 42}'),
                ("toolu_B", "tool_b", '{"x": 99}'),
            ]
        )

        with patch(
            "make_agent.agent_core.agent._acompletion_with_retry",
            _mock_acompletion_with_retry(parallel_stream, _make_text_stream("done")),
        ):
            await agent.arun("run both tools")

        assert received.get("tool_a") == {"x": 42}, f"tool_a got wrong args: {received.get('tool_a')}"
        assert received.get("tool_b") == {"x": 99}, f"tool_b got wrong args: {received.get('tool_b')}"


class TestAnthropicEmptyArguments:
    """Anthropic tools with no arguments may stream only a start event (no input_json_delta).

    The start chunk from any_llm sets arguments="" and no further delta arrives,
    leaving the accumulated arguments as "".  json.loads("") raises JSONDecodeError;
    the fix is to treat "" as "{}".
    """

    def _make_agent(self, tmp_path):
        from make_agent.agent_core import Agent, AgentConfig
        from make_agent.memory import Memory
        from make_agent.skill_backend import MakefileSkillBackend
        from make_agent.tool_handler import ToolHandler

        memory = Memory(tmp_path / "memory.db")
        tool_handler = ToolHandler(MakefileSkillBackend(str(tmp_path), base_dir=tmp_path), memory)
        agent = Agent(
            AgentConfig(system_prompt="You are a helper.", model="anthropic/claude-3-5-haiku-20241022", skills_dir=str(tmp_path)),
            memory,
            tool_handler,
        )
        agent._tool_handler._schemas.append(  # noqa: SLF001
            {
                "type": "function",
                "function": {
                    "name": "list_skills",
                    "description": "List all available skills.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        )
        agent._tool_handler._executors["list_skills"] = lambda **_: "skill1, skill2"  # noqa: SLF001
        return agent

    async def test_empty_arguments_string_does_not_crash(self, tmp_path):
        """A tool call whose arguments accumulate to '' must not raise JSONDecodeError.

        The error surfaces on the *second* LLM request when _convert_messages_for_anthropic
        calls json.loads() on the stored arguments string — so we verify the full two-turn
        round-trip completes successfully.
        """
        agent = self._make_agent(tmp_path)

        with patch(
            "make_agent.agent_core.agent._acompletion_with_retry",
            _mock_acompletion_with_retry(
                _make_tool_call_stream_empty_args("toolu_1", "list_skills"),
                _make_text_stream("here are your skills"),
            ),
        ):
            result = await agent.arun("list skills")

        assert result == "here are your skills"
        tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert "malformed" not in tool_msgs[0]["content"]

        # The stored assistant message must have valid JSON arguments so the
        # next call to _convert_messages_for_anthropic doesn't raise.
        assistant_msgs = [m for m in agent.messages if m.get("role") == "assistant" and "tool_calls" in m]
        assert assistant_msgs
        stored_args = assistant_msgs[0]["tool_calls"][0]["function"]["arguments"]
        import json
        assert json.loads(stored_args) == {}, f"stored arguments must be valid JSON '{{}}', got {stored_args!r}"


# ── _prune_skill_messages ─────────────────────────────────────────────────────


def _tc(tc_id: str, name: str, args: str = "{}") -> dict:
    """Build a single tool-call entry for an assistant message."""
    return {"id": tc_id, "type": "function", "function": {"name": name, "arguments": args}}


def _assistant_tc(*tool_calls, content: str = "") -> dict:
    return {"role": "assistant", "content": content, "tool_calls": list(tool_calls)}


def _tool_result(tc_id: str, content: str = "result") -> dict:
    return {"role": "tool", "tool_call_id": tc_id, "content": content}


def _user(content: str = "hello") -> dict:
    return {"role": "user", "content": content}


def _system(content: str = "system") -> dict:
    return {"role": "system", "content": content}


def _assistant_text(content: str = "ok") -> dict:
    return {"role": "assistant", "content": content}


class TestPruneSkillMessages:
    def test_empty_messages_returns_empty(self):
        assert _prune_skill_messages([]) == []

    def test_no_skill_calls_unchanged(self):
        msgs = [_system(), _user(), _assistant_text()]
        assert _prune_skill_messages(msgs) == msgs

    def test_single_list_skills_call_kept(self):
        msgs = [
            _user(),
            _assistant_tc(_tc("t1", "list_skills")),
            _tool_result("t1", "skill-a"),
        ]
        result = _prune_skill_messages(msgs)
        assert result == msgs

    def test_two_list_skills_calls_keeps_last(self):
        msgs = [
            _user("first"),
            _assistant_tc(_tc("t1", "list_skills")),
            _tool_result("t1", "old listing"),
            _user("second"),
            _assistant_tc(_tc("t2", "list_skills")),
            _tool_result("t2", "new listing"),
        ]
        result = _prune_skill_messages(msgs)
        ids_present = {m.get("tool_call_id") for m in result if m.get("role") == "tool"}
        assert "t2" in ids_present
        assert "t1" not in ids_present

    def test_two_read_skill_calls_keeps_last(self):
        msgs = [
            _assistant_tc(_tc("r1", "read_skill", '{"name":"foo"}')),
            _tool_result("r1", "old content"),
            _assistant_tc(_tc("r2", "read_skill", '{"name":"foo"}')),
            _tool_result("r2", "new content"),
        ]
        result = _prune_skill_messages(msgs)
        ids_present = {m.get("tool_call_id") for m in result if m.get("role") == "tool"}
        assert "r2" in ids_present
        assert "r1" not in ids_present

    def test_mixed_skill_and_other_tools_preserved(self):
        msgs = [
            _assistant_tc(_tc("t1", "list_skills"), _tc("e1", "execute_skill", '{"name":"x","command":"make"}')),
            _tool_result("t1", "listing"),
            _tool_result("e1", "executed"),
            _assistant_tc(_tc("t2", "list_skills")),
            _tool_result("t2", "new listing"),
        ]
        result = _prune_skill_messages(msgs)
        ids_present = {m.get("tool_call_id") for m in result if m.get("role") == "tool"}
        # Old list_skills dropped, execute_skill and new list_skills kept
        assert "t1" not in ids_present
        assert "e1" in ids_present
        assert "t2" in ids_present

    def test_assistant_msg_with_only_old_skill_call_is_dropped(self):
        msgs = [
            _assistant_tc(_tc("t1", "list_skills")),
            _tool_result("t1", "old"),
            _assistant_tc(_tc("t2", "list_skills")),
            _tool_result("t2", "new"),
        ]
        result = _prune_skill_messages(msgs)
        assistant_msgs = [m for m in result if m.get("role") == "assistant"]
        # First assistant msg (only old list_skills) should be gone
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["tool_calls"][0]["id"] == "t2"

    def test_assistant_msg_with_text_and_old_skill_call_keeps_text(self):
        msgs = [
            _assistant_tc(_tc("t1", "list_skills"), content="thinking..."),
            _tool_result("t1", "old"),
            _assistant_tc(_tc("t2", "list_skills")),
            _tool_result("t2", "new"),
        ]
        result = _prune_skill_messages(msgs)
        assistant_msgs = [m for m in result if m.get("role") == "assistant"]
        # The message with content "thinking..." should be kept without tool_calls
        text_msgs = [m for m in assistant_msgs if "tool_calls" not in m]
        assert any(m.get("content") == "thinking..." for m in text_msgs)

    def test_system_user_messages_always_kept(self):
        msgs = [
            _system("system prompt"),
            _user("user msg"),
            _assistant_tc(_tc("t1", "list_skills")),
            _tool_result("t1", "old"),
            _assistant_tc(_tc("t2", "list_skills")),
            _tool_result("t2", "new"),
        ]
        result = _prune_skill_messages(msgs)
        assert result[0] == _system("system prompt")
        assert result[1] == _user("user msg")


# ── _flatten_messages_for_summary ─────────────────────────────────────────────


class TestFlattenMessagesForSummary:
    def test_empty_returns_empty(self):
        assert _flatten_messages_for_summary([]) == []

    def test_system_messages_dropped(self):
        msgs = [_system("sys"), _user("hi"), _assistant_text("hello")]
        result = _flatten_messages_for_summary(msgs)
        assert not any(m.get("role") == "system" for m in result)

    def test_user_messages_passed_through(self):
        msgs = [_user("tell me something")]
        result = _flatten_messages_for_summary(msgs)
        assert result == [{"role": "user", "content": "tell me something"}]

    def test_assistant_text_passed_through(self):
        msgs = [_assistant_text("here you go")]
        result = _flatten_messages_for_summary(msgs)
        assert result == [{"role": "assistant", "content": "here you go"}]

    def test_tool_role_messages_dropped(self):
        msgs = [
            _assistant_tc(_tc("t1", "execute_skill", '{"name":"x","command":"make"}')),
            _tool_result("t1", "build output"),
        ]
        result = _flatten_messages_for_summary(msgs)
        assert not any(m.get("role") == "tool" for m in result)

    def test_tool_calls_inlined_into_assistant_content(self):
        msgs = [
            _assistant_tc(_tc("t1", "execute_skill", '{"name":"x"}')),
            _tool_result("t1", "output text"),
        ]
        result = _flatten_messages_for_summary(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "execute_skill" in result[0]["content"]
        assert "output text" in result[0]["content"]

    def test_assistant_with_text_and_tool_call_combines_both(self):
        msgs = [
            _assistant_tc(_tc("t1", "list_skills"), content="Let me check…"),
            _tool_result("t1", "skill-a\nskill-b"),
        ]
        result = _flatten_messages_for_summary(msgs)
        assert len(result) == 1
        content = result[0]["content"]
        assert "Let me check…" in content
        assert "list_skills" in content
        assert "skill-a" in content

    def test_no_tool_calls_no_text_assistant_dropped(self):
        # An assistant message with no content and no tool_calls produces nothing.
        msgs = [{"role": "assistant", "content": "", "tool_calls": []}]
        result = _flatten_messages_for_summary(msgs)
        assert result == []

    def test_full_conversation_with_tools_only_user_and_assistant_remain(self):
        msgs = [
            _system("sys"),
            _user("list skills"),
            _assistant_tc(_tc("t1", "list_skills")),
            _tool_result("t1", "brainstorming"),
            _assistant_text("I see brainstorming."),
            _user("run it"),
            _assistant_tc(_tc("t2", "execute_skill", '{"name":"brainstorming","command":"make"}')),
            _tool_result("t2", "done"),
            _assistant_text("All done."),
        ]
        result = _flatten_messages_for_summary(msgs)
        roles = [m["role"] for m in result]
        assert "system" not in roles
        assert "tool" not in roles
        assert roles.count("user") == 2
        # execute_skill result should be inlined
        assert any("execute_skill" in m["content"] for m in result if m["role"] == "assistant")


# ── AgentManager auto-compact ─────────────────────────────────────────────────


class TestAgentManagerAutoCompact:
    def _make_manager_and_session(self, tmp_path, compact_threshold: int = 100):
        from make_agent.agent_core import Agent, AgentConfig, AgentManager
        from make_agent.memory import Memory
        from make_agent.skill_backend import MakefileSkillBackend
        from make_agent.tool_handler import ToolHandler

        memory = Memory(tmp_path / "memory.db")
        tool_handler = ToolHandler(MakefileSkillBackend(str(tmp_path), base_dir=tmp_path), memory)
        config = AgentConfig(
            system_prompt="You are a helper.",
            model="openai/gpt-4o-mini",
            compact_threshold=compact_threshold,
        )
        manager = AgentManager(memory, tool_handler)
        session_id = manager.create_session(config)
        return manager, session_id

    async def test_compact_not_triggered_below_threshold(self, tmp_path):
        manager, sid = self._make_manager_and_session(tmp_path, compact_threshold=10_000)
        agent_before = manager.get_agent(sid)
        agent_before._last_prompt_tokens = 5_000  # below threshold

        with patch(
            "make_agent.agent_core.agent._acompletion_with_retry",
            _mock_acompletion_with_retry(_make_text_stream("hello")),
        ):
            result = await manager.arun_agent(sid, "hi")

        assert result == "hello"
        # Agent should be the same object — no replacement happened
        assert manager.get_agent(sid) is agent_before

    async def test_compact_triggered_above_threshold(self, tmp_path):
        manager, sid = self._make_manager_and_session(tmp_path, compact_threshold=100)
        agent_before = manager.get_agent(sid)
        agent_before._last_prompt_tokens = 200  # above threshold

        summary_stream = _make_text_stream("Previous work: user asked to list skills.")
        reply_stream = _make_text_stream("continuing now")

        with patch(
            "make_agent.agent_core.agent._acompletion_with_retry",
            _mock_acompletion_with_retry(summary_stream, reply_stream),
        ):
            result = await manager.arun_agent(sid, "what did we do?")

        assert result == "continuing now"
        # Agent should be a fresh object
        agent_after = manager.get_agent(sid)
        assert agent_after is not agent_before

    async def test_compact_injects_summary_into_fresh_agent(self, tmp_path):
        manager, sid = self._make_manager_and_session(tmp_path, compact_threshold=100)
        agent_before = manager.get_agent(sid)
        agent_before._last_prompt_tokens = 200

        summary_stream = _make_text_stream("Summary: user asked about skills.")
        reply_stream = _make_text_stream("ok")

        with patch(
            "make_agent.agent_core.agent._acompletion_with_retry",
            _mock_acompletion_with_retry(summary_stream, reply_stream),
        ):
            await manager.arun_agent(sid, "continue")

        fresh_agent = manager.get_agent(sid)
        contents = [m.get("content", "") for m in fresh_agent.messages]
        assert any("Summary: user asked about skills." in c for c in contents)

    async def test_compact_disabled_when_threshold_zero(self, tmp_path):
        manager, sid = self._make_manager_and_session(tmp_path, compact_threshold=0)
        agent_before = manager.get_agent(sid)
        agent_before._last_prompt_tokens = 999_999  # way above any threshold

        with patch(
            "make_agent.agent_core.agent._acompletion_with_retry",
            _mock_acompletion_with_retry(_make_text_stream("no compact")),
        ):
            result = await manager.arun_agent(sid, "hi")

        assert result == "no compact"
        assert manager.get_agent(sid) is agent_before

    async def test_compact_yields_compact_event(self, tmp_path):
        manager, sid = self._make_manager_and_session(tmp_path, compact_threshold=100)
        agent_before = manager.get_agent(sid)
        agent_before._last_prompt_tokens = 250

        summary_stream = _make_text_stream("summary text")
        reply_stream = _make_text_stream("reply")

        events = []
        with patch(
            "make_agent.agent_core.agent._acompletion_with_retry",
            _mock_acompletion_with_retry(summary_stream, reply_stream),
        ):
            async for event in manager.astream_agent(sid, "continue"):
                events.append(event)

        compact_events = [e for e in events if isinstance(e, CompactEvent)]
        assert len(compact_events) == 1
        assert compact_events[0].prompt_tokens == 250
        assert compact_events[0].threshold == 100

