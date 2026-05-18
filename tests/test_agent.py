"""Tests for rate limit retry logic — _parse_retry_after and _acompletion_with_retry."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call, patch

import any_llm
import pytest
from make_agent.agent import _acompletion_with_retry, _parse_retry_after


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
        with patch("make_agent.agent.any_llm.acompletion", AsyncMock(return_value=stream)) as mock_c:
            result = await _acompletion_with_retry("model", [], {}, max_retries=3)
        assert result is stream
        mock_c.assert_called_once()

    async def test_retries_on_rate_limit_then_succeeds(self):
        err = _make_rate_limit_error(retry_after=10)
        stream = _make_empty_stream()
        with patch("make_agent.agent.any_llm.acompletion", AsyncMock(side_effect=[err, err, stream])):
            with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
                result = await _acompletion_with_retry("model", [], {}, max_retries=3)
        assert result is stream
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(10.0)

    async def test_exponential_backoff_without_header(self):
        err = _make_rate_limit_error()
        stream = _make_empty_stream()
        with patch("make_agent.agent.any_llm.acompletion", AsyncMock(side_effect=[err, err, stream])):
            with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
                await _acompletion_with_retry("model", [], {}, max_retries=3)
        assert mock_sleep.call_args_list == [call(1), call(2)]

    async def test_exponential_backoff_capped_at_60s(self):
        err = _make_rate_limit_error()
        stream = _make_empty_stream()
        side_effects = [err] * 7 + [stream]
        with patch("make_agent.agent.any_llm.acompletion", AsyncMock(side_effect=side_effects)):
            with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
                await _acompletion_with_retry("model", [], {}, max_retries=10)
        waits = [c.args[0] for c in mock_sleep.call_args_list]
        assert all(w <= 60 for w in waits)
        assert waits[6] == 60  # 2^6=64 capped to 60

    async def test_raises_after_max_retries_exhausted(self):
        err = _make_rate_limit_error(retry_after=1)
        with patch("make_agent.agent.any_llm.acompletion", AsyncMock(side_effect=err)):
            with patch("asyncio.sleep", AsyncMock()):
                with pytest.raises(any_llm.RateLimitError):
                    await _acompletion_with_retry("model", [], {}, max_retries=2)

    async def test_total_calls_equals_max_retries_plus_one(self):
        err = _make_rate_limit_error(retry_after=1)
        with patch("make_agent.agent.any_llm.acompletion", AsyncMock(side_effect=err)) as mock_c:
            with patch("asyncio.sleep", AsyncMock()):
                with pytest.raises(any_llm.RateLimitError):
                    await _acompletion_with_retry("model", [], {}, max_retries=3)
        assert mock_c.call_count == 4  # 1 initial + 3 retries

    async def test_zero_max_retries_raises_immediately(self):
        err = _make_rate_limit_error(retry_after=1)
        with patch("make_agent.agent.any_llm.acompletion", AsyncMock(side_effect=err)):
            with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
                with pytest.raises(any_llm.RateLimitError):
                    await _acompletion_with_retry("model", [], {}, max_retries=0)
        mock_sleep.assert_not_called()


# ── Agent safety guards ───────────────────────────────────────────────────────


class TestAgentSafetyGuards:
    def _make_agent(self, tmp_path):
        from make_agent.agent import Agent, AgentConfig
        from make_agent.memory import Memory

        memory = Memory(tmp_path / "memory.db")
        agent = Agent(AgentConfig(system_prompt="You are a helper.", model="openai/gpt-4o-mini", skills_dir=str(tmp_path)), memory)
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
            "make_agent.agent._acompletion_with_retry",
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
            patch("make_agent.agent._MAX_MODEL_TURNS_PER_REQUEST", 2),
            patch("make_agent.agent._acompletion_with_retry", _always_tool_call),
        ):
            with pytest.raises(RuntimeError, match="model turns"):
                await agent.arun("loop forever")


class TestAssistantMessageContent:
    """Assistant messages with tool calls must never have content=None (breaks Ollama provider)."""

    async def test_tool_call_without_text_has_empty_string_content(self, tmp_path):
        """When the LLM streams a tool call with no text, the assistant message content must be ''."""
        from make_agent.agent import Agent, AgentConfig
        from make_agent.memory import Memory

        memory = Memory(tmp_path / "memory.db")
        agent = Agent(AgentConfig(system_prompt="You are a helper.", model="openai/gpt-4o-mini", skills_dir=str(tmp_path)), memory)
        # Inject say_hi as a known builtin tool
        agent._tool_handler._schemas.append(  # noqa: SLF001
            {
                "type": "function",
                "function": {"name": "say_hi", "description": "Say hi.", "parameters": {"type": "object", "properties": {}, "required": []}},
            }
        )
        agent._tool_handler._executors["say_hi"] = lambda **_: "hi"  # noqa: SLF001

        with patch(
            "make_agent.agent._acompletion_with_retry",
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
