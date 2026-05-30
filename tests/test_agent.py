"""Tests for provider retry logic and agent loop behaviour."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call, patch

import anthropic
import pytest
from make_agent.agent_core import AgentConfig
from make_agent.provider import TextDelta, ToolCallDelta, ToolCallStart, UsageDelta
from make_agent.provider.anthropic import MAX_RETRIES, _parse_retry_after


# ── MockProvider ──────────────────────────────────────────────────────────────


class MockProvider:
    """A test provider that returns pre-built StreamChunk sequences."""

    def __init__(self, *chunk_sequences: list) -> None:
        self._sequences = list(chunk_sequences)
        self._call_count = 0

    async def astream(self, model, messages, tools, max_tokens, **kwargs):
        seq = self._sequences[self._call_count % len(self._sequences)]
        self._call_count += 1
        for chunk in seq:
            yield chunk


# ── Stream helpers ────────────────────────────────────────────────────────────


def _text_chunks(content: str, input_tokens: int = 0, output_tokens: int = 0) -> list:
    chunks = []
    if input_tokens:
        chunks.append(UsageDelta(input_tokens=input_tokens, output_tokens=0))
    chunks.append(TextDelta(text=content))
    if output_tokens:
        chunks.append(UsageDelta(input_tokens=0, output_tokens=output_tokens))
    return chunks


def _tool_call_chunks(tool_id: str, tool_name: str, arguments: str, index: int = 0) -> list:
    chunks = [ToolCallStart(index=index, id=tool_id, name=tool_name)]
    if arguments:
        chunks.append(ToolCallDelta(index=index, args_delta=arguments))
    return chunks


# ── TestParseRetryAfter ───────────────────────────────────────────────────────


def _make_rate_limit_error(
    retry_after: float | None = None,
    retry_after_ms: float | None = None,
) -> anthropic.RateLimitError:
    headers: dict[str, str] = {}
    if retry_after is not None:
        headers["retry-after"] = str(retry_after)
    if retry_after_ms is not None:
        headers["retry-after-ms"] = str(retry_after_ms)
    fake_response = MagicMock()
    fake_response.headers = headers
    return anthropic.RateLimitError(
        message="rate limit exceeded",
        response=fake_response,
        body={},
    )


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
        err = MagicMock()
        err.response = None
        assert _parse_retry_after(err) is None


# ── TestAnthropicProviderRetry ────────────────────────────────────────────────


class TestAnthropicProviderRetry:
    def _make_provider(self):
        from make_agent.provider.anthropic import AnthropicProvider
        return AnthropicProvider()

    async def test_succeeds_on_first_attempt(self):
        provider = self._make_provider()
        mock_stream = MagicMock()
        with patch.object(provider._client.messages, "create", AsyncMock(return_value=mock_stream)) as mock_c:
            result = await provider._create_with_retry({})
        assert result is mock_stream
        mock_c.assert_called_once()

    async def test_retries_on_rate_limit_then_succeeds(self):
        provider = self._make_provider()
        err = _make_rate_limit_error(retry_after=10)
        mock_stream = MagicMock()
        with patch.object(provider._client.messages, "create", AsyncMock(side_effect=[err, err, mock_stream])):
            with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
                result = await provider._create_with_retry({})
        assert result is mock_stream
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(10.0)

    async def test_exponential_backoff_without_header(self):
        provider = self._make_provider()
        err = _make_rate_limit_error()
        mock_stream = MagicMock()
        with patch.object(provider._client.messages, "create", AsyncMock(side_effect=[err, err, mock_stream])):
            with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
                await provider._create_with_retry({})
        assert mock_sleep.call_args_list == [call(1), call(2)]

    async def test_raises_after_max_retries_exhausted(self):
        provider = self._make_provider()
        err = _make_rate_limit_error(retry_after=1)
        with patch.object(provider._client.messages, "create", AsyncMock(side_effect=err)):
            with patch("asyncio.sleep", AsyncMock()):
                with pytest.raises(anthropic.RateLimitError):
                    await provider._create_with_retry({})

    async def test_total_calls_equals_max_retries_plus_one(self):
        provider = self._make_provider()
        err = _make_rate_limit_error(retry_after=1)
        with patch.object(provider._client.messages, "create", AsyncMock(side_effect=err)) as mock_c:
            with patch("asyncio.sleep", AsyncMock()):
                with pytest.raises(anthropic.RateLimitError):
                    await provider._create_with_retry({})
        assert mock_c.call_count == MAX_RETRIES + 1

    async def test_zero_max_retries_raises_immediately(self):
        provider = self._make_provider()
        err = _make_rate_limit_error(retry_after=1)
        with patch.object(provider._client.messages, "create", AsyncMock(side_effect=err)):
            with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
                with patch("make_agent.provider.anthropic.MAX_RETRIES", 0):
                    with pytest.raises(anthropic.RateLimitError):
                        await provider._create_with_retry({})
        mock_sleep.assert_not_called()


# ── Agent safety guards ───────────────────────────────────────────────────────


class TestAgentSafetyGuards:
    def _make_agent(self, tmp_path, provider):
        from make_agent.agent_core import Agent, AgentConfig
        from make_agent.memory import Memory
        from make_agent.skill_backend import MakefileSkillBackend
        from make_agent.tool_handler import ToolHandler

        memory = Memory(tmp_path / "memory.db")
        tool_handler = ToolHandler(
            MakefileSkillBackend(str(tmp_path), base_dir=tmp_path), memory
        )
        agent = Agent(
            AgentConfig(
                system_prompt="You are a helper.",
                model="claude-3-5-haiku-20241022",
                skills_dir=str(tmp_path),
                provider=provider,
            ),
            tool_handler,
        )
        # Inject a custom tool to give the agent a known tool set
        agent._tool_handler._schemas.append(  # noqa: SLF001
            {
                "type": "function",
                "function": {
                    "name": "safe",
                    "description": "A safe tool.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        )
        agent._tool_handler._executors["safe"] = lambda **_: "ok"  # noqa: SLF001
        return agent

    async def test_unknown_tool_is_rejected_without_running_make(self, tmp_path):
        provider = MockProvider(
            _tool_call_chunks("tc1", "hidden", "{}"),
            _text_chunks("done"),
        )
        agent = self._make_agent(tmp_path, provider)
        result = await agent.arun("use hidden target")

        assert result == "done"
        tool_outputs = [m["content"] for m in agent.messages if m.get("role") == "tool"]
        assert any("unknown tool: hidden" in output for output in tool_outputs)

    async def test_model_turn_limit_stops_runaway_tool_loop(self, tmp_path):
        provider = MockProvider(_tool_call_chunks("tc1", "hidden", "{}"))
        agent = self._make_agent(tmp_path, provider)

        with patch("make_agent.agent_core.loop.MAX_MODEL_TURNS_PER_REQUEST", 2):
            with pytest.raises(RuntimeError, match="model turns"):
                await agent.arun("loop forever")


class TestAssistantMessageContent:
    """Assistant messages with tool calls must never have content=None."""

    async def test_tool_call_without_text_has_empty_string_content(self, tmp_path):
        """When the LLM streams a tool call with no text, the assistant message content must be ''."""
        from make_agent.agent_core import Agent, AgentConfig
        from make_agent.memory import Memory
        from make_agent.skill_backend import MakefileSkillBackend
        from make_agent.tool_handler import ToolHandler

        memory = Memory(tmp_path / "memory.db")
        tool_handler = ToolHandler(
            MakefileSkillBackend(str(tmp_path), base_dir=tmp_path), memory
        )
        provider = MockProvider(
            _tool_call_chunks("tc1", "say_hi", "{}"),
            _text_chunks("all done"),
        )
        agent = Agent(
            AgentConfig(
                system_prompt="You are a helper.",
                model="claude-3-5-haiku-20241022",
                skills_dir=str(tmp_path),
                provider=provider,
            ),
            tool_handler,
        )
        # Inject say_hi as a known builtin tool
        agent._tool_handler._schemas.append(  # noqa: SLF001
            {
                "type": "function",
                "function": {
                    "name": "say_hi",
                    "description": "Say hi.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        )
        agent._tool_handler._executors["say_hi"] = lambda **_: "hi"  # noqa: SLF001

        result = await agent.arun("call say_hi")

        assert result == "all done"
        assistant_msgs = [
            m
            for m in agent.messages
            if m.get("role") == "assistant" and "tool_calls" in m
        ]
        assert assistant_msgs, "expected at least one assistant message with tool_calls"
        for msg in assistant_msgs:
            assert msg["content"] is not None, (
                "assistant message content must not be None"
            )
            assert isinstance(msg["content"], str)


class TestAnthropicParallelToolCalls:
    """Native Anthropic SDK assigns correct sequential indices for parallel tool calls."""

    def _make_agent(self, tmp_path, provider):
        from make_agent.agent_core import Agent, AgentConfig
        from make_agent.memory import Memory
        from make_agent.skill_backend import MakefileSkillBackend
        from make_agent.tool_handler import ToolHandler

        memory = Memory(tmp_path / "memory.db")
        tool_handler = ToolHandler(
            MakefileSkillBackend(str(tmp_path), base_dir=tmp_path), memory
        )
        agent = Agent(
            AgentConfig(
                system_prompt="You are a helper.",
                model="anthropic/claude-3-5-sonnet-20241022",
                skills_dir=str(tmp_path),
                provider=provider,
            ),
            tool_handler,
        )
        for name in ("tool_a", "tool_b"):
            agent._tool_handler._schemas.append(  # noqa: SLF001
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": f"Tool {name}.",
                        "parameters": {
                            "type": "object",
                            "properties": {"x": {"type": "integer"}},
                            "required": ["x"],
                        },
                    },
                }
            )
            agent._tool_handler._executors[name] = lambda x, _n=name: f"{_n}_result"  # noqa: SLF001
        return agent

    async def test_parallel_tool_calls_are_kept_separate(self, tmp_path):
        """Two parallel tool calls with sequential indices must each be executed."""
        parallel_chunks = (
            _tool_call_chunks("toolu_A", "tool_a", '{"x": 1}', index=0)
            + _tool_call_chunks("toolu_B", "tool_b", '{"x": 2}', index=1)
        )
        provider = MockProvider(parallel_chunks, _text_chunks("done"))
        agent = self._make_agent(tmp_path, provider)

        result = await agent.arun("run both tools")

        assert result == "done"
        tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 2, (
            f"expected 2 tool results, got {len(tool_msgs)}: {tool_msgs}"
        )
        tool_call_ids = {m["tool_call_id"] for m in tool_msgs}
        assert "toolu_A" in tool_call_ids
        assert "toolu_B" in tool_call_ids

    async def test_parallel_tool_calls_have_correct_arguments(self, tmp_path):
        """Arguments must not be concatenated across parallel tool calls."""
        parallel_chunks = (
            _tool_call_chunks("toolu_A", "tool_a", '{"x": 42}', index=0)
            + _tool_call_chunks("toolu_B", "tool_b", '{"x": 99}', index=1)
        )
        provider = MockProvider(parallel_chunks, _text_chunks("done"))
        agent = self._make_agent(tmp_path, provider)

        received: dict[str, dict] = {}

        def capture_a(**kwargs):
            received["tool_a"] = kwargs
            return "result_a"

        def capture_b(**kwargs):
            received["tool_b"] = kwargs
            return "result_b"

        agent._tool_handler._executors["tool_a"] = capture_a  # noqa: SLF001
        agent._tool_handler._executors["tool_b"] = capture_b  # noqa: SLF001

        await agent.arun("run both tools")

        assert received.get("tool_a") == {"x": 42}, (
            f"tool_a got wrong args: {received.get('tool_a')}"
        )
        assert received.get("tool_b") == {"x": 99}, (
            f"tool_b got wrong args: {received.get('tool_b')}"
        )


class TestAnthropicEmptyArguments:
    """Tools with no arguments send only a ToolCallStart (no ToolCallDelta).

    The accumulated arguments string will be ''; the loop must treat '' as '{}'.
    """

    def _make_agent(self, tmp_path, provider):
        from make_agent.agent_core import Agent, AgentConfig
        from make_agent.memory import Memory
        from make_agent.skill_backend import MakefileSkillBackend
        from make_agent.tool_handler import ToolHandler

        memory = Memory(tmp_path / "memory.db")
        tool_handler = ToolHandler(
            MakefileSkillBackend(str(tmp_path), base_dir=tmp_path), memory
        )
        agent = Agent(
            AgentConfig(
                system_prompt="You are a helper.",
                model="anthropic/claude-3-5-haiku-20241022",
                skills_dir=str(tmp_path),
                provider=provider,
            ),
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
        """A tool call with no ToolCallDelta must not raise JSONDecodeError."""
        # Only ToolCallStart, no ToolCallDelta → arguments accumulate to ""
        empty_arg_chunks = [ToolCallStart(index=0, id="toolu_1", name="list_skills")]
        provider = MockProvider(empty_arg_chunks, _text_chunks("here are your skills"))
        agent = self._make_agent(tmp_path, provider)

        result = await agent.arun("list skills")

        assert result == "here are your skills"
        tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert "malformed" not in tool_msgs[0]["content"]

        # The stored assistant message must have valid JSON arguments.
        assistant_msgs = [
            m
            for m in agent.messages
            if m.get("role") == "assistant" and "tool_calls" in m
        ]
        assert assistant_msgs
        stored_args = assistant_msgs[0]["tool_calls"][0]["function"]["arguments"]
        import json

        assert json.loads(stored_args) == {}, (
            f"stored arguments must be valid JSON '{{}}', got {stored_args!r}"
        )


# ── TestAgentSystemPromptCache ────────────────────────────────────────────────


class TestAgentSystemPromptCache:
    """The loop always stores system prompt as a plain string.

    Cache-control headers are added by the Anthropic provider internally at
    call time — not embedded in the loop's message history.
    """

    def _make_agent(self, tmp_path, model: str, use_prompt_cache: bool):
        from make_agent.agent_core import Agent
        from make_agent.memory import Memory
        from make_agent.skill_backend import PythonSkillBackend
        from make_agent.tool_handler import ToolHandler

        memory = Memory(tmp_path / "memory.db")
        tool_handler = ToolHandler(PythonSkillBackend(str(tmp_path), 60), memory)
        config = AgentConfig(
            system_prompt="You are a helpful assistant.",
            model=model,
            use_prompt_cache=use_prompt_cache,
            provider=MockProvider(),  # no actual API calls
        )
        return Agent(config, tool_handler)

    def test_no_cache_stores_plain_string(self, tmp_path):
        agent = self._make_agent(tmp_path, "anthropic/claude-3-5-haiku-20241022", False)
        system_msg = agent.messages[0]
        assert system_msg["role"] == "system"
        assert system_msg["content"] == "You are a helpful assistant."

    def test_cache_enabled_still_stores_plain_string(self, tmp_path):
        """Even with use_prompt_cache=True, the loop stores a plain string.

        Cache-control is added by AnthropicProvider.astream() at API call time.
        """
        agent = self._make_agent(tmp_path, "anthropic/claude-3-5-haiku-20241022", True)
        system_msg = agent.messages[0]
        assert system_msg["role"] == "system"
        assert system_msg["content"] == "You are a helpful assistant."

    def test_non_anthropic_model_stores_plain_string(self, tmp_path):
        agent = self._make_agent(tmp_path, "openai/gpt-4o", False)
        system_msg = agent.messages[0]
        assert system_msg["role"] == "system"
        assert system_msg["content"] == "You are a helpful assistant."


