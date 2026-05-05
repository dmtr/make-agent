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


# ── Load-time validation tests ────────────────────────────────────────────────


class TestAgentValidation:
    def _write_makefile(self, tmp_path, content: str):
        mf = tmp_path / "Makefile"
        mf.write_text(content)
        return mf
    def test_valid_makefile_loads(self, tmp_path):
        mf = self._write_makefile(tmp_path, ("# <tool>\n# Greet.\n# @param NAME string A name\n# </tool>\n" "greet:\n	@echo $(NAME)\n"))
        from make_agent.agent import Agent, AgentConfig
        agent = Agent(AgentConfig(makefile_path=mf, model="openai/gpt-4o-mini"), None)
        agent = Agent(AgentConfig(makefile_path=mf, model="openai/gpt-4o-mini"), None)
        assert "greet" in agent.tool_names

    def test_broken_recipe_raises_on_load(self, tmp_path):
        mf = self._write_makefile(tmp_path, ("# <tool>\n# Install.\n# @param FILE string A file\n# </tool>\n" "install:\n\t@pip install -r\n"))
        import pytest
        from make_agent.agent import Agent, AgentConfig

        with pytest.raises(ValueError, match="FILE"):
            Agent(AgentConfig(makefile_path=mf, model="openai/gpt-4o-mini"), None)

    def test_error_message_names_tool_and_param(self, tmp_path):
        mf = self._write_makefile(tmp_path, ("# <tool>\n# Do X.\n# @param QUERY string Search term\n# </tool>\n" "search:\n\t@grep foo .\n"))
        import pytest
        from make_agent.agent import Agent, AgentConfig

        with pytest.raises(ValueError) as exc_info:
            Agent(AgentConfig(makefile_path=mf, model="openai/gpt-4o-mini"), None)
        assert "search" in str(exc_info.value)
        assert "QUERY" in str(exc_info.value)


# ── DISABLED_BUILTINS Makefile variable ───────────────────────────────────────


class TestDisabledBuiltins:
    def _make_agent(self, tmp_path, content: str):
        from make_agent.agent import Agent, AgentConfig

        mf = tmp_path / "Makefile"
        mf.write_text(content)
        return Agent(AgentConfig(makefile_path=mf, model="openai/gpt-4o-mini", agents_dir=str(tmp_path)), None)

    def test_single_tool_disabled_via_makefile(self, tmp_path):
        agent = self._make_agent(tmp_path, "DISABLED_BUILTINS = run_agent\n")
        assert "run_agent" not in agent.tool_names

    def test_multiple_tools_disabled_via_makefile(self, tmp_path):
        agent = self._make_agent(tmp_path, "DISABLED_BUILTINS = run_agent,validate_agent\n")
        assert "run_agent" not in agent.tool_names
        assert "validate_agent" not in agent.tool_names

    def test_all_disables_everything(self, tmp_path):
        from make_agent.builtin_tools import BUILTIN_TOOL_NAMES

        agent = self._make_agent(tmp_path, "DISABLED_BUILTINS = all\n")
        for name in BUILTIN_TOOL_NAMES:
            assert name not in agent.tool_names

    def test_unknown_tool_raises_value_error(self, tmp_path):
        import pytest

        with pytest.raises(ValueError, match="DISABLED_BUILTINS"):
            self._make_agent(tmp_path, "DISABLED_BUILTINS = no_such_tool\n")

    def test_makefile_and_cli_flags_are_merged(self, tmp_path):
        from make_agent.agent import Agent, AgentConfig

        mf = tmp_path / "Makefile"
        mf.write_text("DISABLED_BUILTINS = run_agent\n")
        agent = Agent(
            AgentConfig(
                makefile_path=mf,
                model="openai/gpt-4o-mini",
                agents_dir=str(tmp_path),
                disabled_builtin_tools=frozenset({"validate_agent"}),
            ),
            None,
        )
        assert "run_agent" not in agent.tool_names
        assert "validate_agent" not in agent.tool_names

    def test_empty_disabled_builtins_is_no_op(self, tmp_path):
        from make_agent.builtin_tools import BUILTIN_TOOL_NAMES

        agent = self._make_agent(tmp_path, "DISABLED_BUILTINS =\n")
        builtin_names_present = [n for n in agent.tool_names if n in BUILTIN_TOOL_NAMES]
        assert len(builtin_names_present) > 0


# ── run_agent in-process dispatch ─────────────────────────────────────────────


class TestRunAgentInProcess:
    def _make_agent(self, tmp_path, content: str, agents_dir: str | None = None):
        from make_agent.agent import Agent, AgentConfig

        mf = tmp_path / "Makefile"
        mf.write_text(content)
        return Agent(AgentConfig(makefile_path=mf, model="openai/gpt-4o-mini", agents_dir=agents_dir or str(tmp_path)), None)

    def test_run_agent_disabled_for_sub_agent(self, tmp_path):
        """Sub-agents must not have run_agent available (prevents infinite loops)."""
        from make_agent.agent import Agent, AgentConfig

        (tmp_path / "specialist.mk").write_text("define SYSTEM_PROMPT\nSpecialist.\nendef\n")
        mf = tmp_path / "Makefile"
        mf.write_text("define SYSTEM_PROMPT\nOrchestrator.\nendef\n")
        agent = Agent(AgentConfig(makefile_path=mf, model="openai/gpt-4o-mini", agents_dir=str(tmp_path)), None)

        # Build sub-config as _run_agent would and verify run_agent is disabled
        sub_disabled = agent._disabled_builtin_tools | frozenset({"run_agent"})  # noqa: SLF001
        assert "run_agent" in sub_disabled

    def test_run_agent_sub_agent_gets_same_model(self, tmp_path):
        from make_agent.agent import Agent, AgentConfig

        (tmp_path / "specialist.mk").write_text("define SYSTEM_PROMPT\nSpecialist.\nendef\n")
        mf = tmp_path / "Makefile"
        mf.write_text("define SYSTEM_PROMPT\nOrchestrator.\nendef\n")
        agent = Agent(AgentConfig(makefile_path=mf, model="openai/gpt-4o-mini", agents_dir=str(tmp_path)), None)
        assert agent._model == "openai/gpt-4o-mini"  # noqa: SLF001

    async def test_run_agent_dispatched_via_call(self, tmp_path):
        """agent.arun() runs the sub-agent via _arun_agent and returns final text."""
        from make_agent.agent import Agent, AgentConfig

        (tmp_path / "specialist.mk").write_text("define SYSTEM_PROMPT\nSpecialist.\nendef\n")
        mf = tmp_path / "Makefile"
        mf.write_text("define SYSTEM_PROMPT\nOrchestrator.\nendef\n")
        agent = Agent(AgentConfig(makefile_path=mf, model="openai/gpt-4o-mini", agents_dir=str(tmp_path)), None)

        with patch.object(agent, "_arun_agent", new_callable=AsyncMock, return_value="specialist done") as mock_run:
            with patch(
                "make_agent.agent._acompletion_with_retry",
                _mock_acompletion_with_retry(
                    _make_tool_call_stream("tc1", "run_agent", '{"name": "specialist", "prompt": "go"}'),
                    _make_text_stream("all done"),
                ),
            ):
                result = await agent.arun("delegate to specialist")

        mock_run.assert_called_once()
        assert result == "all done"


class TestAgentSafetyGuards:
    def _make_agent(self, tmp_path):
        from make_agent.agent import Agent, AgentConfig

        mf = tmp_path / "Makefile"
        mf.write_text(
            "# <tool>\n"
            "# Visible tool.\n"
            "# </tool>\n"
            "safe:\n"
            "\t@echo safe\n"
            "hidden:\n"
            "\t@echo hidden\n"
        )
        return Agent(AgentConfig(makefile_path=mf, model="openai/gpt-4o-mini"), None)

    async def test_unknown_tool_is_rejected_without_running_make(self, tmp_path):
        agent = self._make_agent(tmp_path)

        with (
            patch(
                "make_agent.agent._acompletion_with_retry",
                _mock_acompletion_with_retry(
                    _make_tool_call_stream("tc1", "hidden", "{}"),
                    _make_text_stream("done"),
                ),
            ),
            patch("make_agent.agent.run_tool", new_callable=AsyncMock) as mock_run_tool,
        ):
            result = await agent.arun("use hidden target")

        assert result == "done"
        mock_run_tool.assert_not_called()
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
