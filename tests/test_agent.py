"""Tests for rate limit retry logic — _parse_retry_after and _completion_with_retry."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import any_llm
import pytest
from make_agent.agent import _completion_with_retry, _parse_retry_after


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


def _make_success_response() -> MagicMock:
    resp = MagicMock()
    resp.choices[0].message.tool_calls = None
    resp.choices[0].message.content = "done"
    return resp


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


class TestCompletionWithRetry:
    def test_succeeds_on_first_attempt(self):
        success = _make_success_response()
        with patch("make_agent.agent.any_llm.completion", return_value=success) as mock_c:
            result = _completion_with_retry("model", [], {}, max_retries=3)
        assert result is success
        mock_c.assert_called_once()

    def test_retries_on_rate_limit_then_succeeds(self):
        err = _make_rate_limit_error(retry_after=10)
        success = _make_success_response()
        with patch("make_agent.agent.any_llm.completion", side_effect=[err, err, success]):
            with patch("make_agent.agent.time.sleep") as mock_sleep:
                result = _completion_with_retry("model", [], {}, max_retries=3)
        assert result is success
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(10.0)

    def test_exponential_backoff_without_header(self):
        err = _make_rate_limit_error()  # no retry-after header
        success = _make_success_response()
        with patch("make_agent.agent.any_llm.completion", side_effect=[err, err, success]):
            with patch("make_agent.agent.time.sleep") as mock_sleep:
                _completion_with_retry("model", [], {}, max_retries=3)
        # attempt 0 → 2^0 = 1s, attempt 1 → 2^1 = 2s
        assert mock_sleep.call_args_list == [call(1), call(2)]

    def test_exponential_backoff_capped_at_60s(self):
        err = _make_rate_limit_error()
        success = _make_success_response()
        # need 7 failures to reach 2^6=64 > 60; cap kicks in at attempt 6
        side_effects = [err] * 7 + [success]
        with patch("make_agent.agent.any_llm.completion", side_effect=side_effects):
            with patch("make_agent.agent.time.sleep") as mock_sleep:
                _completion_with_retry("model", [], {}, max_retries=10)
        waits = [c.args[0] for c in mock_sleep.call_args_list]
        assert all(w <= 60 for w in waits)
        assert waits[6] == 60  # 2^6=64 capped to 60

    def test_raises_after_max_retries_exhausted(self):
        err = _make_rate_limit_error(retry_after=1)
        with patch("make_agent.agent.any_llm.completion", side_effect=err):
            with patch("make_agent.agent.time.sleep"):
                with pytest.raises(any_llm.RateLimitError):
                    _completion_with_retry("model", [], {}, max_retries=2)

    def test_total_calls_equals_max_retries_plus_one(self):
        err = _make_rate_limit_error(retry_after=1)
        with patch("make_agent.agent.any_llm.completion", side_effect=err) as mock_c:
            with patch("make_agent.agent.time.sleep"):
                with pytest.raises(any_llm.RateLimitError):
                    _completion_with_retry("model", [], {}, max_retries=3)
        assert mock_c.call_count == 4  # 1 initial + 3 retries

    def test_zero_max_retries_raises_immediately(self):
        err = _make_rate_limit_error(retry_after=1)
        with patch("make_agent.agent.any_llm.completion", side_effect=err):
            with patch("make_agent.agent.time.sleep") as mock_sleep:
                with pytest.raises(any_llm.RateLimitError):
                    _completion_with_retry("model", [], {}, max_retries=0)
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

    def test_run_agent_dispatched_via_call(self, tmp_path):
        """Agent.__call__ runs the sub-agent and returns its response as a tool result."""
        from unittest.mock import MagicMock, patch

        from make_agent.agent import Agent, AgentConfig

        (tmp_path / "specialist.mk").write_text("define SYSTEM_PROMPT\nSpecialist.\nendef\n")
        mf = tmp_path / "Makefile"
        mf.write_text("define SYSTEM_PROMPT\nOrchestrator.\nendef\n")
        agent = Agent(AgentConfig(makefile_path=mf, model="openai/gpt-4o-mini", agents_dir=str(tmp_path)), None)

        # Patch _run_agent to return a known string without hitting the LLM
        with patch.object(agent, "_run_agent", return_value="specialist done") as mock_run:
            # Build a fake LLM response that calls run_agent then returns text
            tool_call = MagicMock()
            tool_call.id = "tc1"
            tool_call.function.name = "run_agent"
            tool_call.function.arguments = '{"name": "specialist", "prompt": "go"}'

            tool_response = MagicMock()
            tool_response.choices[0].message.tool_calls = [tool_call]

            final_response = MagicMock()
            final_response.choices[0].message.tool_calls = None
            final_response.choices[0].message.content = "all done"

            with patch("make_agent.agent.any_llm.completion", side_effect=[tool_response, final_response]):
                result = agent("delegate to specialist")

        mock_run.assert_called_once()
        assert result == "all done"
