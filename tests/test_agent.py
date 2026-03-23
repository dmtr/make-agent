"""Tests for rate limit retry logic — _parse_retry_after and _completion_with_retry."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import httpx
import litellm
import pytest

from make_agent.agent import _completion_with_retry, _parse_retry_after


def _make_rate_limit_error(
    retry_after: float | None = None,
    retry_after_ms: float | None = None,
) -> litellm.RateLimitError:
    headers: dict[str, str] = {}
    if retry_after is not None:
        headers["retry-after"] = str(retry_after)
    if retry_after_ms is not None:
        headers["retry-after-ms"] = str(retry_after_ms)
    response = httpx.Response(
        status_code=429,
        headers=headers,
        request=httpx.Request("POST", "https://api.anthropic.com"),
    )
    return litellm.RateLimitError(
        message="rate limit exceeded",
        llm_provider="anthropic",
        model="claude-haiku",
        response=response,
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
        err = litellm.RateLimitError(
            message="rate limit exceeded",
            llm_provider="anthropic",
            model="claude-haiku",
            response=None,
        )
        assert _parse_retry_after(err) is None


class TestCompletionWithRetry:
    def test_succeeds_on_first_attempt(self):
        success = _make_success_response()
        with patch("make_agent.agent.litellm.completion", return_value=success) as mock_c:
            result = _completion_with_retry("model", [], {}, max_retries=3)
        assert result is success
        mock_c.assert_called_once()

    def test_retries_on_rate_limit_then_succeeds(self):
        err = _make_rate_limit_error(retry_after=10)
        success = _make_success_response()
        with patch("make_agent.agent.litellm.completion", side_effect=[err, err, success]):
            with patch("make_agent.agent.time.sleep") as mock_sleep:
                result = _completion_with_retry("model", [], {}, max_retries=3)
        assert result is success
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(10.0)

    def test_exponential_backoff_without_header(self):
        err = _make_rate_limit_error()  # no retry-after header
        success = _make_success_response()
        with patch("make_agent.agent.litellm.completion", side_effect=[err, err, success]):
            with patch("make_agent.agent.time.sleep") as mock_sleep:
                _completion_with_retry("model", [], {}, max_retries=3)
        # attempt 0 → 2^0 = 1s, attempt 1 → 2^1 = 2s
        assert mock_sleep.call_args_list == [call(1), call(2)]

    def test_exponential_backoff_capped_at_60s(self):
        err = _make_rate_limit_error()
        success = _make_success_response()
        # need 7 failures to reach 2^6=64 > 60; cap kicks in at attempt 6
        side_effects = [err] * 7 + [success]
        with patch("make_agent.agent.litellm.completion", side_effect=side_effects):
            with patch("make_agent.agent.time.sleep") as mock_sleep:
                _completion_with_retry("model", [], {}, max_retries=10)
        waits = [c.args[0] for c in mock_sleep.call_args_list]
        assert all(w <= 60 for w in waits)
        assert waits[6] == 60  # 2^6=64 capped to 60

    def test_raises_after_max_retries_exhausted(self):
        err = _make_rate_limit_error(retry_after=1)
        with patch("make_agent.agent.litellm.completion", side_effect=err):
            with patch("make_agent.agent.time.sleep"):
                with pytest.raises(litellm.RateLimitError):
                    _completion_with_retry("model", [], {}, max_retries=2)

    def test_total_calls_equals_max_retries_plus_one(self):
        err = _make_rate_limit_error(retry_after=1)
        with patch("make_agent.agent.litellm.completion", side_effect=err) as mock_c:
            with patch("make_agent.agent.time.sleep"):
                with pytest.raises(litellm.RateLimitError):
                    _completion_with_retry("model", [], {}, max_retries=3)
        assert mock_c.call_count == 4  # 1 initial + 3 retries

    def test_zero_max_retries_raises_immediately(self):
        err = _make_rate_limit_error(retry_after=1)
        with patch("make_agent.agent.litellm.completion", side_effect=err):
            with patch("make_agent.agent.time.sleep") as mock_sleep:
                with pytest.raises(litellm.RateLimitError):
                    _completion_with_retry("model", [], {}, max_retries=0)
        mock_sleep.assert_not_called()
