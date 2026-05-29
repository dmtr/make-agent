"""LiteLLM provider: configuration, retry logic, and model helpers."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import litellm

from make_agent.agent_core.constants import DEFAULT_MAX_TOKENS, DEFAULT_REASONING_EFFORT

DEFAULT_CONTEXT_WINDOW = 64_000
DEFAULT_COMPACT_THRESHOLD_RATIO = 0.75
DEFAULT_COMPACT_TARGET_RATIO = 0.5

litellm.suppress_debug_info = True
litellm.verbose = False
# Ensure all LiteLLM loggers propagate to our root file handler.
for _name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy"):
    _llm_logger = logging.getLogger(_name)
    _llm_logger.setLevel(logging.WARNING)
    _llm_logger.propagate = True
    # Remove any handlers LiteLLM attached to itself
    for _h in _llm_logger.handlers[:]:
        _llm_logger.removeHandler(_h)
litellm.drop_params = True

logger = logging.getLogger(__name__)


def is_anthropic_model(model: str) -> bool:
    """Return True if *model* targets Anthropic (supports cache_control blocks)."""
    lower = model.lower()
    return lower.startswith("anthropic/") or "claude" in lower


def is_context_exceeded(exc: Exception) -> bool:
    """Return True when *exc* signals a context-window overflow."""
    if isinstance(exc, litellm.ContextWindowExceededError):
        return True
    is_bad_request = isinstance(exc, litellm.BadRequestError) or getattr(exc, "status_code", None) == 400
    if is_bad_request:
        msg = str(exc).lower()
        return "context" in msg and any(w in msg for w in ("exceed", "window", "length", "limit", "size"))
    return False


def is_corrupt_message_history(exc: Exception) -> bool:
    """Return True when *exc* indicates the message history contains invalid content.

    Anthropic raises a 400 BadRequestError when an assistant message in the
    conversation history contains a ``tool_use`` block whose ``input`` field is
    not valid JSON (e.g. truncated arguments from a prior streaming turn).
    Compacting the history removes the offending message, making a retry safe.
    """
    is_bad_request = isinstance(exc, litellm.BadRequestError) or getattr(exc, "status_code", None) == 400
    if not is_bad_request:
        return False
    msg = str(exc).lower()
    return "failed to parse tool call" in msg or (
        "tool" in msg and "unterminated string" in msg
    )


def parse_retry_after(e: litellm.RateLimitError) -> float | None:
    """Return the wait time in seconds from a RateLimitError's response headers.

    Checks ``retry-after-ms`` (milliseconds) then ``retry-after`` (seconds).
    Returns ``None`` when neither header is present.
    """
    try:
        headers = e.response.headers if hasattr(e, "response") and e.response is not None else {}
    except Exception:
        return None
    if ms := headers.get("retry-after-ms"):
        return float(ms) / 1000
    if sec := headers.get("retry-after"):
        return float(sec)
    return None


async def acompletion_with_retry(
    model: str,
    messages: list[dict],
    tool_kwargs: dict[str, Any],
    max_retries: int,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> Any:
    """Call ``litellm.acompletion`` with streaming, retrying on rate limit.

    On each ``RateLimitError`` the wait time is read from the ``Retry-After``
    response header when present, otherwise exponential backoff is used
    (``2^attempt`` seconds, capped at 60 s).  A message is printed before
    each retry so the user can see what is happening.

    Returns an ``AsyncIterator[ChatCompletionChunk]``.
    """
    for attempt in range(max_retries + 1):
        try:
            return await litellm.acompletion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                stream=True,
                stream_options={"include_usage": True},
                **tool_kwargs,
            )
        except litellm.RateLimitError as e:
            if attempt == max_retries:
                raise
            wait = parse_retry_after(e) or min(2**attempt, 60)
            print(
                f"Rate limited, retrying in {wait:.0f}s" f" (attempt {attempt + 1}/{max_retries})...",
                flush=True,
            )
            await asyncio.sleep(wait)


def _get_context_window(model: str, context_window: int = 0) -> int:
    """Return the context window size for *model* in tokens.

    Uses *context_window* when non-zero; otherwise queries ``litellm.get_model_info``.
    Falls back to ``DEFAULT_CONTEXT_WINDOW`` when the model is unknown to litellm.
    """
    if context_window:
        return context_window
    try:
        info = litellm.get_model_info(model)
        window = info.get("max_input_tokens") or 0
    except Exception:
        logger.debug("Could not get model info for %r; using default context window", model)
        window = 0
    return window or DEFAULT_CONTEXT_WINDOW


def compute_compact_threshold(
    model: str,
    threshold_ratio: float = DEFAULT_COMPACT_THRESHOLD_RATIO,
    context_window: int = 0,
) -> int:
    """Return the auto-compact threshold in tokens.

    Uses *context_window* when non-zero; otherwise queries ``litellm.get_model_info``.
    Falls back to ``DEFAULT_CONTEXT_WINDOW`` when the model is unknown to litellm.
    """
    return int(_get_context_window(model, context_window) * threshold_ratio)


def compute_compact_target(
    model: str,
    target_ratio: float = DEFAULT_COMPACT_TARGET_RATIO,
    context_window: int = 0,
) -> int:
    """Return the post-compaction target token count (hysteresis lower bound).

    Proactive compaction is suppressed after a successful compact until the
    estimated context size climbs back above the threshold.  The target marks
    the level below which the context must fall after compaction.
    """
    return int(_get_context_window(model, context_window) * target_ratio)


def compute_summary_max_tokens(model: str, context_window: int = 0) -> int:
    """Return a dynamic token budget for the compaction summary.

    Clamps ``5 %`` of the context window between 256 and 2 048 tokens.
    """
    window = _get_context_window(model, context_window)
    return max(256, min(2048, int(window * 0.05)))


_MESSAGE_OVERHEAD = 4  # approximate special tokens per message (role, separators)


def estimate_tokens(messages: list[dict], model: str) -> int:
    """Estimate the token count for *messages* against *model*.

    Uses ``litellm.token_counter`` when available; falls back to a cheap
    character-based approximation on any failure.
    """
    try:
        return litellm.token_counter(model=model, messages=messages)
    except Exception:
        logger.warning(
            "litellm.token_counter failed for model %r; using character approximation",
            model,
        )
        total = 0
        for m in messages:
            # Content field
            total += len(str(m.get("content") or ""))
            # Tool calls: arguments can be large
            for tc in m.get("tool_calls") or []:
                total += len(str(tc.get("function", {}).get("arguments", "")))
                total += len(str(tc.get("function", {}).get("name", "")))
            # Name field
            total += len(str(m.get("name") or ""))
            # Per-message formatting overhead (role labels, special tokens)
            total += _MESSAGE_OVERHEAD
        return total // 4


__all__ = [
    "DEFAULT_COMPACT_THRESHOLD_RATIO",
    "DEFAULT_COMPACT_TARGET_RATIO",
    "DEFAULT_CONTEXT_WINDOW",
    "acompletion_with_retry",
    "compute_compact_target",
    "compute_compact_threshold",
    "compute_summary_max_tokens",
    "estimate_tokens",
    "is_anthropic_model",
    "is_context_exceeded",
    "is_corrupt_message_history",
    "parse_retry_after",
]
