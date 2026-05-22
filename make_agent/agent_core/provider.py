"""LiteLLM provider: configuration, retry logic, and model helpers."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import litellm

from .constants import DEFAULT_MAX_TOKENS, DEFAULT_REASONING_EFFORT

litellm.suppress_debug_info = True
litellm.verbose = False
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Proxy").setLevel(logging.WARNING)
litellm.drop_params = True


def _is_anthropic_model(model: str) -> bool:
    """Return True if *model* targets Anthropic (supports cache_control blocks)."""
    lower = model.lower()
    return lower.startswith("anthropic/") or "claude" in lower


def _parse_retry_after(e: litellm.RateLimitError) -> float | None:
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


async def _acompletion_with_retry(
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
            wait = _parse_retry_after(e) or min(2**attempt, 60)
            print(
                f"Rate limited, retrying in {wait:.0f}s" f" (attempt {attempt + 1}/{max_retries})...",
                flush=True,
            )
            await asyncio.sleep(wait)
