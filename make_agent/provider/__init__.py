"""LLM provider package.

Exports the provider protocol, stream chunk types, and the concrete
Anthropic implementation.  Use :func:`provider_for` to get a provider
instance for a given model name.
"""

from __future__ import annotations

import anthropic as anthropic_sdk

from .anthropic import AnthropicProvider, _parse_retry_after, provider_for
from .base import Provider, StreamChunk, TextDelta, ToolCallDelta, ToolCallStart, UsageDelta


def is_context_exceeded(exc: Exception) -> bool:
    """Return True when *exc* signals a context-window overflow."""
    is_bad_request = isinstance(exc, anthropic_sdk.BadRequestError) or getattr(exc, "status_code", None) == 400
    if is_bad_request:
        msg = str(exc).lower()
        return "context" in msg and any(w in msg for w in ("exceed", "window", "length", "limit", "size"))
    return False


def is_corrupt_message_history(exc: Exception) -> bool:
    """Return True when *exc* signals a corrupt or invalid message history."""
    is_bad_request = isinstance(exc, anthropic_sdk.BadRequestError) or getattr(exc, "status_code", None) == 400
    if is_bad_request:
        msg = str(exc).lower()
        return "failed to parse tool call" in msg
    return False


__all__ = [
    "AnthropicProvider",
    "Provider",
    "StreamChunk",
    "TextDelta",
    "ToolCallDelta",
    "ToolCallStart",
    "UsageDelta",
    "is_context_exceeded",
    "is_corrupt_message_history",
    "provider_for",
    "_parse_retry_after",
]
