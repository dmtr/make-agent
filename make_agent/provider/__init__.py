"""LLM provider package.

Exports the provider protocol, stream chunk types, and concrete provider
implementations.  Use :func:`provider_for` to get a provider instance for a
given model name.
"""

from __future__ import annotations

from .anthropic import AnthropicProvider, _parse_retry_after
from .base import Provider, StreamChunk, TextDelta, ToolCallDelta, ToolCallStart, UsageDelta
from .openai import OpenAIProvider


def provider_for(model: str) -> Provider:
    """Return a provider instance appropriate for *model*.

    Routing rules (checked against the lower-cased model name):

    - ``openai/`` prefix or ``gpt`` / ``o1`` / ``o3`` / ``o4`` in name → :class:`OpenAIProvider`
    - ``anthropic/`` prefix or ``claude`` in name → :class:`AnthropicProvider`
    - otherwise → :exc:`NotImplementedError`
    """
    lower = model.lower()
    if lower.startswith("openai/") or any(kw in lower for kw in ("gpt", "o1", "o3", "o4")):
        return OpenAIProvider()
    if lower.startswith("anthropic/") or "claude" in lower:
        return AnthropicProvider()
    raise NotImplementedError(
        f"No provider for model {model!r}. "
        "Supported prefixes / names: openai/, gpt, o1, o3, o4, anthropic/, claude."
    )


def is_context_exceeded(exc: Exception) -> bool:
    """Return True when *exc* signals a context-window overflow."""
    if getattr(exc, "status_code", None) == 400:
        msg = str(exc).lower()
        return "context" in msg and any(w in msg for w in ("exceed", "window", "length", "limit", "size"))
    return False


def is_corrupt_message_history(exc: Exception) -> bool:
    """Return True when *exc* signals a corrupt or invalid message history."""
    if getattr(exc, "status_code", None) == 400:
        msg = str(exc).lower()
        return "failed to parse tool call" in msg
    return False


__all__ = [
    "AnthropicProvider",
    "OpenAIProvider",
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
