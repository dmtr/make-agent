"""Provider protocol, stream chunk types, and exception helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol, Union


@dataclass
class TextDelta:
    """A streamed text token from the LLM."""

    text: str


@dataclass
class ToolCallStart:
    """The beginning of a tool call — carries id and name."""

    index: int
    id: str
    name: str


@dataclass
class ToolCallDelta:
    """An incremental chunk of tool-call arguments (JSON string delta)."""

    index: int
    args_delta: str


@dataclass
class UsageDelta:
    """Token usage reported by the provider (may be split across events)."""

    input_tokens: int
    output_tokens: int


@dataclass
class ContextExceededChunk:
    """Provider signals context-window overflow via the stream instead of raising."""


StreamChunk = Union[TextDelta, ToolCallStart, ToolCallDelta, UsageDelta, ContextExceededChunk]


class Provider(Protocol):
    """Minimal streaming interface shared by all LLM provider backends."""

    async def astream(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        use_prompt_cache: bool = False,
        reasoning_effort: str = "",
    ) -> AsyncIterator[StreamChunk]: ...


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
