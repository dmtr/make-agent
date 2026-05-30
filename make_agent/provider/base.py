"""Provider protocol and stream chunk types."""

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


StreamChunk = Union[TextDelta, ToolCallStart, ToolCallDelta, UsageDelta]


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
