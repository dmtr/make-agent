"""Agent event types emitted by :class:`~make_agent.agent_core.AgentManager`."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


@dataclass
class TokenEvent:
    """A partial token streamed from the LLM."""

    text: str


@dataclass
class ToolStartEvent:
    """The LLM requested a tool call; execution is about to begin."""

    name: str
    args: dict
    description: str = ""


@dataclass
class ToolDoneEvent:
    """A tool call completed."""

    name: str
    output: str
    is_error: bool
    duration_ms: float | None = None


@dataclass
class DoneEvent:
    """The LLM produced a final text response (no tool calls)."""

    content: str


@dataclass
class ConfirmEvent:
    """An untrusted skill requires user confirmation before execution.

    The manager awaits :meth:`wait` after yielding this event.  The shell
    (or any consumer) must call :meth:`allow` or :meth:`deny` to unblock it.
    """

    skill_name: str
    target: str
    kwargs: dict
    _future: asyncio.Future[bool] = field(
        default_factory=lambda: asyncio.get_event_loop().create_future(), repr=False
    )

    def allow(self) -> None:
        if not self._future.done():
            self._future.set_result(True)

    def deny(self) -> None:
        if not self._future.done():
            self._future.set_result(False)

    async def wait(self) -> bool:
        return await self._future


@dataclass
class UsageEvent:
    """Token usage reported after each LLM model turn."""

    model: str
    input_tokens: int
    output_tokens: int


@dataclass
class CompactEvent:
    """History was compacted (oldest turns dropped) after a context-window error."""

    attempt: int
    messages_dropped: int


AgentEvent = (
    TokenEvent
    | ToolStartEvent
    | ToolDoneEvent
    | DoneEvent
    | ConfirmEvent
    | UsageEvent
    | CompactEvent
)
