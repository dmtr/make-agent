"""Middleware types for :class:`~make_agent.agent_core.AgentManager`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Callable

from make_agent.agent_core.events import AgentEvent
from make_agent.protocols import MemoryProtocol


@dataclass
class Request:
    """Represents one agent turn entering the middleware chain."""

    session_id: str
    message: str


@dataclass
class Response:
    """Fully-built result after a turn's event stream is exhausted."""

    session_id: str
    content: str
    input_tokens: int
    output_tokens: int
    model: str


class MiddlewareBase:
    """Base class providing a pass-through ``__call__`` and no-op ``after_response``.

    Subclass this and override the methods you need; everything else is a no-op.
    """

    def __call__(
        self,
        request: Request,
        call_next: Callable[[Request], AsyncIterator[AgentEvent]],
    ) -> AsyncIterator[AgentEvent]:
        async def _passthrough():
            async for event in call_next(request):
                yield event

        return _passthrough()

    async def after_response(self, request: Request, response: Response) -> None:
        pass


class SessionMiddleware(MiddlewareBase):
    """Pass-through middleware that persists messages and token usage after each turn."""

    def __init__(self, memory: MemoryProtocol) -> None:
        self._memory = memory

    def __call__(
        self,
        request: Request,
        call_next: Callable[[Request], AsyncIterator[AgentEvent]],
    ) -> AsyncIterator[AgentEvent]:
        async def _passthrough():
            async for event in call_next(request):
                yield event

        return _passthrough()

    async def after_response(self, request: Request, response: Response) -> None:
        self._memory.store("user", request.message)
        self._memory.store("agent", response.content)
        if response.model:
            self._memory.record_token_usage(
                response.session_id,
                response.model,
                response.input_tokens,
                response.output_tokens,
            )
