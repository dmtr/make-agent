"""Session management: AgentManager and related types."""

from __future__ import annotations

import time
from pathlib import Path
from typing import AsyncIterator, Callable
from uuid import uuid4

from make_agent.protocols import ToolHandlerProtocol

from .events import AgentEvent, ConfirmEvent, DoneEvent, TokenEvent, ToolDoneEvent, ToolStartEvent, UsageEvent
from .export import export_conversation
from .loop import AgentConfig, AgenticLoop, MessageCallback, TokenCallback, ToolCallback, UsageCallback
from .middleware import MiddlewareBase, Request, Response, SessionMiddleware


# Backward-compatible alias.
Agent = AgenticLoop


class SessionNotFoundError(Exception):
    pass


class AgentManager:
    def __init__(
        self,
        tool_handler: ToolHandlerProtocol,
        middlewares: list[MiddlewareBase] | None = None,
    ) -> None:
        self._tool_handler = tool_handler
        self._middlewares: list[MiddlewareBase] = middlewares if middlewares is not None else []
        self._sessions: dict[str, AgenticLoop] = {}

    @staticmethod
    def get_session_id() -> str:
        return str(uuid4())

    def create_session(self, config: AgentConfig) -> str:
        session_id = self.get_session_id()
        loop = AgenticLoop(config._replace(session_id=session_id), self._tool_handler)
        self._sessions[session_id] = loop
        return session_id

    def get_agent(self, session_id: str) -> AgenticLoop:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise SessionNotFoundError(f"Session with id {session_id} not found.")

    async def arun_agent(self, session_id: str, message: str) -> str:
        """Run one agent turn and return the final reply text."""
        result = ""
        async for event in self.astream_events(session_id, message):
            if isinstance(event, DoneEvent):
                result = event.content
        return result

    async def _stream_events_core(self, request: Request) -> AsyncIterator[AgentEvent]:
        """Core event-streaming logic with no middleware side-effects."""
        loop = self.get_agent(request.session_id)
        async for cb in loop.astream(request.message):
            if isinstance(cb, TokenCallback):
                yield TokenEvent(text=cb.message)
            elif isinstance(cb, MessageCallback):
                yield DoneEvent(content=cb.message)
            elif isinstance(cb, UsageCallback):
                yield UsageEvent(
                    model=cb.model,
                    input_tokens=cb.input_tokens,
                    output_tokens=cb.output_tokens,
                )
            elif isinstance(cb, ToolCallback):
                if cb.tool_name == "execute_skill":
                    skill_name = cb.tool_args.get("name", "")
                    target = cb.tool_args.get("target") or cb.tool_args.get("command", "")
                    if not self._tool_handler.is_skill_trusted(skill_name, target):
                        kwargs = cb.tool_args.get("kwargs") or {}
                        confirm = ConfirmEvent(skill_name=skill_name, target=target, kwargs=kwargs)
                        yield confirm
                        allowed = await confirm.wait()
                        if not allowed:
                            denial = f"User denied execution of '{skill_name}/{target}'"
                            cb.set_response(denial)
                            continue
                yield ToolStartEvent(
                    name=cb.tool_name,
                    args=cb.tool_args,
                    description=cb.description,
                )
                start_time = time.monotonic()
                result = await self._tool_handler.execute(
                    cb.tool_name, cb.tool_args, loop._max_tool_output
                )
                cb.set_response(result.output, is_error=result.is_error)
                duration_ms = (time.monotonic() - start_time) * 1000
                cb.duration_ms = duration_ms
                yield ToolDoneEvent(
                    name=cb.tool_name,
                    output=result.output,
                    is_error=result.is_error,
                    duration_ms=duration_ms,
                )

    def _build_chain(self) -> Callable[[Request], AsyncIterator[AgentEvent]]:
        """Build the middleware chain; first middleware in the list is innermost."""
        current: Callable[[Request], AsyncIterator[AgentEvent]] = self._stream_events_core
        for mw in self._middlewares:
            prev = current

            def make_wrapper(
                _mw: MiddlewareBase, _prev: Callable[[Request], AsyncIterator[AgentEvent]]
            ) -> Callable[[Request], AsyncIterator[AgentEvent]]:
                return lambda req: _mw(req, _prev)

            current = make_wrapper(mw, prev)
        return current

    async def astream_events(
        self, session_id: str, message: str
    ) -> AsyncIterator[AgentEvent]:
        """Stream :class:`AgentEvent` objects for one agent turn.

        Tool execution and skill confirmation are handled internally.
        Yields :class:`ConfirmEvent` for untrusted skills — the consumer must
        call :meth:`~ConfirmEvent.allow` or :meth:`~ConfirmEvent.deny` to
        unblock the generator.
        After the stream is exhausted, ``after_response`` is called on each
        middleware in order (innermost first).
        """
        request = Request(session_id=session_id, message=message)
        chain = self._build_chain()

        content = ""
        input_tokens = 0
        output_tokens = 0
        model = ""

        async for event in chain(request):
            if isinstance(event, DoneEvent):
                content = event.content
            elif isinstance(event, UsageEvent):
                input_tokens += event.input_tokens
                output_tokens += event.output_tokens
                model = event.model
            yield event

        response = Response(
            session_id=session_id,
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
        )
        for mw in self._middlewares:
            await mw.after_response(request, response)

    def export_conversation(self, session_id: str) -> Path | None:
        loop = self.get_agent(session_id)
        if loop.messages:
            return export_conversation(loop.messages, loop.model)
        return None

    def get_token_stats(self, session_id: str) -> dict:
        """Return aggregated token usage for *session_id*, or an empty dict when unavailable."""
        for mw in self._middlewares:
            if isinstance(mw, SessionMiddleware):
                return mw._memory.get_session_stats(session_id)
        return {}
