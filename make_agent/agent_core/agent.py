"""Session management: AgentManager and related types."""

from __future__ import annotations

import time
from pathlib import Path
from typing import AsyncGenerator
from uuid import uuid4

from make_agent.protocols import MemoryProtocol, ToolHandlerProtocol

from .events import AgentEvent, ConfirmEvent, DoneEvent, TokenEvent, ToolDoneEvent, ToolStartEvent, UsageEvent
from .export import export_conversation
from .loop import AgentConfig, AgenticLoop, MessageCallback, TokenCallback, ToolCallback, UsageCallback


# Backward-compatible alias.
Agent = AgenticLoop


class SessionNotFoundError(Exception):
    pass


class AgentManager:
    def __init__(self, memory: MemoryProtocol, tool_handler: ToolHandlerProtocol) -> None:
        self._memory = memory
        self._tool_handler = tool_handler
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
        async for event in self.astream_events(session_id, message):
            if isinstance(event, DoneEvent):
                return event.content
        return ""

    async def astream_events(
        self, session_id: str, message: str
    ) -> AsyncGenerator[AgentEvent, None]:
        """Stream :class:`AgentEvent` objects for one agent turn.

        Tool execution and skill confirmation are handled internally.
        Yields :class:`ConfirmEvent` for untrusted skills — the consumer must
        call :meth:`~ConfirmEvent.allow` or :meth:`~ConfirmEvent.deny` to
        unblock the generator.
        """
        loop = self.get_agent(session_id)
        self._memory.store("user", message)
        async for cb in loop.astream(message):
            if isinstance(cb, TokenCallback):
                yield TokenEvent(text=cb.message)
            elif isinstance(cb, MessageCallback):
                self._memory.store("agent", cb.message)
                yield DoneEvent(content=cb.message)
            elif isinstance(cb, UsageCallback):
                self._memory.record_token_usage(
                    session_id, cb.model, cb.input_tokens, cb.output_tokens
                )
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

    def export_conversation(self, session_id: str) -> Path | None:
        loop = self.get_agent(session_id)
        if loop.messages:
            return export_conversation(loop.messages, loop.model)
        return None

    def get_token_stats(self, session_id: str) -> dict:
        """Return aggregated token usage for *session_id*, or an empty dict when unavailable."""
        return self._memory.get_session_stats(session_id)
