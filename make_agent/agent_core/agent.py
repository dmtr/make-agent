"""Session management: AgentManager and related types."""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from pathlib import Path
from typing import AsyncIterator, Callable
from uuid import uuid4

from make_agent.protocols import ToolHandlerProtocol

from .bridge import (
    ApprovalRequested,
    CancelTurn,
    CompactNotice,
    DenySkill,
    ApproveSkill,
    ManagerError,
    ShellCommand,
    ShellEvent,
    Shutdown,
    StartTurn,
    StatusChanged,
    ToolFinished,
    ToolStarted,
    TokenEmitted,
    TurnCancelled,
    TurnFinished,
    TurnStarted,
)
from .events import AgentEvent, CompactEvent, ConfirmEvent, DoneEvent, TokenEvent, ToolDoneEvent, ToolStartEvent, UsageEvent
from .export import export_conversation
from .loop import AgentConfig, AgenticLoop, CompactCallback, MessageCallback, TokenCallback, ToolCallback, UsageCallback
from .middleware import MiddlewareBase, Request, Response, SessionMiddleware


# Backward-compatible alias.
Agent = AgenticLoop


_SKILL_INSPECTION_TOOLS = frozenset({"list_skills", "read_skill"})


def _prune_skill_messages(messages: list[dict]) -> list[dict]:
    """Remove outdated skill-inspection assistant turns.

    An assistant turn is "purgeable" when every tool call it contains is either
    ``list_skills`` or ``read_skill``.  The most recent purgeable turn that
    contains a ``list_skills`` call and the most recent purgeable turn that
    contains a ``read_skill`` call are both retained (they may be the same turn).
    All other purgeable turns are dropped together with their corresponding
    tool-result messages.  Everything else (system, user, assistant text, other
    tool calls) is kept.

    Returns the original list unchanged when there is nothing to drop.
    """
    purgeable: list[tuple[int, frozenset[str]]] = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            continue
        names = frozenset(tc.get("function", {}).get("name") for tc in tool_calls)
        if names <= _SKILL_INSPECTION_TOOLS:
            purgeable.append((i, names))

    last_list = next((i for i, names in reversed(purgeable) if "list_skills" in names), None)
    last_read = next((i for i, names in reversed(purgeable) if "read_skill" in names), None)
    to_keep = {x for x in (last_list, last_read) if x is not None}
    to_drop = {i for i, _ in purgeable} - to_keep

    if not to_drop:
        return list(messages)

    dropped_call_ids: set[str] = set()
    for i in to_drop:
        for tc in messages[i].get("tool_calls", []):
            if tc.get("id"):
                dropped_call_ids.add(tc["id"])

    result: list[dict] = []
    for i, msg in enumerate(messages):
        if i in to_drop:
            continue
        if msg.get("role") == "tool" and msg.get("tool_call_id") in dropped_call_ids:
            continue
        result.append(msg)
    return result


class SessionNotFoundError(Exception):
    pass


class AgentManager:
    def __init__(
        self,
        tool_handler: ToolHandlerProtocol,
        middlewares: list[MiddlewareBase] | None = None,
        compact_threshold: int = 0,
    ) -> None:
        self._tool_handler = tool_handler
        self._middlewares: list[MiddlewareBase] = middlewares if middlewares is not None else []
        self._sessions: dict[str, AgenticLoop] = {}
        self._compact_threshold = compact_threshold

    @staticmethod
    def get_session_id() -> str:
        return str(uuid4())

    def create_session(self, config: AgentConfig) -> str:
        session_id = self.get_session_id()
        loop = AgenticLoop(
            config._replace(session_id=session_id, compact_fn=_prune_skill_messages),
            self._tool_handler,
        )
        self._sessions[session_id] = loop
        return session_id

    def get_agent(self, session_id: str) -> AgenticLoop:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise SessionNotFoundError(f"Session with id {session_id} not found.")

    def _compact_session(self, session_id: str) -> int:
        """Prune skill-inspection messages from the session's conversation history.

        Returns the number of messages removed (0 when nothing was pruned).
        """
        loop = self.get_agent(session_id)
        pruned = _prune_skill_messages(loop.messages)
        removed = len(loop.messages) - len(pruned)
        if removed > 0:
            loop._messages = pruned
        return removed

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
            elif isinstance(cb, CompactCallback):
                yield CompactEvent(messages_removed=cb.messages_removed)
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

        When the API raises a context-window-exceeded error the loop's
        ``compact_fn`` prunes the history and retries automatically, emitting a
        :class:`CompactEvent`.  A proactive compact is also triggered after each
        turn when token usage exceeds ``compact_threshold``.
        """
        request = Request(session_id=session_id, message=message)

        content = ""
        input_tokens = 0
        output_tokens = 0
        model = ""

        async for event in self._build_chain()(request):
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

        if self._compact_threshold > 0 and input_tokens >= self._compact_threshold:
            messages_removed = self._compact_session(session_id)
            if messages_removed > 0:
                yield CompactEvent(messages_removed=messages_removed)

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

    # ── queue bridge ──────────────────────────────────────────────────────────

    async def run_shell_bridge(
        self,
        session_id: str,
        command_queue: asyncio.Queue[ShellCommand],
        event_queue: asyncio.Queue[ShellEvent],
    ) -> None:
        """Run the manager-owned bridge that mediates between the shell and a session.

        Reads :class:`ShellCommand` items from *command_queue* and writes
        :class:`ShellEvent` items to *event_queue*.  Runs until a
        :class:`Shutdown` command is received.

        Only one active turn per session is allowed.  A :class:`StartTurn`
        received while a turn is running is rejected with a
        :class:`ManagerError`.
        """
        active_turn_task: asyncio.Task[None] | None = None
        pending_approvals: dict[str, asyncio.Future[bool]] = {}

        while True:
            cmd = await command_queue.get()

            if isinstance(cmd, Shutdown):
                if active_turn_task and not active_turn_task.done():
                    active_turn_task.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await active_turn_task
                break

            elif isinstance(cmd, StartTurn):
                if active_turn_task and not active_turn_task.done():
                    await event_queue.put(
                        ManagerError(
                            message="Turn already active; send CancelTurn first",
                            turn_id=cmd.turn_id,
                        )
                    )
                    continue
                active_turn_task = asyncio.create_task(
                    self._execute_bridge_turn(
                        session_id,
                        cmd.message,
                        cmd.turn_id,
                        event_queue,
                        pending_approvals,
                    )
                )

            elif isinstance(cmd, CancelTurn):
                if active_turn_task and not active_turn_task.done():
                    active_turn_task.cancel()
                else:
                    await event_queue.put(ManagerError(message="No active turn to cancel"))

            elif isinstance(cmd, (ApproveSkill, DenySkill)):
                future = pending_approvals.pop(cmd.request_id, None)
                if future and not future.done():
                    future.set_result(isinstance(cmd, ApproveSkill))
                else:
                    await event_queue.put(
                        ManagerError(message=f"Unknown approval request: {cmd.request_id}")
                    )

    async def _execute_bridge_turn(
        self,
        session_id: str,
        message: str,
        turn_id: str,
        event_queue: asyncio.Queue[ShellEvent],
        pending_approvals: dict[str, asyncio.Future[bool]],
    ) -> None:
        """Stream one agent turn, translating AgentEvents into ShellEvents."""
        await event_queue.put(StatusChanged(is_busy=True))
        await event_queue.put(TurnStarted(turn_id=turn_id))
        current_tool_id: str | None = None
        content = ""
        try:
            async for event in self.astream_events(session_id, message):
                if isinstance(event, TokenEvent):
                    await event_queue.put(TokenEmitted(turn_id=turn_id, text=event.text))
                elif isinstance(event, ToolStartEvent):
                    current_tool_id = str(uuid4())
                    await event_queue.put(
                        ToolStarted(
                            turn_id=turn_id,
                            tool_id=current_tool_id,
                            name=event.name,
                            args=event.args,
                            description=event.description,
                        )
                    )
                elif isinstance(event, ToolDoneEvent):
                    await event_queue.put(
                        ToolFinished(
                            turn_id=turn_id,
                            tool_id=current_tool_id or "",
                            name=event.name,
                            output=event.output,
                            is_error=event.is_error,
                            duration_ms=event.duration_ms,
                        )
                    )
                    current_tool_id = None
                elif isinstance(event, DoneEvent):
                    content = event.content
                elif isinstance(event, ConfirmEvent):
                    request_id = str(uuid4())
                    loop = asyncio.get_running_loop()
                    approval_future: asyncio.Future[bool] = loop.create_future()
                    pending_approvals[request_id] = approval_future
                    await event_queue.put(
                        ApprovalRequested(
                            request_id=request_id,
                            turn_id=turn_id,
                            skill_name=event.skill_name,
                            target=event.target,
                            kwargs=event.kwargs,
                        )
                    )
                    allowed = await approval_future
                    if allowed:
                        event.allow()
                    else:
                        event.deny()
                elif isinstance(event, CompactEvent):
                    await event_queue.put(CompactNotice(messages_removed=event.messages_removed))
        except asyncio.CancelledError:
            for fut in list(pending_approvals.values()):
                if not fut.done():
                    fut.cancel()
            pending_approvals.clear()
            await event_queue.put(TurnCancelled(turn_id=turn_id))
            await event_queue.put(StatusChanged(is_busy=False))
            raise
        except Exception as exc:
            await event_queue.put(ManagerError(turn_id=turn_id, message=str(exc)))
            await event_queue.put(StatusChanged(is_busy=False))
        else:
            await event_queue.put(TurnFinished(turn_id=turn_id, content=content))
            await event_queue.put(StatusChanged(is_busy=False))
