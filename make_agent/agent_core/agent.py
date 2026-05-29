"""Session management: AgentManager and related types."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from pathlib import Path
from typing import AsyncIterator, Callable
from uuid import uuid4

import litellm

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
from .constants import KEEP_COMPACT_TURNS, KEEP_RECENT_TURNS
from .events import AgentEvent, CompactEvent, ConfirmEvent, DoneEvent, TokenEvent, ToolDoneEvent, ToolStartEvent, UsageEvent
from .export import export_conversation
from .loop import AgentConfig, AgenticLoop, CompactCallback, MessageCallback, TokenCallback, ToolCallback, UsageCallback
from .middleware import MiddlewareBase, Request, Response, SessionMiddleware
from make_agent.provider import estimate_tokens

logger = logging.getLogger(__name__)

COMPACT_SUMMARY_MARKER = "Summary of earlier conversation:\n"

COMPACT_SUMMARY_PROMPT = (
    "You are compacting a conversation history that has grown too long for the context window.\n\n"
    "Summarize the following conversation in 3–5 concise paragraphs. Focus on:\n"
    "- Decisions made and conclusions reached\n"
    "- Files examined, modified, or created\n"
    "- Errors encountered and how they were resolved\n"
    "- Important facts discovered\n"
    "- The current state of any ongoing tasks\n\n"
    "If a prior summary is provided at the start, merge it with the new material into one unified summary.\n"
    "Be factual and specific. Omit greetings and filler text."
)

COMPACT_SUMMARY_MAX_TOKENS = 1024  # fallback default; actual value computed from context window


# Backward-compatible alias.
Agent = AgenticLoop


def _split_into_turns(messages: list[dict]) -> list[list[dict]]:
    """Split a list of non-system messages into turns.

    A turn starts with a ``user`` message and includes all following
    non-user messages up to (but not including) the next ``user`` message.
    """
    turns: list[list[dict]] = []
    for msg in messages:
        if msg.get("role") == "user":
            turns.append([msg])
        elif turns:
            turns[-1].append(msg)
    return turns


def compact_messages(messages: list[dict], keep_turns: int = KEEP_COMPACT_TURNS) -> list[dict]:
    """Compact conversation history to the last *keep_turns* turns.

    A "turn" is one user message and all subsequent messages until the next
    user message.  When the conversation already has *keep_turns* or fewer
    turns, returns the original list unchanged (``removed == 0``).

    Otherwise, drops older turns and returns the system prompt (if any)
    followed by the last *keep_turns* turns.  The agent can use memory search
    tools to recall earlier context.
    """
    system = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    turns = _split_into_turns(non_system)

    if len(turns) <= keep_turns:
        return list(messages)

    kept = [msg for turn in turns[-keep_turns:] for msg in turn]
    return system + kept


def _format_messages_for_summary(messages: list[dict]) -> str:
    """Render a list of messages as a readable text block for summarization."""
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        if role == "user":
            content = msg.get("content") or ""
            parts.append(f"User: {content}")
        elif role == "assistant":
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                names = ", ".join(tc.get("function", {}).get("name", "?") for tc in tool_calls)
                line = f"Assistant: [Called tools: {names}]"
                if content:
                    line = f"Assistant: {content}\n[Called tools: {names}]"
                parts.append(line)
            elif content:
                parts.append(f"Assistant: {content}")
        elif role == "tool":
            content = msg.get("content") or ""
            if len(content) > 500:
                content = content[:500] + "...[truncated]"
            parts.append(f"Tool result: {content}")
    return "\n\n".join(parts)


async def _summarize_messages(
    messages: list[dict], model: str, max_tokens: int = COMPACT_SUMMARY_MAX_TOKENS
) -> str:
    """Call the LLM to summarise *messages*. Returns an empty string on failure."""
    text = _format_messages_for_summary(messages)
    if not text.strip():
        return ""
    try:
        response = await litellm.acompletion(
            model=model,
            messages=[
                {"role": "system", "content": COMPACT_SUMMARY_PROMPT},
                {"role": "user", "content": text},
            ],
            max_tokens=max_tokens,
            stream=False,
        )
        return response.choices[0].message.content or ""
    except Exception:
        logger.warning("Summarization LLM call failed; falling back to turn-based compact")
        return ""


async def compact_with_summary(
    messages: list[dict],
    model: str,
    keep_recent_turns: int = KEEP_RECENT_TURNS,
    summary_max_tokens: int = COMPACT_SUMMARY_MAX_TOKENS,
) -> list[dict]:
    """Compact history to [system messages] + [one summary] + [recent turns].

    Summarises turns older than the last *keep_recent_turns* turns using a
    single LLM call and replaces them with a compact summary injected as a
    system message.  Any previously inserted summary is merged into the new
    one rather than stacked, keeping exactly one summary message at all times.

    Token-based gating: if the summarised result does not reduce the estimated
    token count, falls back to :func:`compact_messages` (turn-drop), and if
    that also fails to reduce tokens, returns the original list unchanged.
    """
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    if len(non_system) < 2:
        return list(messages)

    # Detect an existing summary message to merge rather than stack.
    summary_idx = next(
        (
            i
            for i, m in enumerate(system_msgs)
            if isinstance(m.get("content"), str) and m["content"].startswith(COMPACT_SUMMARY_MARKER)
        ),
        None,
    )
    existing_summary = ""
    base_system_msgs = system_msgs
    if summary_idx is not None:
        existing_summary = system_msgs[summary_idx]["content"][len(COMPACT_SUMMARY_MARKER):]
        base_system_msgs = [m for i, m in enumerate(system_msgs) if i != summary_idx]

    # Split into turns; keep the most recent ones verbatim.
    turns = _split_into_turns(non_system)
    if len(turns) <= keep_recent_turns:
        return list(messages)

    to_summarize = [msg for turn in turns[:-keep_recent_turns] for msg in turn]
    to_keep = [msg for turn in turns[-keep_recent_turns:] for msg in turn]

    # Prepend any prior summary so the LLM merges it into one unified summary.
    if existing_summary:
        summarize_input = [{"role": "system", "content": f"Prior summary:\n{existing_summary}"}] + to_summarize
    else:
        summarize_input = to_summarize

    summary = await _summarize_messages(summarize_input, model, max_tokens=summary_max_tokens)
    if not summary:
        return compact_messages(messages)

    result = (
        base_system_msgs
        + [{"role": "system", "content": f"{COMPACT_SUMMARY_MARKER}{summary}"}]
        + to_keep
    )

    # Token-based validation: only commit if the new history is actually smaller.
    before_tokens = estimate_tokens(messages, model)
    after_tokens = estimate_tokens(result, model)
    if after_tokens >= before_tokens:
        fallback = compact_messages(messages)
        if estimate_tokens(fallback, model) < before_tokens:
            return fallback
        return list(messages)

    return result


class SessionNotFoundError(Exception):
    pass


class AgentManager:
    def __init__(
        self,
        tool_handler: ToolHandlerProtocol,
        middlewares: list[MiddlewareBase] | None = None,
        compact_threshold: int = 0,
        compact_target: int = 0,
        summary_max_tokens: int = COMPACT_SUMMARY_MAX_TOKENS,
    ) -> None:
        self._tool_handler = tool_handler
        self._middlewares: list[MiddlewareBase] = middlewares if middlewares is not None else []
        self._sessions: dict[str, AgenticLoop] = {}
        self._compact_threshold = compact_threshold
        self._compact_target = compact_target
        self._summary_max_tokens = summary_max_tokens
        self._skip_proactive: dict[str, bool] = {}

    @staticmethod
    def get_session_id() -> str:
        return str(uuid4())

    def create_session(self, config: AgentConfig) -> str:
        session_id = self.get_session_id()
        model = config.model
        summary_max_tokens = self._summary_max_tokens

        async def _compact_fn(messages: list[dict]) -> list[dict]:
            return await compact_with_summary(messages, model, summary_max_tokens=summary_max_tokens)

        loop = AgenticLoop(
            config._replace(session_id=session_id, compact_fn=_compact_fn),
            self._tool_handler,
        )
        self._sessions[session_id] = loop
        return session_id

    def get_agent(self, session_id: str) -> AgenticLoop:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise SessionNotFoundError(f"Session with id {session_id} not found.")

    async def _compact_session(self, session_id: str) -> int:
        """Compact the session's conversation history to a summary.

        Returns the number of messages removed (0 when nothing was pruned or
        when compaction would not reduce the estimated token count).
        """
        loop = self.get_agent(session_id)
        pruned = await compact_with_summary(
            loop.messages, loop.model, summary_max_tokens=self._summary_max_tokens
        )
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
        peak_input_tokens = 0
        model = ""

        async for event in self._build_chain()(request):
            if isinstance(event, DoneEvent):
                content = event.content
            elif isinstance(event, UsageEvent):
                input_tokens += event.input_tokens
                output_tokens += event.output_tokens
                peak_input_tokens = max(peak_input_tokens, event.input_tokens)
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

        # Hysteresis: clear the skip flag once tokens have dropped below the target.
        if self._compact_target > 0 and self._skip_proactive.get(session_id, False):
            if peak_input_tokens < self._compact_target:
                self._skip_proactive[session_id] = False

        if (
            self._compact_threshold > 0
            and peak_input_tokens >= self._compact_threshold
            and not self._skip_proactive.get(session_id, False)
        ):
            messages_removed = await self._compact_session(session_id)
            if messages_removed > 0:
                yield CompactEvent(messages_removed=messages_removed)
                if self._compact_target > 0:
                    self._skip_proactive[session_id] = True

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
