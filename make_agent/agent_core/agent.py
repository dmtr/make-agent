"""LLM agent: conversation loop, tool dispatch, and session management."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, NamedTuple
from uuid import uuid4

import any_llm
from any_llm.types.completion import (
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageToolCall,
    Function,
)

from make_agent.protocols import MemoryProtocol, ToolHandlerProtocol

from .export import export_conversation

_DEFAULT_MAX_RETRIES = 5
_DEFAULT_TOOL_TIMEOUT = 600  # seconds
_DEFAULT_MAX_TOOL_OUTPUT = 16000  # characters; 0 = unlimited
_DEFAULT_MAX_TOKENS = 4096
_DEFAULT_REASONING_EFFORT = "auto"
_DEFAULT_COMPACT_THRESHOLD = 80_000  # prompt tokens before auto-compact fires
_MAX_REPEATED_FAILURES = 8
_MAX_MODEL_TURNS_PER_REQUEST = 64
_MAX_TOOL_CALLS_PER_REQUEST = 256
_MAX_RUN_SECONDS_PER_REQUEST = 900

_COMPACT_SKILL_TOOLS = frozenset({"list_skills", "read_skill"})
_COMPACT_SUMMARY_SYSTEM = (
    "Summarize this conversation concisely. Include: the user's goals, "
    "tasks completed, key decisions made, and current state. "
    "Be brief — the agent has persistent memory to look up details."
)

logger = logging.getLogger(__name__)


@dataclass
class TokenEvent:
    """A partial text token streamed from the LLM."""

    text: str


@dataclass
class ToolStartEvent:
    """Emitted just before a tool call is executed."""

    name: str
    args: dict


@dataclass
class ToolDoneEvent:
    """Emitted after a tool call completes."""

    name: str
    output: str
    is_error: bool


@dataclass
class DoneEvent:
    """Emitted once the agent has a final text response (no more tool calls)."""

    content: str


AgentEvent = TokenEvent | ToolStartEvent | ToolDoneEvent | DoneEvent


class AgentConfig(NamedTuple):
    system_prompt: str
    model: str
    max_retries: int = _DEFAULT_MAX_RETRIES
    tool_timeout: int = _DEFAULT_TOOL_TIMEOUT
    max_tool_output: int = _DEFAULT_MAX_TOOL_OUTPUT
    max_tokens: int = _DEFAULT_MAX_TOKENS
    skills_dir: str = ""
    disabled_builtin_tools: frozenset[str] = frozenset()
    reasoning_effort: str = _DEFAULT_REASONING_EFFORT
    session_id: str | None = None
    project_dir: Path = Path()
    compact_threshold: int = _DEFAULT_COMPACT_THRESHOLD


def _parse_retry_after(e: any_llm.RateLimitError) -> float | None:
    """Return the wait time in seconds from a RateLimitError's response headers.

    Checks ``retry-after-ms`` (milliseconds) then ``retry-after`` (seconds).
    Returns ``None`` when neither header is present.
    """
    try:
        orig = e.original_exception
        headers = (
            orig.response.headers
            if orig is not None
            and hasattr(orig, "response")
            and orig.response is not None
            else {}
        )
    except Exception:
        return None
    if ms := headers.get("retry-after-ms"):
        return float(ms) / 1000
    if sec := headers.get("retry-after"):
        return float(sec)
    return None


async def _acompletion_with_retry(
    model: str,
    messages: list[dict],
    tool_kwargs: dict[str, Any],
    max_retries: int,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    reasoning_effort: str = _DEFAULT_REASONING_EFFORT,
) -> Any:
    """Call ``any_llm.acompletion`` with streaming, retrying on rate limit.

    On each ``RateLimitError`` the wait time is read from the ``Retry-After``
    response header when present, otherwise exponential backoff is used
    (``2^attempt`` seconds, capped at 60 s).  A message is printed before
    each retry so the user can see what is happening.

    Returns an ``AsyncIterator[ChatCompletionChunk]``.
    """
    for attempt in range(max_retries + 1):
        try:
            return await any_llm.acompletion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                stream=True,
                stream_options={"include_usage": True},
                **tool_kwargs,
            )
        except any_llm.RateLimitError as e:
            if attempt == max_retries:
                raise
            wait = _parse_retry_after(e) or min(2**attempt, 60)
            print(
                f"Rate limited, retrying in {wait:.0f}s"
                f" (attempt {attempt + 1}/{max_retries})...",
                flush=True,
            )
            await asyncio.sleep(wait)


def _parse_item(doc: Any) -> ChatCompletionMessageToolCall | None:
    result: list[ChatCompletionMessageToolCall] = []
    for item in doc:
        if not isinstance(item, dict) or item.get("type") != "function":
            return None
        func = item.get("function", {})
        if "name" not in func:
            return None
        args = func.get("arguments", {})
        args_str = json.dumps(args) if isinstance(args, dict) else args
        result.append(
            ChatCompletionMessageFunctionToolCall(
                id=item.get("id", ""),
                type="function",
                function=Function(name=func["name"], arguments=args_str),
            )
        )
    return result


def _parse_content_tool_calls(
    content: str,
) -> list[ChatCompletionMessageToolCall] | None:
    """Parse tool calls embedded in message content (e.g. Gemma-style responses).

    Some models encode tool calls as a JSON array in ``content`` instead of
    populating the ``tool_calls`` field.  Each element is expected to have
    ``type == "function"`` and a ``function`` object with ``name`` and
    ``arguments``.  ``arguments`` may be a dict (Gemma) or a JSON string
    (standard); both are normalised to a JSON string.

    Returns a list of :class:`ChatCompletionMessageFunctionToolCall` objects,
    or ``None`` if *content* does not match the expected format.
    """
    if not content or not content.strip().startswith("["):
        return None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, list):
        return _parse_item(parsed)

    return None


def _prune_skill_messages(messages: list[dict]) -> list[dict]:
    """Remove old list_skills/read_skill call+result pairs, keeping only the last of each.

    Scans the message list for assistant tool-call messages that invoke
    ``list_skills`` or ``read_skill``.  For each tool name, all but the most
    recent call (and its corresponding ``tool`` result message) are dropped.
    Assistant messages that still have other tool calls or text content are
    kept intact with the removed calls stripped out.
    """
    # Pass 1: find the last tool_call_id for each skip-tool.
    last_id_for: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            name = tc.get("function", {}).get("name", "")
            if name in _COMPACT_SKILL_TOOLS:
                last_id_for[name] = tc["id"]

    if not last_id_for:
        return list(messages)

    # Pass 2: mark tool_call_ids to remove (all but the last per tool name).
    ids_to_remove: set[str] = set()
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            name = tc.get("function", {}).get("name", "")
            if name in _COMPACT_SKILL_TOOLS and tc["id"] != last_id_for[name]:
                ids_to_remove.add(tc["id"])

    if not ids_to_remove:
        return list(messages)

    # Pass 3: rebuild the message list.
    result: list[dict] = []
    for msg in messages:
        role = msg.get("role")

        if role == "tool" and msg.get("tool_call_id") in ids_to_remove:
            continue

        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            kept = [tc for tc in tool_calls if tc["id"] not in ids_to_remove]
            if kept == tool_calls:
                result.append(msg)
            elif kept:
                result.append({**msg, "tool_calls": kept})
            else:
                # All tool calls removed; keep only if there is text content.
                content = msg.get("content") or ""
                if content:
                    result.append({k: v for k, v in msg.items() if k != "tool_calls"})
        else:
            result.append(msg)

    return result


async def _build_compact_summary(messages: list[dict], config: "AgentConfig") -> str:
    """Call the LLM with a summarisation prompt and return the summary text.

    The agent's original system message is replaced with the compact-summary
    instruction so the model focuses on producing a brief narrative rather than
    acting as the agent.
    """
    summary_messages: list[dict] = [
        {"role": "system", "content": _COMPACT_SUMMARY_SYSTEM},
        *[m for m in messages if m.get("role") != "system"],
    ]
    stream = await _acompletion_with_retry(
        config.model,
        summary_messages,
        {},
        config.max_retries,
        config.max_tokens,
        config.reasoning_effort,
    )
    parts: list[str] = []
    async for chunk in stream:
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content:
            parts.append(content)
    return "".join(parts)


class Agent:
    """LLM agent that maintains conversation history and dispatches tool calls.

    Await ``arun()`` with a user message to get the assistant's reply, or use
    ``astream()`` to receive events as they are produced::

        config = AgentConfig(system_prompt="You are a helpful assistant.", model="anthropic/claude-haiku-4-5")
        agent = Agent(config, memory=memory, tool_handler=tool_handler)
        reply = await agent.arun("List the skills available.")
    """

    def __init__(
        self,
        config: AgentConfig,
        memory: MemoryProtocol,
        tool_handler: ToolHandlerProtocol,
    ) -> None:
        self._model = config.model
        self._max_retries = config.max_retries
        self._max_tokens = config.max_tokens
        self._tool_timeout = config.tool_timeout
        self._max_tool_output = config.max_tool_output
        self._memory = memory
        self._reasoning_effort = config.reasoning_effort
        self._session_id = config.session_id
        self._tool_handler = tool_handler
        self._config = config
        self._last_prompt_tokens: int = 0
        self._messages: list[dict] = []
        if config.system_prompt:
            self._messages.append({"role": "system", "content": config.system_prompt})
            logger.debug("[system]\n%s", config.system_prompt)

    @property
    def tool_names(self) -> list[str]:
        return list(self._tool_handler.tool_names)

    @property
    def _tool_kwargs(self) -> dict:
        return self._tool_handler.llm_tool_kwargs

    @property
    def messages(self) -> list[dict]:
        """Read-only view of the current conversation history."""
        return list(self._messages)

    @property
    def model(self) -> str:
        return self._model

    def __repr__(self) -> str:
        return f"Agent(model={self._model!r}, tools={self.tool_names!r})"

    async def astream(self, user_input: str) -> AsyncGenerator[AgentEvent, None]:
        """Stream events produced while processing *user_input*.

        Yields :class:`TokenEvent` for each partial LLM token,
        :class:`ToolStartEvent` / :class:`ToolDoneEvent` around each tool call,
        and a final :class:`DoneEvent` when the agent is done.
        """
        self._messages.append({"role": "user", "content": user_input})
        logger.debug("[user]\n%s", user_input)
        self._memory.store("user", user_input)

        last_fail_key: str | None = None
        consecutive_failures = 0
        model_turns = 0
        tool_calls_executed = 0
        started_at = time.monotonic()

        while True:
            if model_turns >= _MAX_MODEL_TURNS_PER_REQUEST:
                raise RuntimeError(
                    f"aborted: exceeded {_MAX_MODEL_TURNS_PER_REQUEST} model turns in a single request"
                )
            if time.monotonic() - started_at >= _MAX_RUN_SECONDS_PER_REQUEST:
                raise RuntimeError(
                    f"aborted: exceeded {_MAX_RUN_SECONDS_PER_REQUEST}s runtime in a single request"
                )

            stream = await _acompletion_with_retry(
                self._model,
                self._messages,
                self._tool_kwargs,
                self._max_retries,
                self._max_tokens,
                self._reasoning_effort,
            )
            model_turns += 1

            # Accumulate streaming response.
            content_parts: list[str] = []
            tool_call_acc: dict[int, dict] = {}  # index → {id, name, arguments}
            usage = None

            async for chunk in stream:
                if not chunk.choices:
                    if chunk.usage is not None:
                        usage = chunk.usage
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    content_parts.append(delta.content)
                    yield TokenEvent(delta.content)
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        if tc_delta.id:
                            # Tool call start event (identified by non-empty id).
                            # Some providers (e.g. Anthropic via any_llm) hardcode
                            # index=0 for every tool call, so use id-based lookup
                            # instead of the index to correctly handle parallel calls.
                            idx = next(
                                (
                                    k
                                    for k, v in tool_call_acc.items()
                                    if v["id"] == tc_delta.id
                                ),
                                None,
                            )
                            if idx is None:
                                idx = max(tool_call_acc.keys(), default=-1) + 1
                                tool_call_acc[idx] = {
                                    "id": tc_delta.id,
                                    "name": "",
                                    "arguments": "",
                                }
                        else:
                            # Argument delta: belongs to the most recently started call.
                            idx = max(tool_call_acc.keys(), default=tc_delta.index)
                            if idx not in tool_call_acc:
                                tool_call_acc[idx] = {
                                    "id": "",
                                    "name": "",
                                    "arguments": "",
                                }
                        if tc_delta.function:
                            tool_call_acc[idx]["name"] += tc_delta.function.name or ""
                            tool_call_acc[idx]["arguments"] += (
                                tc_delta.function.arguments or ""
                            )
                if chunk.usage is not None:
                    usage = chunk.usage

            content = "".join(content_parts)
            logger.debug(
                "[model_response] content=%r tool_calls=%d",
                content[:120],
                len(tool_call_acc),
            )

            if usage is not None:
                self._last_prompt_tokens = usage.prompt_tokens
                self._memory.record_token_usage(
                    self._session_id or "",
                    self._model,
                    usage.prompt_tokens,
                    usage.completion_tokens,
                )

            # Support models that embed tool calls as a JSON array in content.
            content_tool_calls = None
            if not tool_call_acc and content:
                content_tool_calls = _parse_content_tool_calls(content)

            if tool_call_acc or content_tool_calls:
                if tool_call_acc:
                    sorted_tcs = [tool_call_acc[i] for i in sorted(tool_call_acc)]
                    # Normalise empty arguments to "{}" so _convert_messages_for_anthropic
                    # (which calls json.loads on the stored string) doesn't raise.
                    # Anthropic omits input_json_delta events for tools with no arguments,
                    # leaving the accumulated string as "".
                    for tc in sorted_tcs:
                        if not tc["arguments"]:
                            tc["arguments"] = "{}"
                    assistant_msg: dict = {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": tc["arguments"],
                                },
                            }
                            for tc in sorted_tcs
                        ],
                    }
                    tool_calls_to_run = [
                        ChatCompletionMessageFunctionToolCall(
                            id=tc["id"],
                            type="function",
                            function=Function(
                                name=tc["name"], arguments=tc["arguments"]
                            ),
                        )
                        for tc in sorted_tcs
                    ]
                else:
                    assistant_msg = {"role": "assistant", "content": content}
                    tool_calls_to_run = content_tool_calls  # type: ignore[assignment]

                self._messages.append(assistant_msg)

                for tc in tool_calls_to_run:
                    if tool_calls_executed >= _MAX_TOOL_CALLS_PER_REQUEST:
                        raise RuntimeError(
                            f"aborted: exceeded {_MAX_TOOL_CALLS_PER_REQUEST} tool calls in a single request"
                        )
                    tool_calls_executed += 1
                    target = tc.function.name
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError as e:
                        result = self._tool_handler.get_tool_result(
                            "", f"malformed JSON arguments: {e}", None
                        )
                        logger.error("[tool_result] %s -> %s", target, result.output)
                        self._messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": result.output,
                            }
                        )
                        continue

                    logger.debug("[tool_call] %s args=%s", target, arguments)
                    yield ToolStartEvent(name=target, args=arguments)

                    result = await self._tool_handler.execute(
                        target, arguments, self._max_tool_output
                    )

                    logger.info("[tool_result] %s -> %s", target, result.output)
                    yield ToolDoneEvent(
                        name=target, output=result.output, is_error=result.is_error
                    )

                    self._messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result.output,
                        }
                    )

                    call_key = f"{target}:{tc.function.arguments}"
                    if result.is_error and call_key == last_fail_key:
                        consecutive_failures += 1
                    elif result.is_error:
                        last_fail_key = call_key
                        consecutive_failures = 1
                    else:
                        last_fail_key = None
                        consecutive_failures = 0

                if consecutive_failures >= _MAX_REPEATED_FAILURES:
                    hint = (
                        "You have repeated the same failing tool call "
                        f"{consecutive_failures} times. The arguments appear to be "
                        "incorrect. Try a different approach: rewrite the affected lines, break the "
                        "task into smaller steps, or ask the user for help."
                    )
                    logger.warning("[repeated_failure_hint] %s", hint)
                    self._messages.append({"role": "system", "content": hint})
                    last_fail_key = None
                    consecutive_failures = 0
            else:
                self._messages.append({"role": "assistant", "content": content})
                logger.debug("[assistant]\n%s", content)
                self._memory.store("agent", content)
                yield DoneEvent(content=content)
                return

    async def arun(self, user_input: str) -> str:
        """Send *user_input* to the LLM and return the assistant's final reply.

        Convenience wrapper around :meth:`astream` that discards intermediate
        events and returns the final text.
        """
        async for event in self.astream(user_input):
            if isinstance(event, DoneEvent):
                return event.content
        return ""


class SessionNotFoundError(Exception):
    pass


class AgentManager:
    def __init__(
        self, memory: MemoryProtocol, tool_handler: ToolHandlerProtocol
    ) -> None:
        self._memory = memory
        self._tool_handler = tool_handler
        self._sessions: dict[str, Agent] = {}

    @staticmethod
    def get_session_id() -> str:
        return str(uuid4())

    def create_session(self, config: AgentConfig) -> str:
        session_id = self.get_session_id()
        agent = Agent(
            config._replace(session_id=session_id), self._memory, self._tool_handler
        )
        self._sessions[session_id] = agent
        return session_id

    def get_agent(self, session_id: str) -> Agent:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise SessionNotFoundError(f"Session with id {session_id} not found.")

    def set_confirm_callback(self, confirm: Any) -> None:
        """Register a skill-execution confirmation callback on the tool handler."""
        self._tool_handler.set_confirm(confirm)  # type: ignore[union-attr]

    async def _compact_if_needed(self, session_id: str) -> None:
        """Compact the session's conversation if it has exceeded the token threshold.

        Prunes old ``list_skills``/``read_skill`` exchanges, asks the LLM for a
        brief summary, then replaces the session agent with a fresh instance that
        starts from the summary.
        """
        agent = self._sessions.get(session_id)
        if agent is None:
            return
        threshold = agent._config.compact_threshold
        if threshold <= 0 or agent._last_prompt_tokens < threshold:
            return

        logger.info(
            "[compact] context at %d tokens (threshold %d), compacting...",
            agent._last_prompt_tokens,
            threshold,
        )
        print(
            f"\n[Auto-compacting context ({agent._last_prompt_tokens:,} tokens ≥ "
            f"{threshold:,} threshold)...]\n",
            flush=True,
        )

        pruned = _prune_skill_messages(agent.messages)
        summary = await _build_compact_summary(pruned, agent._config)

        new_agent = Agent(agent._config, self._memory, self._tool_handler)
        new_agent._messages.append(
            {
                "role": "user",
                "content": f"[Conversation summary – continuing]\n{summary}",
            }
        )
        new_agent._messages.append(
            {
                "role": "assistant",
                "content": "Understood. Continuing from where we left off.",
            }
        )
        self._sessions[session_id] = new_agent
        logger.info("[compact] done, fresh agent created with summary")

    async def arun_agent(self, session_id: str, message: str) -> str:
        async for event in self.astream_agent(session_id, message):
            if isinstance(event, DoneEvent):
                return event.content
        return ""

    async def astream_agent(
        self, session_id: str, message: str
    ) -> AsyncGenerator[AgentEvent, None]:
        await self._compact_if_needed(session_id)
        agent = self.get_agent(session_id)
        async for event in agent.astream(message):
            yield event

    def export_conversation(self, session_id: str) -> Path | None:
        agent = self.get_agent(session_id)
        if agent.messages:
            return export_conversation(agent.messages, agent.model)
        return None

    def get_token_stats(self, session_id: str) -> dict:
        """Return aggregated token usage for *session_id*, or an empty dict when unavailable."""
        agent = self.get_agent(session_id)
        return agent._memory.get_session_stats(session_id)
