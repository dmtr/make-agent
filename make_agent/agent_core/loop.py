"""LLM agent conversation loop and associated types."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, AsyncIterator, NamedTuple

from make_agent.protocols import ToolHandlerProtocol

from .constants import (
    COMPACT_SUMMARY_MAX_TOKENS,
    DEFAULT_COMPACT_MODE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TOOL_OUTPUT,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_TOOL_TIMEOUT,
    DEFAULT_USE_PROMPT_CACHE,
    MAX_COMPACT_RETRIES,
    MAX_MODEL_TURNS_PER_REQUEST,
    MAX_REPEATED_FAILURES,
    MAX_RUN_SECONDS_PER_REQUEST,
    MAX_TOOL_CALLS_PER_REQUEST,
)
from make_agent.provider import ContextExceededChunk, Provider, TextDelta, ToolCallDelta, ToolCallStart, UsageDelta, provider_for

logger = logging.getLogger(__name__)


@dataclass
class _Function:
    name: str
    arguments: str


@dataclass
class _ToolCall:
    id: str
    type: str
    function: _Function


class CallBack:
    """Base class for all agentic loop callbacks.

    Callbacks represent events emitted during LLM conversation turns:
    token streaming, tool requests, messages, and token usage.
    The loop yields callbacks; AgentManager handles them and converts them to events.
    """

    def __init__(self, message: str) -> None:
        self._message = message
        self._response: str | None = None

    @property
    def message(self) -> str:
        return self._message

    @property
    def ready(self) -> bool:
        return True

    def set_response(self, response: str, is_error: bool = False) -> None:
        self._response = response

    def __call__(self) -> str | None:
        return self._response


class TokenCallback(CallBack):
    """A partial token streamed from the LLM. Fire-and-forget; always ready.

    AgentManager converts this to a TokenEvent for external consumption.
    """


class MessageCallback(CallBack):
    """LLM produced a final text response (no tool calls). Terminal.

    AgentManager stores this message and converts it to a DoneEvent.
    """


class ToolCallback(CallBack):
    """LLM requested a tool call. Blocks the loop until set_response() is called."""

    def __init__(
        self,
        message: str,
        tool_name: str,
        tool_args: dict,
        tool_call_id: str,
        description: str = "",
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.tool_call_id = tool_call_id
        self.description = description
        self.duration_ms: float | None = None
        self._is_error: bool = False
        self._event: asyncio.Event = asyncio.Event()

    @property
    def ready(self) -> bool:
        return self._event.is_set()

    @property
    def is_error(self) -> bool:
        return self._is_error

    @property
    def output(self) -> str:
        return self._response or ""

    def set_response(self, response: str, is_error: bool = False) -> None:
        self._response = response
        self._is_error = is_error
        self._event.set()

    async def wait(self) -> None:
        await self._event.wait()


class CompactCallback(CallBack):
    """Loop compacted history due to context-window overflow."""

    def __init__(self, attempt: int, messages_dropped: int, turns_kept: int) -> None:
        super().__init__("")
        self.attempt = attempt
        self.messages_dropped = messages_dropped
        self.turns_kept = turns_kept


class UsageCallback(CallBack):
    """Token usage from one LLM model turn. Fire-and-forget; always ready."""

    def __init__(self, model: str, input_tokens: int, output_tokens: int) -> None:
        super().__init__("")
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class AgentConfig(NamedTuple):
    system_prompt: str
    model: str
    max_retries: int = DEFAULT_MAX_RETRIES
    tool_timeout: int = DEFAULT_TOOL_TIMEOUT
    max_tool_output: int = DEFAULT_MAX_TOOL_OUTPUT
    max_tokens: int = DEFAULT_MAX_TOKENS
    skills_dir: str = ""
    disabled_builtin_tools: frozenset[str] = frozenset()
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    session_id: str | None = None
    project_dir: Path = Path()
    use_prompt_cache: bool = DEFAULT_USE_PROMPT_CACHE
    provider: Any = None  # Provider | None; resolved via provider_for(model) when None
    compact_mode: str = DEFAULT_COMPACT_MODE  # "drop" | "summarize"


def _parse_item(doc: Any) -> list[_ToolCall] | None:
    result: list[_ToolCall] = []
    for item in doc:
        if not isinstance(item, dict) or item.get("type") != "function":
            return None
        func = item.get("function", {})
        if "name" not in func:
            return None
        args = func.get("arguments", {})
        args_str = json.dumps(args) if isinstance(args, dict) else args
        result.append(
            _ToolCall(
                id=item.get("id", ""),
                type="function",
                function=_Function(name=func["name"], arguments=args_str),
            )
        )
    return result


class AgenticLoop:
    """LLM agent async iterator for streaming conversation turns.

    Maintains conversation history across turns. Call :meth:`astream` with a
    user message to begin a turn; then iterate with ``async for cb in loop``
    to receive :class:`CallBack` objects.  :class:`ToolCallback` objects must
    have :meth:`~ToolCallback.set_response` called before iteration can
    continue (done automatically by :class:`AgentManager`).

    Note: Memory storage (user input, messages, token usage) is handled by
    :class:`AgentManager`, not by the loop itself. The loop focuses solely on
    conversation management and tool dispatching.
    """

    def __init__(
        self,
        config: AgentConfig,
        tool_handler: ToolHandlerProtocol,
    ) -> None:
        self._model = config.model
        self._max_retries = config.max_retries
        self._max_tokens = config.max_tokens
        self._tool_timeout = config.tool_timeout
        self._max_tool_output = config.max_tool_output
        self._reasoning_effort = config.reasoning_effort
        self._use_prompt_cache = config.use_prompt_cache
        self._tool_handler = tool_handler
        self._config = config
        self._provider: Provider = config.provider if config.provider is not None else provider_for(config.model)
        self._compact_mode: str = config.compact_mode
        self._messages: list[dict] = []
        self._turn_count: int = 0
        self._gen: AsyncGenerator[CallBack, None] | None = None
        if config.system_prompt:
            self._messages.append({"role": "system", "content": config.system_prompt})
            logger.debug("[system]\n%s", config.system_prompt)

    @property
    def tool_names(self) -> list[str]:
        return list(self._tool_handler.tool_names)

    @property
    def _tools(self) -> list[dict]:
        """OpenAI-format tool schemas to pass to the provider."""
        return self._tool_handler.llm_tool_kwargs.get("tools", [])

    @property
    def messages(self) -> list[dict]:
        """Read-only view of the current conversation history."""
        return list(self._messages)

    @property
    def model(self) -> str:
        return self._model

    def _get_tool_description(self, tool_name: str) -> str:
        for schema in self._tool_handler.schemas:
            if schema.get("function", {}).get("name") == tool_name:
                return schema.get("function", {}).get("description", "")
        return ""

    def __repr__(self) -> str:
        return f"AgenticLoop(model={self._model!r}, tools={self.tool_names!r})"

    def __aiter__(self) -> AsyncIterator[CallBack]:
        return self

    async def __anext__(self) -> CallBack:
        if self._gen is None:
            raise StopAsyncIteration
        try:
            return await anext(self._gen)
        except StopAsyncIteration:
            self._gen = None
            raise

    def astream(self, user_input: str) -> AgenticLoop:
        """Begin a new turn for *user_input*. Returns ``self`` for use in ``async for``."""
        self._turn_count += 1
        self._gen = self._run(user_input)
        return self

    async def _run(self, user_input: str) -> AsyncGenerator[CallBack, None]:
        """Internal async generator: manages LLM turns, yields CallBack objects."""
        self._messages.append({"role": "user", "content": user_input})
        logger.debug("[user]\n%s", user_input)

        last_fail_key: str | None = None
        consecutive_failures = 0
        model_turns = 0
        tool_calls_executed = 0
        started_at = time.monotonic()
        compact_attempts = 0

        while True:
            if model_turns >= MAX_MODEL_TURNS_PER_REQUEST:
                raise RuntimeError(f"aborted: exceeded {MAX_MODEL_TURNS_PER_REQUEST} model turns in a single request")
            if time.monotonic() - started_at >= MAX_RUN_SECONDS_PER_REQUEST:
                raise RuntimeError(f"aborted: exceeded {MAX_RUN_SECONDS_PER_REQUEST}s runtime in a single request")

            # Snapshot before each LLM call so we can restore on context overflow.
            snapshot = list(self._messages)

            # Accumulate streaming response.
            content_parts: list[str] = []
            tool_call_acc: dict[int, dict] = {}  # index → {id, name, arguments}
            input_tokens = 0
            output_tokens = 0
            context_exceeded = False

            async for chunk in self._provider.astream(
                self._model,
                self._messages,
                self._tools,
                self._max_tokens,
                use_prompt_cache=self._use_prompt_cache,
                reasoning_effort=self._reasoning_effort,
            ):
                if isinstance(chunk, TextDelta):
                    content_parts.append(chunk.text)
                    yield TokenCallback(chunk.text)
                elif isinstance(chunk, ContextExceededChunk):
                    context_exceeded = True
                    break
                elif isinstance(chunk, ToolCallStart):
                    tool_call_acc[chunk.index] = {"id": chunk.id, "name": chunk.name, "arguments": ""}
                elif isinstance(chunk, ToolCallDelta):
                    if chunk.index in tool_call_acc:
                        tool_call_acc[chunk.index]["arguments"] += chunk.args_delta
                elif isinstance(chunk, UsageDelta):
                    input_tokens += chunk.input_tokens
                    output_tokens += chunk.output_tokens

            if context_exceeded:
                if compact_attempts >= MAX_COMPACT_RETRIES:
                    raise RuntimeError(
                        f"aborted: context window exceeded after {MAX_COMPACT_RETRIES} compact attempts"
                    )
                self._messages = snapshot
                if self._compact_mode == "summarize":
                    dropped, kept = await self._smart_compact()
                else:
                    dropped, kept = self.compact_history()
                if dropped == 0:
                    raise RuntimeError("aborted: context window exceeded and no messages can be compacted")
                compact_attempts += 1
                logger.warning(
                    "[compact] context exceeded — dropped %d messages, kept %d turns (attempt %d/%d)",
                    dropped, kept, compact_attempts, MAX_COMPACT_RETRIES,
                )
                yield CompactCallback(attempt=compact_attempts, messages_dropped=dropped, turns_kept=kept)
                continue

            content = "".join(content_parts)
            model_turns += 1

            logger.debug(
                "[model_response] content=%r tool_calls=%d",
                content[:120],
                len(tool_call_acc),
            )

            if input_tokens or output_tokens:
                yield UsageCallback(
                    model=self._model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            if tool_call_acc:
                # Normalise empty arguments to "{}" (tools with no args have no delta).
                for tc in tool_call_acc.values():
                    if not tc["arguments"]:
                        tc["arguments"] = "{}"

                sorted_tcs = [tool_call_acc[i] for i in sorted(tool_call_acc)]
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
                    _ToolCall(
                        id=tc["id"],
                        type="function",
                        function=_Function(name=tc["name"], arguments=tc["arguments"]),
                    )
                    for tc in sorted_tcs
                ]

                self._messages.append(assistant_msg)

                for tc in tool_calls_to_run:
                    if tool_calls_executed >= MAX_TOOL_CALLS_PER_REQUEST:
                        raise RuntimeError(f"aborted: exceeded {MAX_TOOL_CALLS_PER_REQUEST} tool calls in a single request")
                    tool_calls_executed += 1
                    target = tc.function.name
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError as e:
                        error_output = self._tool_handler.get_tool_result("", f"malformed JSON arguments: {e}", None).output
                        logger.error("[tool_result] %s -> %s", target, error_output)
                        self._messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": error_output,
                            }
                        )
                        call_key = f"{target}:{tc.function.arguments}"
                        if call_key == last_fail_key:
                            consecutive_failures += 1
                        else:
                            last_fail_key = call_key
                            consecutive_failures = 1
                        continue

                    logger.debug("[tool_call] %s args=%s", target, arguments)
                    tc_cb = ToolCallback(
                        message=tc.function.arguments,
                        tool_name=target,
                        tool_args=arguments,
                        tool_call_id=tc.id,
                        description=self._get_tool_description(target),
                    )
                    yield tc_cb
                    await tc_cb.wait()  # blocks until set_response() is called

                    result_output = tc_cb.output
                    logger.info("[tool_result] %s -> %s", target, result_output)
                    self._messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result_output,
                        }
                    )

                    call_key = f"{target}:{tc.function.arguments}"
                    if tc_cb.is_error and call_key == last_fail_key:
                        consecutive_failures += 1
                    elif tc_cb.is_error:
                        last_fail_key = call_key
                        consecutive_failures = 1
                    else:
                        last_fail_key = None
                        consecutive_failures = 0

                if consecutive_failures >= MAX_REPEATED_FAILURES:
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
                yield MessageCallback(content)
                return

    def compact_history(self) -> tuple[int, int]:
        """Drop turns to fit within context. Returns ``(messages_dropped, turns_kept)``.

        Policy: keep the last 2 turns when there are more than 2, otherwise keep
        the last 1 turn.  System messages are never touched.  Returns
        ``(0, len(turns))`` when there is only one turn (nothing to drop).
        """
        system = [m for m in self._messages if m.get("role") == "system"]
        non_system = [m for m in self._messages if m.get("role") != "system"]

        turns: list[list[dict]] = []
        for msg in non_system:
            if msg.get("role") == "user":
                turns.append([msg])
            elif turns:
                turns[-1].append(msg)

        if len(turns) <= 1:
            return 0, len(turns)

        keep = 2 if len(turns) > 2 else 1
        kept = [m for turn in turns[-keep:] for m in turn]
        old_len = len(self._messages)
        self._messages = system + kept
        return old_len - len(self._messages), keep

    async def _summarize_turn(self, msgs: list[dict]) -> str:
        """Ask the LLM for a one-paragraph summary of a single conversation turn."""
        lines: list[str] = []
        for msg in msgs:
            role = msg.get("role", "")
            content = str(msg.get("content") or "")
            if role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant":
                tool_calls = msg.get("tool_calls") or []
                if tool_calls:
                    names = ", ".join(tc.get("function", {}).get("name", "?") for tc in tool_calls)
                    lines.append(f"Assistant called tools: {names}")
                if content:
                    lines.append(f"Assistant: {content}")
            elif role == "tool":
                lines.append(f"Tool result: {content[:300]}")

        prompt = (
            "Summarize the following conversation turn in one concise paragraph. "
            "Focus on what was asked, what actions were taken, and any key outcomes or decisions.\n\n"
            + "\n".join(lines)
        )
        parts: list[str] = []
        async for chunk in self._provider.astream(
            self._model,
            [{"role": "user", "content": prompt}],
            [],
            COMPACT_SUMMARY_MAX_TOKENS,
            use_prompt_cache=False,
            reasoning_effort=self._reasoning_effort,
        ):
            if isinstance(chunk, TextDelta):
                parts.append(chunk.text)
        return "".join(parts).strip()

    async def _smart_compact(self) -> tuple[int, int]:
        """Summarize all prior turns in parallel and replace them with a combined
        summary system message. The last (current) turn is always preserved wholesale,
        whether or not it contains tool calls.
        Returns ``(messages_dropped, turns_summarized)``.
        """
        system = [m for m in self._messages if m.get("role") == "system"]
        non_system = [m for m in self._messages if m.get("role") != "system"]

        turns: list[list[dict]] = []
        for msg in non_system:
            if msg.get("role") == "user":
                turns.append([msg])
            elif turns:
                turns[-1].append(msg)

        if len(turns) <= 1:
            return 0, 0

        prior_turns = turns[:-1]
        current_turn = turns[-1]

        summaries = await asyncio.gather(*[self._summarize_turn(t) for t in prior_turns])
        combined = "\n".join(f"Turn {i + 1}: {s}" for i, s in enumerate(summaries))
        summary_msg = {"role": "system", "content": f"Prior conversation summary:\n{combined}"}

        old_len = len(self._messages)
        self._messages = system + [summary_msg] + current_turn
        return old_len - len(self._messages), len(prior_turns)

    async def arun(self, user_input: str) -> str:
        """Send *user_input* to the LLM and return the assistant's final reply.

        Executes tool calls inline using the configured tool handler.
        Convenience wrapper for scripts and tests; use :class:`AgentManager`
        for full streaming and display control.
        """
        async for cb in self.astream(user_input):
            if isinstance(cb, ToolCallback):
                result = await self._tool_handler.execute(cb.tool_name, cb.tool_args, self._max_tool_output)
                cb.set_response(result.output, is_error=result.is_error)
            elif isinstance(cb, MessageCallback):
                return cb.message
        return ""
