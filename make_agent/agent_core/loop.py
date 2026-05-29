"""LLM agent conversation loop and associated types."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, AsyncIterator, Awaitable, Callable, NamedTuple

from make_agent.protocols import ToolHandlerProtocol

from .constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TOOL_OUTPUT,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_TOOL_TIMEOUT,
    DEFAULT_USE_PROMPT_CACHE,
    MAX_MODEL_TURNS_PER_REQUEST,
    MAX_REPEATED_FAILURES,
    MAX_RUN_SECONDS_PER_REQUEST,
    MAX_TOOL_CALLS_PER_REQUEST,
)
from make_agent.provider import acompletion_with_retry, is_anthropic_model, is_context_exceeded, is_corrupt_message_history

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


class UsageCallback(CallBack):
    """Token usage from one LLM model turn. Fire-and-forget; always ready."""

    def __init__(self, model: str, input_tokens: int, output_tokens: int) -> None:
        super().__init__("")
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class CompactCallback(CallBack):
    """Context window was compacted. Fire-and-forget; always ready."""

    def __init__(self, messages_removed: int) -> None:
        super().__init__("")
        self.messages_removed = messages_removed


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
    compact_fn: Callable[[list[dict]], Awaitable[list[dict]]] | None = None


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


def _parse_content_tool_calls(
    content: str,
) -> list[_ToolCall] | None:
    """Parse tool calls embedded in message content (e.g. Gemma-style responses).

    Some models encode tool calls as a JSON array in ``content`` instead of
    populating the ``tool_calls`` field.  Each element is expected to have
    ``type == "function"`` and a ``function`` object with ``name`` and
    ``arguments``.  ``arguments`` may be a dict (Gemma) or a JSON string
    (standard); both are normalised to a JSON string.

    Returns a list of :class:`_ToolCall` objects,
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
        self._tool_handler = tool_handler
        self._config = config
        self._compact_fn = config.compact_fn
        self._messages: list[dict] = []
        self._gen: AsyncGenerator[CallBack, None] | None = None
        if config.system_prompt:
            if config.use_prompt_cache and is_anthropic_model(config.model):
                system_content: str | list = [
                    {
                        "type": "text",
                        "text": config.system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                system_content = config.system_prompt
            self._messages.append({"role": "system", "content": system_content})
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

        while True:
            if model_turns >= MAX_MODEL_TURNS_PER_REQUEST:
                raise RuntimeError(f"aborted: exceeded {MAX_MODEL_TURNS_PER_REQUEST} model turns in a single request")
            if time.monotonic() - started_at >= MAX_RUN_SECONDS_PER_REQUEST:
                raise RuntimeError(f"aborted: exceeded {MAX_RUN_SECONDS_PER_REQUEST}s runtime in a single request")

            try:
                stream = await acompletion_with_retry(
                    self._model,
                    self._messages,
                    self._tool_kwargs,
                    self._max_retries,
                    self._max_tokens,
                    self._reasoning_effort,
                )
            except Exception as exc:
                if self._compact_fn and (is_context_exceeded(exc) or is_corrupt_message_history(exc)):
                    logger.info("Context window exceeded; attempting compaction")
                    pruned = await self._compact_fn(self._messages)
                    removed = len(self._messages) - len(pruned)
                    if removed > 0:
                        self._messages = pruned
                        yield CompactCallback(messages_removed=removed)
                        continue
                raise
            model_turns += 1

            # Accumulate streaming response.
            content_parts: list[str] = []
            tool_call_acc: dict[int, dict] = {}  # index → {id, name, arguments}
            usage = None

            try:
                async for chunk in stream:
                    if not chunk.choices:
                        if getattr(chunk, "usage", None) is not None:
                            usage = chunk.usage
                        continue
                    delta = chunk.choices[0].delta
                    if delta.content:
                        content_parts.append(delta.content)
                        yield TokenCallback(delta.content)
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            if tc_delta.id:
                                # Tool call start event (identified by non-empty id).
                                # Some providers (e.g. Anthropic via any_llm) hardcode
                                # index=0 for every tool call, so use id-based lookup
                                # instead of the index to correctly handle parallel calls.
                                idx = next(
                                    (k for k, v in tool_call_acc.items() if v["id"] == tc_delta.id),
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
                                tool_call_acc[idx]["arguments"] += tc_delta.function.arguments or ""
                    if getattr(chunk, "usage", None) is not None:
                        usage = chunk.usage
            except Exception as exc:
                if self._compact_fn and (is_context_exceeded(exc) or is_corrupt_message_history(exc)):
                    pruned = await self._compact_fn(self._messages)
                    removed = len(self._messages) - len(pruned)
                    if removed > 0:
                        self._messages = pruned
                        yield CompactCallback(messages_removed=removed)
                        continue
                raise

            content = "".join(content_parts)
            logger.debug(
                "[model_response] content=%r tool_calls=%d",
                content[:120],
                len(tool_call_acc),
            )

            if usage is not None:
                yield UsageCallback(
                    model=self._model,
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
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
                        _ToolCall(
                            id=tc["id"],
                            type="function",
                            function=_Function(name=tc["name"], arguments=tc["arguments"]),
                        )
                        for tc in sorted_tcs
                    ]
                else:
                    assistant_msg = {"role": "assistant", "content": content}
                    tool_calls_to_run = content_tool_calls  # type: ignore[assignment]

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
