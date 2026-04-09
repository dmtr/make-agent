from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, NamedTuple
from uuid import uuid4

import any_llm
from any_llm.types.completion import (
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageToolCall,
    Function,
)

from make_agent.app_dirs import default_agents_dir, project_dir
from make_agent.builtin_tools import BUILTIN_SCHEMAS, FILE_TOOL_SCHEMAS, _RunAgent, get_builtin_tools, get_memory_schemas
from make_agent.commands import export_conversation
from make_agent.memory import Memory
from make_agent.parser import parse_file, validate_or_raise
from make_agent.tools import build_tools, format_tool_result, run_tool

_DEFAULT_MODEL = "anthropic/claude-haiku-4-5-20251001"
_DEFAULT_MAX_RETRIES = 5
_DEFAULT_TOOL_TIMEOUT = 600  # seconds
_DEFAULT_MAX_TOOL_OUTPUT = 20000  # characters; 0 = unlimited
_DEFAULT_MAX_TOKENS = 4096
_DEFAULT_REASONING_EFFORT = "auto"
_MAX_REPEATED_FAILURES = 3

logger = logging.getLogger(__name__)


class AgentConfig(NamedTuple):
    makefile_path: Path
    model: str = _DEFAULT_MODEL
    max_retries: int = _DEFAULT_MAX_RETRIES
    tool_timeout: int = _DEFAULT_TOOL_TIMEOUT
    max_tool_output: int = _DEFAULT_MAX_TOOL_OUTPUT
    max_tokens: int = _DEFAULT_MAX_TOKENS
    agents_dir: str | None = None
    debug: bool = False
    disabled_builtin_tools: frozenset[str] = frozenset()
    reasoning_effort: str = _DEFAULT_REASONING_EFFORT
    session_id: str | None = None


def _parse_retry_after(e: any_llm.RateLimitError) -> float | None:
    """Return the wait time in seconds from a RateLimitError's response headers.

    Checks ``retry-after-ms`` (milliseconds) then ``retry-after`` (seconds).
    Returns ``None`` when neither header is present.
    """
    try:
        orig = e.original_exception
        headers = orig.response.headers if orig is not None and hasattr(orig, "response") and orig.response is not None else {}
    except Exception:
        return None
    if ms := headers.get("retry-after-ms"):
        return float(ms) / 1000
    if sec := headers.get("retry-after"):
        return float(sec)
    return None


def _completion_with_retry(
    model: str,
    messages: list[dict],
    tool_kwargs: dict[str, Any],
    max_retries: int,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    reasoning_effort: str = _DEFAULT_REASONING_EFFORT,
) -> Any:
    """Call ``any_llm.completion``, retrying on rate limit up to *max_retries* times.

    On each ``RateLimitError`` the wait time is read from the ``Retry-After``
    response header when present, otherwise exponential backoff is used
    (``2^attempt`` seconds, capped at 60 s).  A message is printed before
    each retry so the user can see what is happening.
    """
    for attempt in range(max_retries + 1):
        try:
            return any_llm.completion(model=model, messages=messages, max_tokens=max_tokens, reasoning_effort=reasoning_effort, **tool_kwargs)
        except any_llm.RateLimitError as e:
            if attempt == max_retries:
                raise
            wait = _parse_retry_after(e) or min(2**attempt, 60)
            print(
                f"Rate limited, retrying in {wait:.0f}s" f" (attempt {attempt + 1}/{max_retries})...",
                flush=True,
            )
            time.sleep(wait)


def _parse_content_tool_calls(content: str) -> list[ChatCompletionMessageToolCall] | None:
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
    if not isinstance(parsed, list):
        return None
    result: list[ChatCompletionMessageToolCall] = []
    for item in parsed:
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
    return result or None


class Agent:
    """LLM agent that maintains conversation history and dispatches tool calls.

    Call the instance with a user message to get the assistant's reply::

        agent = Agent(Path("Makefile"), model="anthropic/claude-haiku-4-5-20251001")
        reply = agent("List the files in the current directory.")
    """

    def __init__(self, config: AgentConfig, memory: Memory | None) -> None:
        mf = parse_file(config.makefile_path)
        validate_or_raise(mf)
        self._model = config.model
        self._makefile_path = config.makefile_path
        self._max_retries = config.max_retries
        self._max_tokens = config.max_tokens
        self._tool_timeout = config.tool_timeout
        self._max_tool_output = config.max_tool_output
        self._memory = memory
        self._reasoning_effort = config.reasoning_effort
        self._session_id = config.session_id
        agents_dir = config.agents_dir if config.agents_dir is not None else default_agents_dir()
        self._agents_dir = agents_dir
        self._disabled_builtin_tools = config.disabled_builtin_tools
        self._builtins = get_builtin_tools(agents_dir, memory, config.disabled_builtin_tools, config.tool_timeout, config.makefile_path.stem)
        makefile_tools = build_tools(mf)
        memory_schemas = get_memory_schemas() if memory is not None else []
        active_builtin_schemas = [s for s in BUILTIN_SCHEMAS if s["function"]["name"] not in config.disabled_builtin_tools]
        active_memory_schemas = [s for s in memory_schemas if s["function"]["name"] not in config.disabled_builtin_tools]
        active_file_schemas = [s for s in FILE_TOOL_SCHEMAS if s["function"]["name"] not in config.disabled_builtin_tools]
        self._static_schemas = active_builtin_schemas + active_memory_schemas + active_file_schemas
        self._tools = self._static_schemas + makefile_tools
        self._tool_kwargs: dict = {"tools": self._tools, "tool_choice": "auto"} if self._tools else {}
        self._messages: list[dict] = []
        if mf.system_prompt:
            self._messages.append({"role": "system", "content": mf.system_prompt})
            logger.debug("[system]\n%s", mf.system_prompt)

    @property
    def tool_names(self) -> list[str]:
        return [t["function"]["name"] for t in self._tools]

    @property
    def messages(self) -> list[dict]:
        """Read-only view of the current conversation history."""
        return list(self._messages)

    @property
    def model(self) -> str:
        return self._model

    def _run_agent(self, mk_path: Path, prompt: str) -> str:
        """Instantiate a specialist agent in-process and return its response."""
        sub_disabled = self._disabled_builtin_tools | frozenset({"run_agent"})
        sub_config = AgentConfig(
            makefile_path=mk_path,
            model=self._model,
            max_retries=self._max_retries,
            tool_timeout=self._tool_timeout,
            max_tool_output=self._max_tool_output,
            max_tokens=self._max_tokens,
            agents_dir=self._agents_dir,
            disabled_builtin_tools=sub_disabled,
            reasoning_effort=self._reasoning_effort,
            session_id=self._session_id,
        )
        return Agent(sub_config, self._memory)(prompt)

    def __repr__(self) -> str:
        return f"Agent(model={self._model!r}, tools={self.tool_names!r})"

    def __call__(self, user_input: str) -> str:
        """Send *user_input* to the LLM and return the assistant's reply.

        Dispatches tool calls in a loop until the model returns a plain
        text response.
        """
        self._messages.append({"role": "user", "content": user_input})
        logger.debug("[user]\n%s", user_input)
        if self._memory is not None:
            self._memory.store("user", user_input)

        # Track consecutive identical failing tool calls to detect loops.
        last_fail_key: str | None = None
        consecutive_failures = 0

        while True:
            response = _completion_with_retry(
                self._model,
                self._messages,
                self._tool_kwargs,
                self._max_retries,
                self._max_tokens,
                self._reasoning_effort,
            )
            msg = response.choices[0].message
            logger.debug("[model_response]\n%s", msg)

            if self._memory is not None and response.usage is not None:
                self._memory.record_token_usage(
                    self._session_id or "",
                    self._makefile_path.name,
                    self._model,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )

            tool_calls = msg.tool_calls or _parse_content_tool_calls(msg.content or "")
            if tool_calls:
                self._messages.append(msg.model_dump(exclude_none=True))

                for tc in tool_calls:
                    target = tc.function.name
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError as e:
                        output = format_tool_result("", f"malformed JSON arguments: {e}", None)
                        logger.debug("[tool_result] %s -> %s", target, output)
                        self._messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})
                        continue

                    logger.debug("[tool_call] %s args=%s", target, arguments)
                    try:
                        if target in self._builtins:
                            raw = self._builtins[target](**arguments)
                            if isinstance(raw, _RunAgent):
                                result = self._run_agent(raw.mk_path, raw.prompt)
                                output = format_tool_result(result, "", 0, self._max_tool_output)
                            else:
                                output = format_tool_result(str(raw), "", 0, self._max_tool_output)
                        else:
                            output = run_tool(
                                target,
                                arguments,
                                self._makefile_path,
                                self._tool_timeout,
                                self._max_tool_output,
                            )
                    except TypeError as e:
                        logger.error("argument type error when running tool %s: %s", target, e)
                        output = format_tool_result("", f"argument type error: {e}", None)
                    except Exception as e:
                        logger.error("unexpected error when running tool %s: %s", target, e)
                        output = format_tool_result("", f"unexpected error: {e}", None)
                    logger.debug("[tool_result] %s -> %s", target, output)

                    self._messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": output,
                        }
                    )

                    # Detect repeated identical failing tool calls.
                    is_error = '"error"' in output or '"stderr"' in output
                    call_key = f"{target}:{tc.function.arguments}"
                    if is_error and call_key == last_fail_key:
                        consecutive_failures += 1
                    elif is_error:
                        last_fail_key = call_key
                        consecutive_failures = 1
                    else:
                        last_fail_key = None
                        consecutive_failures = 0

                if consecutive_failures >= _MAX_REPEATED_FAILURES:
                    hint = (
                        "You have repeated the same failing tool call "
                        f"{consecutive_failures} times. The arguments appear to be "
                        "incorrect. Try a different approach: use a simpler tool "
                        "(e.g. write_file instead of replace_lines), break the "
                        "task into smaller steps, or ask the user for help."
                    )
                    logger.debug("[repeated_failure_hint] %s", hint)
                    self._messages.append({"role": "system", "content": hint})
                    last_fail_key = None
                    consecutive_failures = 0
            else:
                content = msg.content or ""
                self._messages.append({"role": "assistant", "content": content})
                logger.debug("[assistant]\n%s", content)
                if self._memory is not None:
                    self._memory.store("agent", content)
                return content


class SessionNotFoundError(Exception):
    pass


class AgentManager:

    def __init__(self):
        self._sessions = {}

    @staticmethod
    def get_session_id() -> str:
        return str(uuid4())

    def create_session(self, config: AgentConfig, with_memory: bool = False) -> str:
        session_id = self.get_session_id()

        memory = None
        if with_memory:
            memory = self.init_memory(session_id)

        agent = Agent(config._replace(session_id=session_id), memory)
        self._sessions[session_id] = agent

        return session_id

    def get_agent(self, session_id: str) -> Agent:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise SessionNotFoundError(f"Session with id {session_id} not found.")

    def notify_agent(self, session_id: str, message: str) -> str:
        agent = self.get_agent(session_id)
        return agent(message)

    def export_conversation(self, session_id: str) -> Path | None:
        agent = self.get_agent(session_id)
        if agent.messages:
            return export_conversation(agent.messages, agent.model)
        return None

    def get_token_stats(self, session_id: str) -> dict:
        """Return aggregated token usage for *session_id*, or an empty dict when unavailable."""
        agent = self.get_agent(session_id)
        if agent._memory is None:
            return {}
        return agent._memory.get_session_stats(session_id)

    def init_memory(self, session_id: str) -> Memory:
        db_path = project_dir() / "memory.db"
        return Memory(db_path)
