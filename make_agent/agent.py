"""Interactive REPL agent loop for the make-agent."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, NamedTuple

import any_llm

from make_agent.app_dirs import default_agents_dir
from make_agent.builtin_tools import BUILTIN_SCHEMAS, get_builtin_tools, get_memory_schemas
from make_agent.memory import Memory
from make_agent.parser import parse_file, validate_or_raise
from make_agent.tools import build_tools, format_tool_result, run_tool

_DEFAULT_MODEL = "anthropic/claude-haiku-4-5-20251001"
_DEFAULT_MAX_RETRIES = 5
_DEFAULT_TOOL_TIMEOUT = 600  # seconds
_DEFAULT_MAX_TOOL_OUTPUT = 20000  # characters; 0 = unlimited

logger = logging.getLogger(__name__)


class AgentConfig(NamedTuple):
    makefile_path: Path
    model: str = _DEFAULT_MODEL
    max_retries: int = _DEFAULT_MAX_RETRIES
    tool_timeout: int = _DEFAULT_TOOL_TIMEOUT
    max_tool_output: int = _DEFAULT_MAX_TOOL_OUTPUT
    agents_dir: str | None = None
    debug: bool = False
    memory: Memory | None = None


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
) -> Any:
    """Call ``any_llm.completion``, retrying on rate limit up to *max_retries* times.

    On each ``RateLimitError`` the wait time is read from the ``Retry-After``
    response header when present, otherwise exponential backoff is used
    (``2^attempt`` seconds, capped at 60 s).  A message is printed before
    each retry so the user can see what is happening.
    """
    for attempt in range(max_retries + 1):
        try:
            return any_llm.completion(model=model, messages=messages, **tool_kwargs)
        except any_llm.RateLimitError as e:
            if attempt == max_retries:
                raise
            wait = _parse_retry_after(e) or min(2**attempt, 60)
            print(
                f"Rate limited, retrying in {wait:.0f}s" f" (attempt {attempt + 1}/{max_retries})...",
                flush=True,
            )
            time.sleep(wait)


class Agent:
    """LLM agent that maintains conversation history and dispatches tool calls.

    Call the instance with a user message to get the assistant's reply::

        agent = Agent(Path("Makefile"), model="anthropic/claude-haiku-4-5-20251001")
        reply = agent("List the files in the current directory.")
    """

    def __init__(self, config: AgentConfig) -> None:
        mf = parse_file(config.makefile_path)
        validate_or_raise(mf)
        self._model = config.model
        self._makefile_path = config.makefile_path
        self._max_retries = config.max_retries
        self._tool_timeout = config.tool_timeout
        self._max_tool_output = config.max_tool_output
        self._memory = config.memory
        agents_dir = config.agents_dir if config.agents_dir is not None else default_agents_dir()
        self._builtins = get_builtin_tools(agents_dir, config.model, config.debug, config.memory)
        makefile_tools = build_tools(mf)
        memory_schemas = get_memory_schemas() if config.memory is not None else []
        self._tools = BUILTIN_SCHEMAS + memory_schemas + makefile_tools
        self._tool_kwargs: dict = {"tools": self._tools, "tool_choice": "auto"} if self._tools else {}
        self._messages: list[dict] = []
        if mf.system_prompt:
            self._messages.append({"role": "system", "content": mf.system_prompt})
            logger.debug("[system]\n%s", mf.system_prompt)

    @property
    def tool_names(self) -> list[str]:
        return [t["function"]["name"] for t in self._tools]

    def __call__(self, user_input: str) -> str:
        """Send *user_input* to the LLM and return the assistant's reply.

        Dispatches tool calls in a loop until the model returns a plain
        text response.
        """
        self._messages.append({"role": "user", "content": user_input})
        logger.debug("[user]\n%s", user_input)
        if self._memory is not None:
            self._memory.store("user", user_input)

        while True:
            response = _completion_with_retry(
                self._model,
                self._messages,
                self._tool_kwargs,
                self._max_retries,
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                self._messages.append(msg.model_dump(exclude_none=True))

                for tc in msg.tool_calls:
                    target = tc.function.name
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                    logger.debug("[tool_call] %s args=%s", target, arguments)
                    try:
                        if target in self._builtins:
                            raw = self._builtins[target](**arguments)
                            output = format_tool_result(raw, "", 0, self._max_tool_output)
                        else:
                            output = run_tool(
                                target,
                                arguments,
                                self._makefile_path,
                                self._tool_timeout,
                                self._max_tool_output,
                            )
                    except Exception as e:
                        output = format_tool_result("", f"unexpected error: {e}", None)
                    logger.debug("[tool_result] %s -> %s", target, output)

                    self._messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": output,
                        }
                    )
            else:
                content = msg.content or ""
                self._messages.append({"role": "assistant", "content": content})
                logger.debug("[assistant]\n%s", content)
                if self._memory is not None:
                    self._memory.store("agent", content)
                return content
