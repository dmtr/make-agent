"""Interactive REPL agent loop for the make-agent."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, NamedTuple

import litellm

from make_agent.parser import parse_file, validate_or_raise
from make_agent.tools import build_tools, get_content_params, run_tool

_DEFAULT_MODEL = "anthropic/claude-haiku-4-5-20251001"
_DEFAULT_MAX_RETRIES = 5
_DEFAULT_TOOL_TIMEOUT = 600  # seconds

logger = logging.getLogger(__name__)


class AgentConfig(NamedTuple):
    makefile_path: Path
    model: str = _DEFAULT_MODEL
    max_retries: int = _DEFAULT_MAX_RETRIES
    tool_timeout: int = _DEFAULT_TOOL_TIMEOUT


def _parse_retry_after(e: litellm.RateLimitError) -> float | None:
    """Return the wait time in seconds from a RateLimitError's response headers.

    Checks ``retry-after-ms`` (milliseconds) then ``retry-after`` (seconds).
    Returns ``None`` when neither header is present.
    """
    try:
        headers = e.response.headers if e.response is not None else {}
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
    """Call ``litellm.completion``, retrying on rate limit up to *max_retries* times.

    On each ``RateLimitError`` the wait time is read from the ``Retry-After``
    response header when present, otherwise exponential backoff is used
    (``2^attempt`` seconds, capped at 60 s).  A message is printed before
    each retry so the user can see what is happening.
    """
    for attempt in range(max_retries + 1):
        try:
            return litellm.completion(model=model, messages=messages, **tool_kwargs)
        except litellm.RateLimitError as e:
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
        self._tools = build_tools(mf)
        self._content_params = get_content_params(mf)
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
                        output = run_tool(
                            target,
                            arguments,
                            self._makefile_path,
                            self._tool_timeout,
                            content_params=self._content_params.get(target, frozenset()),
                        )
                    except Exception as e:
                        output = f"Error (unexpected): {e}"
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
                return content
