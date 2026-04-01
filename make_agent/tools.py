"""Tool schema builder and executor for the make-agent.

Converts parsed Makefile rules into tool definitions and
executes them by invoking ``make`` with parameters injected via the subprocess
environment.

Parameter injection
-------------------
Every parameter value is set as an environment variable for the subprocess.
Recipes access it with shell syntax (``$$PARAM``)::

    greet:
        @echo "Hello, $$NAME!"

    write-file:
        @printf '%s' "$$CONTENT" > output.txt

This works for both single-line and multiline values — the OS passes env vars
to the recipe shell intact regardless of newlines.  Make also auto-imports
environment variables as Make variables, so ``$(PARAM)`` continues to work
for simple values where the Makefile does not define its own ``PARAM``.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, NamedTuple

from make_agent.parser import Makefile, Param

logger = logging.getLogger(__name__)


_VALID_MAKE_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ToolExecutionResult(NamedTuple):
    is_error: bool
    output: str


def _is_valid_make_var_name(name: str) -> bool:
    return bool(_VALID_MAKE_VAR_NAME_RE.fullmatch(name))


def get_tool_result(stdout: str, stderr: str, exit_code: int | None, max_output: int = 0) -> ToolExecutionResult:
    """
    *max_output* limits how many characters of the final combined output are kept.
    When the combined output exceeds that limit, the excess is dropped and a
    truncation notice is included within the limit.  ``0`` means no limit.
    """
    result = []
    is_error = (exit_code != 0 if exit_code is not None else True) or bool(stderr.strip())
    is_stdout_empty = stdout.strip() == ""

    if is_error:
        stdout_stripped = stdout.strip()
        stderr_stripped = stderr.strip()
        if stdout_stripped:
            result.append(stdout_stripped)
        if stderr_stripped:
            result.append("ERROR: ")
            result.append(stderr_stripped)
        else:
            result.append("ERROR: unknown error")
    else:
        result.append(stdout.strip())

    if not is_error and is_stdout_empty:
        result.append("OK. Execution succeeded with no output.")

    final_result = "\n".join(result)

    if max_output > 0 and len(final_result) > max_output:
        omitted = len(final_result) - max_output
        notice = f"(Output was truncated, {omitted} omitted_chars)"
        notice_len = len(notice)
        if notice_len >= max_output:
            final_result = notice[:max_output]
        else:
            available = max_output - notice_len
            final_result = final_result[:available] + notice

    return ToolExecutionResult(is_error=is_error, output=final_result)


def _param_schema(p: Param) -> dict[str, str]:
    """Return the JSON Schema fragment for a single tool parameter."""
    json_type = p.type if p.type in ("string", "number", "integer", "boolean") else "string"
    return {"type": json_type, "description": p.description}


def build_tools(makefile: Makefile) -> list[dict[str, Any]]:
    """Return a list of OpenAI function-tool dicts for every rule that has a
    ``# <tool>`` description block."""
    tools = []
    for rule in makefile.rules:
        if rule.description is None:
            continue
        properties = {p.name: _param_schema(p) for p in rule.params}
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": rule.target,
                    "description": rule.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": [p.name for p in rule.params],
                    },
                },
            }
        )
    return tools


def run_tool(
    target: str,
    arguments: dict[str, Any],
    makefile_path: Path,
    timeout: int = 600,
    max_output: int = 0,
) -> ToolExecutionResult:
    """Invoke ``make`` with safely injected parameters and return output as JSON.

    Returns a JSON string with keys ``stdout``, ``stderr``, and ``exit_code``
    (``null`` for framework-level errors such as timeout or OS failure).  When
    *max_output* is non-zero and the stdout exceeds that limit, the output is
    truncated and an ``omitted_chars`` key is added.

    All parameter values are injected as environment variables.  Recipes access
    them with shell syntax (``$$PARAM``).
    """
    for k in arguments:
        if not _is_valid_make_var_name(k):
            return get_tool_result("", f"{k!r} is not a valid make variable name", None)
        if k in os.environ:
            return get_tool_result("", f"argument {k!r} shadows the system environment variable {k!r}", None)

    env = {**os.environ, **{k: str(v) for k, v in arguments.items()}}
    cmd = ["make", "--no-print-directory", "-f", str(makefile_path), target]
    logger.debug(f"running tool with command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=timeout)
        logger.info(f"result of '{' '.join(cmd)}': exit {result.returncode}, stdout: {result.stdout!r}, stderr: {result.stderr!r}")
    except subprocess.TimeoutExpired:
        logger.error("tool '%s' exceeded %ds timeout", target, timeout)
        return get_tool_result("", f"tool '{target}' exceeded {timeout}s limit", None)
    except OSError as e:
        logger.error("OS error when running tool %s %s", target, e)
        return get_tool_result("", f"failed to run make: {e}", None)
    return get_tool_result(result.stdout, result.stderr, result.returncode, max_output)
