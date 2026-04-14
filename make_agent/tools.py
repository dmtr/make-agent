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

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from make_agent.parser import Makefile, Param

logger = logging.getLogger(__name__)


_VALID_MAKE_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _is_valid_make_var_name(name: str) -> bool:
    return bool(_VALID_MAKE_VAR_NAME_RE.fullmatch(name))


def format_tool_result(stdout: str, stderr: str, exit_code: int | None, max_output: int = 0) -> str:
    """Serialise tool output as a JSON string for the LLM.

    *max_output* limits how many characters of *stdout* and *stderr* are kept
    (each stream capped independently).  When a stream is longer, the excess is
    dropped and an ``omitted_chars`` key is added so the LLM knows it received
    a partial result.  ``0`` means no limit.
    """
    omitted = 0
    if max_output > 0 and len(stdout) > max_output:
        omitted += len(stdout) - max_output
        stdout = stdout[:max_output]
    if max_output > 0 and len(stderr) > max_output:
        omitted += len(stderr) - max_output
        stderr = stderr[:max_output]
    result: dict[str, Any] = {"stdout": stdout, "stderr": stderr, "exit_code": exit_code}
    if omitted:
        result["omitted_chars"] = omitted
    return json.dumps(result)


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
) -> str:
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
            return format_tool_result("", f"{k!r} is not a valid make variable name", None)

    env = {**os.environ, **{k: str(v) for k, v in arguments.items()}}
    cmd = ["make", "--no-print-directory", "-f", str(makefile_path), target]
    logger.debug(f"running tool with command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=timeout)
        logger.info(f"result of '{' '.join(cmd)}': exit {result.returncode}, stdout: {result.stdout!r}, stderr: {result.stderr!r}")
    except subprocess.TimeoutExpired:
        logger.error("tool '%s' exceeded %ds timeout", target, timeout)
        return format_tool_result("", f"tool '{target}' exceeded {timeout}s limit", None)
    except OSError as e:
        logger.error("OS error when running tool %s %s", target, e)
        return format_tool_result("", f"failed to run make: {e}", None)
    return format_tool_result(result.stdout, result.stderr, result.returncode, max_output)
