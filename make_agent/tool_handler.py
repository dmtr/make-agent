"""ToolHandler — owns tool schemas, executor map, and dispatch for a single agent session.

Tool schema builder and subprocess executor for make-based tools.

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

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any, NamedTuple

from make_agent.memory import Memory
from make_agent.parser import Makefile, Param

logger = logging.getLogger(__name__)

_VALID_MAKE_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ToolExecutionResult(NamedTuple):
    is_error: bool
    output: str


def _is_valid_make_var_name(name: str) -> bool:
    return bool(_VALID_MAKE_VAR_NAME_RE.fullmatch(name))


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


async def run_tool(
    target: str,
    arguments: dict[str, Any],
    makefile_path: Path,
    timeout: int = 600,
    max_output: int = 0,
) -> ToolExecutionResult:
    """Invoke ``make`` with safely injected parameters and return the result.

    All parameter values are injected as environment variables.  Recipes access
    them with shell syntax (``$$PARAM``).  When *max_output* is non-zero and
    the combined output exceeds that limit, the excess is dropped and a
    truncation notice is appended.

    Raises :exc:`asyncio.CancelledError` if cancelled mid-execution (the
    subprocess is killed before re-raising).
    """
    for k in arguments:
        if not _is_valid_make_var_name(k):
            return ToolHandler.get_tool_result("", f"{k!r} is not a valid make variable name", None)
        if k in os.environ:
            return ToolHandler.get_tool_result("", f"argument {k!r} shadows the system environment variable {k!r}", None)

    env = {**os.environ, **{k: str(v) for k, v in arguments.items()}}
    cmd = ["make", "--no-print-directory", "-f", str(makefile_path), target]
    logger.debug("running tool with command: %s", " ".join(cmd))
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
        )
    except OSError as e:
        logger.error("OS error when running tool %s: %s", target, e)
        return ToolHandler.get_tool_result("", f"failed to run make: {e}", None)

    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        logger.error("tool '%s' exceeded %ds timeout", target, timeout)
        return ToolHandler.get_tool_result("", f"tool '{target}' exceeded {timeout}s limit", None)
    except asyncio.CancelledError:
        proc.kill()
        await proc.wait()
        raise

    stdout = stdout_b.decode(errors="replace")
    stderr = stderr_b.decode(errors="replace")
    logger.info(
        "result of '%s': exit %s, stdout: %r, stderr: %r",
        " ".join(cmd),
        proc.returncode,
        stdout,
        stderr,
    )
    return ToolHandler.get_tool_result(stdout, stderr, proc.returncode, max_output)


class ToolHandler:
    """Owns tool schemas, executor map, and dispatch for a single agent session.

    Assembles the full list of tool schemas (built-ins + memory tools, minus any
    disabled names) and the corresponding executor callables.  Call :meth:`execute`
    to dispatch a named tool call and receive a :class:`ToolExecutionResult`.
    """

    def __init__(
        self,
        memory: Memory,
        skills_dir: str,
        disabled: frozenset[str] = frozenset(),
        tool_timeout: int = 600,
        base_dir: Path | None = None,
    ) -> None:
        from make_agent.builtin_tools import BUILTIN_SCHEMAS, get_builtin_tools, get_memory_schemas

        _base_dir = base_dir if base_dir is not None else Path.cwd()
        memory_schemas = get_memory_schemas()
        active_builtin_schemas = [s for s in BUILTIN_SCHEMAS if s["function"]["name"] not in disabled]
        active_memory_schemas = [s for s in memory_schemas if s["function"]["name"] not in disabled]
        self._schemas: list[dict] = active_builtin_schemas + active_memory_schemas
        self._executors: dict[str, Any] = get_builtin_tools(
            skills_dir, memory, disabled, tool_timeout, base_dir=_base_dir
        )

    @staticmethod
    def get_tool_result(stdout: str, stderr: str, exit_code: int | None, max_output: int = 0) -> ToolExecutionResult:
        """Build a :class:`ToolExecutionResult` from raw subprocess output.

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

    @property
    def schemas(self) -> list[dict]:
        """Tool schema list passed to the LLM as the ``tools`` parameter."""
        return self._schemas

    @property
    def tool_names(self) -> set[str]:
        """Set of tool names known to this handler."""
        return {t["function"]["name"] for t in self._schemas}

    @property
    def llm_tool_kwargs(self) -> dict:
        """Returns ``{"tools": ..., "tool_choice": "auto"}`` when tools exist, else ``{}``."""
        if self._schemas:
            return {"tools": self._schemas, "tool_choice": "auto"}
        return {}

    async def execute(
        self,
        name: str,
        arguments: dict,
        max_output: int = 0,
    ) -> ToolExecutionResult:
        """Route *name* to its executor and return a :class:`ToolExecutionResult`.

        Handles unknown tool names, argument type errors, and unexpected exceptions,
        returning an error result in each case rather than propagating.
        """
        if name not in self._executors:
            return ToolHandler.get_tool_result("", f"unknown tool: {name}", None)
        try:
            raw = self._executors[name](**arguments)
            return ToolHandler.get_tool_result(str(raw), "", 0, max_output)
        except TypeError as e:
            logger.error("argument type error when running tool %s: %s", name, e)
            return ToolHandler.get_tool_result("", f"argument type error: {e}", None)
        except Exception as e:
            logger.error("unexpected error when running tool %s: %s", name, e)
            return ToolHandler.get_tool_result("", f"unexpected error: {e}", None)
