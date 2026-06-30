"""Tool schema builders and execution result type."""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

from make_agent.parser import Makefile, Param

logger = logging.getLogger(__name__)


class ToolExecutionResult(NamedTuple):
    is_error: bool
    output: str


def _param_schema(p: Param) -> dict[str, str]:
    """Return the JSON Schema fragment for a single tool parameter."""
    json_type = (
        p.type if p.type in ("string", "number", "integer", "boolean") else "string"
    )
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


def get_tool_result(
    stdout: str, stderr: str, exit_code: int | None, max_output: int = 0
) -> ToolExecutionResult:
    """Build a :class:`ToolExecutionResult` from raw subprocess output.

    *max_output* limits how many characters of the final combined output are kept.
    When the combined output exceeds that limit, the excess is dropped and a
    truncation notice is included within the limit.  ``0`` means no limit.
    """
    result = []
    is_error = (exit_code != 0 if exit_code is not None else True) or bool(
        stderr.strip()
    )
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
