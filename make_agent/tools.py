"""Tool schema builder and executor for the make-agent.

Converts parsed Makefile rules into OpenAI-compatible tool definitions and
executes them by invoking ``make -f <makefile> <target> KEY=value …``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from make_agent.parser import Makefile


def build_tools(makefile: Makefile) -> list[dict[str, Any]]:
    """Return a list of OpenAI function-tool dicts for every rule that has a
    ``# <tool>`` description block."""
    tools = []
    for rule in makefile.rules:
        if rule.description is None:
            continue
        properties = {p.name: {"type": p.type, "description": p.description} for p in rule.params}
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
    arguments: dict[str, str],
    makefile_path: Path,
) -> str:
    """Invoke ``make -f <makefile_path> <target> KEY=val …`` and return stdout.

    On a non-zero exit code the error message (with stderr) is returned instead
    so the LLM can reason about failures.
    """
    cmd = ["make", "--no-print-directory", "-f", str(makefile_path), target] + [f"{k}={v}" for k, v in arguments.items()]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        return f"Error (exit {result.returncode}):\n{stderr}" if stderr else f"Error (exit {result.returncode})"
    return result.stdout
