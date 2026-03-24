"""Tool schema builder and executor for the make-agent.

Converts parsed Makefile rules into OpenAI-compatible tool definitions and
executes them by invoking ``make -f <makefile> <target> KEY=value …``.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

from make_agent.parser import Makefile

logger = logging.getLogger(__name__)


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
    timeout: int = 600,
) -> str:
    """Invoke ``make -f <makefile_path> <target> KEY=val …`` and return output.

    On a non-zero exit code an error message combining stdout and stderr is
    returned so the LLM receives all available context without losing partial
    output produced before the failure.

    *timeout* is the maximum number of seconds to wait for the subprocess.
    If the process exceeds this limit it is killed and an error is returned.
    """
    # Escape $ in argument values so Make does not expand $(VAR) references
    # inside user-supplied data (e.g. SPEC strings containing Makefile syntax).
    cmd = ["make", "--no-print-directory", "-f", str(makefile_path), target] + [f"{k}={v.replace('$', '$$')}" for k, v in arguments.items()]
    logger.debug(f"running tool with command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        logger.debug(f"result of '{' '.join(cmd)}': exit {result.returncode}, stdout: {result.stdout!r}, stderr: {result.stderr!r}")
    except subprocess.TimeoutExpired:
        return f"Error (timeout): tool '{target}' exceeded {timeout}s limit"
    except OSError as e:
        return f"Error (failed to run make): {e}"
    if result.returncode != 0:
        parts = [p for p in [result.stdout.strip(), result.stderr.strip()] if p]
        body = "\n".join(parts)
        return f"Error (exit {result.returncode}):\n{body}" if body else f"Error (exit {result.returncode})"
    return result.stdout
