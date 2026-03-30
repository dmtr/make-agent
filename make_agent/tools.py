"""Tool schema builder and executor for the make-agent.

Converts parsed Makefile rules into OpenAI-compatible tool definitions and
executes them by injecting parameters safely before invoking ``make``.

Parameter injection
-------------------
For each tool call, values are injected in two complementary ways:

1. **Temp file** (``PARAM_FILE``): Every parameter value is written to a
   temporary file.  Make receives ``PARAM_FILE=/tmp/...`` so recipes can
   read arbitrary content::

       write-file:
           @cat "$(PARAM_FILE)" > "$(DEST)"

2. **params.mk** (``PARAM``): For single-line values, a temporary
   ``params.mk`` is also written with ``PARAM = <value>`` (``$`` signs
   escaped as ``$$``) and loaded with ``-f params.mk``.  Recipes that
   reference ``$(PARAM)`` directly continue to work for simple values::

       greet:
           @echo "Hello, $(NAME)!"

The ``_FILE`` suffix form works for **all** values including multiline
text; the bare form is a convenience for the common single-line case.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from make_agent.parser import Makefile, Param

logger = logging.getLogger(__name__)


_VALID_MAKE_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _is_valid_make_var_name(name: str) -> bool:
    return bool(_VALID_MAKE_VAR_NAME_RE.fullmatch(name))


def _escape_make_value(value: str) -> str:
    """Escape *value* for use on the right-hand side of a Make ``=`` assignment.

    Only ``$`` needs escaping: ``$`` → ``$$``.  Everything else (quotes,
    backslashes, spaces) is safe in a Make variable value.
    """
    return value.replace("$", "$$")


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

    Every argument value is written to a temporary file; Make receives
    ``PARAM_FILE=/tmp/...`` for each parameter.  For single-line values
    that contain no newlines, a temporary ``params.mk`` is also written
    with ``PARAM = <escaped-value>`` so recipes can use ``$(PARAM)``
    directly without any quoting ceremony.

    All temporary files are always removed after the subprocess exits.
    """
    tmp_files: list[Path] = []
    try:
        for k in arguments:
            if not _is_valid_make_var_name(k):
                return format_tool_result("", f"{k!r} is not a valid make variable name", None)

        params_mk_lines: list[str] = []
        make_vars: list[str] = []

        for k, v in arguments.items():
            # Normalise JSON primitives (int, float, bool) to str.
            v_str = str(v)
            # Always write a temp file for $(PARAM_FILE) access.
            with tempfile.NamedTemporaryFile(
                mode="w",
                prefix=f"make-agent-{k}-",
                suffix=".tmp",
                delete=False,
            ) as tf:
                tf.write(v_str)
                file_path = Path(tf.name)
            tmp_files.append(file_path)
            make_vars.append(f"{k}_FILE={file_path}")

            # Also provide $(PARAM) for single-line values via params.mk.
            if "\n" not in v_str:
                params_mk_lines.append(f"{k} = {_escape_make_value(v_str)}")

        cmd: list[str] = ["make", "--no-print-directory"]

        if params_mk_lines:
            params_content = "\n".join(params_mk_lines) + "\n"
            with tempfile.NamedTemporaryFile(
                mode="w",
                prefix="make-agent-params-",
                suffix=".mk",
                delete=False,
            ) as tf:
                tf.write(params_content)
                params_mk = Path(tf.name)
            tmp_files.append(params_mk)
            cmd += ["-f", str(params_mk)]

        cmd += ["-f", str(makefile_path), target] + make_vars

        logger.debug(f"running tool with command: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=timeout)
            logger.debug(f"result of '{' '.join(cmd)}': exit {result.returncode}, stdout: {result.stdout!r}, stderr: {result.stderr!r}")
        except subprocess.TimeoutExpired:
            return format_tool_result("", f"tool '{target}' exceeded {timeout}s limit", None)
        except OSError as e:
            return format_tool_result("", f"failed to run make: {e}", None)
        return format_tool_result(result.stdout, result.stderr, result.returncode, max_output)
    finally:
        for tmp in tmp_files:
            try:
                tmp.unlink()
            except OSError:
                pass
