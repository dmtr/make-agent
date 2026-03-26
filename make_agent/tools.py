"""Tool schema builder and executor for the make-agent.

Converts parsed Makefile rules into OpenAI-compatible tool definitions and
executes them by invoking ``make -f <makefile> <target> KEY=value …``.

The ``content`` param type
--------------------------
Parameters declared with ``@param NAME content …`` carry arbitrary multi-line
text (file contents, scripts, etc.).  They are handled specially by
:func:`run_tool`: the value is written to a temporary file and the Make
variable ``{NAME}_FILE`` is set to that path instead of passing the raw value
on the command line.  Recipes must therefore reference ``$(NAME_FILE)``::

    # @param CONTENT content  Text to write
    write-file:
        @cat "$(CONTENT_FILE)" > "$(FILE)"
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from make_agent.parser import Makefile, Param

logger = logging.getLogger(__name__)


_CONTENT_TYPE = "content"
_VALID_MAKE_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _is_valid_make_var_name(name: str) -> bool:
    return bool(_VALID_MAKE_VAR_NAME_RE.fullmatch(name))


def _escape_make_assignment_value(value: Any) -> str:
    s = str(value)
    # Keep values literal when they are later interpolated by make and shell:
    # - \  => \\   (preserve backslashes)
    # - "  => \"   (avoid closing shell double-quoted strings)
    # - `  => \`   (avoid backtick command substitution)
    # - $  => \$$  (avoid make/shell expansion such as $(...) and $VAR)
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`").replace("$", "\\$$")


def _param_schema(p: Param) -> dict[str, str]:
    """Return the JSON Schema fragment for a single tool parameter.

    ``content``-typed params are advertised as plain strings with an extra hint
    so the LLM knows it can pass full multi-line text without escaping.
    """
    if p.type == _CONTENT_TYPE:
        return {
            "type": "string",
            "description": p.description + " Pass the full text as-is — the framework handles special characters safely.",
        }
    return {"type": p.type, "description": p.description}


def get_content_params(makefile: Makefile) -> dict[str, frozenset[str]]:
    """Return ``{target: frozenset_of_content_param_names}`` for every rule that
    has at least one ``content``-typed parameter."""
    result: dict[str, frozenset[str]] = {}
    for rule in makefile.rules:
        names = frozenset(p.name for p in rule.params if p.type == _CONTENT_TYPE)
        if names:
            result[rule.target] = names
    return result


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
    arguments: dict[str, str],
    makefile_path: Path,
    timeout: int = 600,
    content_params: frozenset[str] = frozenset(),
) -> str:
    """Invoke ``make -f <makefile_path> <target> KEY=val …`` and return output.

    Parameters listed in *content_params* are written to temporary files and
    passed as ``{KEY}_FILE=/tmp/…`` instead of inline ``KEY=value``.  This
    avoids shell-expansion errors when the value contains newlines, quotes, or
    other special characters.  Temporary files are always removed after the
    subprocess exits.

    On a non-zero exit code an error message combining stdout and stderr is
    returned so the LLM receives all available context without losing partial
    output produced before the failure.

    *timeout* is the maximum number of seconds to wait for the subprocess.
    If the process exceeds this limit it is killed and an error is returned.
    """
    tmp_files: list[Path] = []
    try:
        make_vars: list[str] = []
        for k, v in arguments.items():
            if not _is_valid_make_var_name(k):
                return f"Error (invalid argument name): {k!r} is not a valid make variable name"
            if k in content_params:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    prefix=f"make-agent-{k}-",
                    suffix=".tmp",
                    delete=False,
                ) as tf:
                    tf.write(v)
                    tmp_path = Path(tf.name)
                tmp_files.append(tmp_path)
                make_vars.append(f"{k}_FILE={tmp_path}")
            else:
                make_vars.append(f"{k}={_escape_make_assignment_value(v)}")

        cmd = ["make", "--no-print-directory", "-f", str(makefile_path), target] + make_vars
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
    finally:
        for tmp in tmp_files:
            try:
                tmp.unlink()
            except OSError:
                pass
