"""Generate a make-agent Makefile from a structured JSON agent spec.

Usage (stdin)::

    echo '<JSON>' | make-agent-create [-o OUTPUT]

Usage (argument)::

    make-agent-create --spec '<JSON>' [-o OUTPUT]
    make-agent-create --file spec.json  [-o OUTPUT]

JSON spec schema::

    {
      "system_prompt": "You are a specialist that ...",
      "tools": [
        {
          "name": "tool-name",
          "description": "What this tool does.",
          "params": [
            {"name": "PARAM", "type": "string", "description": "The param purpose"}
          ],
          "recipe": [
            "@shell command $(PARAM)"
          ]
        }
      ]
    }

``params`` may be omitted for tools that take no arguments.
``type`` must be a JSON Schema primitive: ``string``, ``number``, ``integer``,
or ``boolean``.
Each ``recipe`` entry becomes one tab-indented line in the Makefile target.
"""

from __future__ import annotations

import argparse
import json
import string
import sys
from pathlib import Path

_SYSTEM_BLOCK = string.Template("# <system>\n${body}# </system>\n")

_PARAM_LINE = string.Template("# @param ${name} ${type} ${description}\n")

_TOOL_BLOCK = string.Template("# <tool>\n${description}${params}# </tool>\n${name}:\n${recipe}\n")

_MAKEFILE = string.Template("${system_block}\n.PHONY: ${phony}\n\n${tools}")


def _comment_lines(text: str) -> str:
    """Prefix every line of *text* with ``# ``, blank lines become bare ``#``."""
    lines = []
    for line in text.splitlines():
        lines.append(f"# {line}" if line.strip() else "#")
    return "\n".join(lines) + "\n"


def _render_system_block(system_prompt: str) -> str:
    return _SYSTEM_BLOCK.substitute(body=_comment_lines(system_prompt))


def _render_tool(tool: dict) -> str:
    description = _comment_lines(tool["description"])
    params = "".join(
        _PARAM_LINE.substitute(
            name=p["name"],
            type=p["type"],
            description=p["description"],
        )
        for p in tool.get("params", [])
    )
    recipe = "".join(f"\t{line}\n" for line in tool["recipe"])
    return _TOOL_BLOCK.substitute(
        name=tool["name"],
        description=description,
        params=params,
        recipe=recipe,
    )


def render(spec: dict) -> str:
    """Return a Makefile string rendered from an agent *spec* dict.

    Raises ``KeyError`` if required fields are missing, ``TypeError`` if
    ``tools`` is not a list.
    """
    system_block = _render_system_block(spec["system_prompt"])
    tools_list: list[dict] = spec["tools"]
    phony = " ".join(t["name"] for t in tools_list)
    tools = "\n".join(_render_tool(t) for t in tools_list)
    return _MAKEFILE.substitute(
        system_block=system_block,
        phony=phony,
        tools=tools,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="make-agent-create",
        description="Generate a make-agent Makefile from a JSON agent spec.",
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--spec",
        metavar="JSON",
        help="Agent spec as a JSON string",
    )
    src.add_argument(
        "--file",
        metavar="FILE",
        help="Path to a JSON file containing the agent spec",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="Write output to FILE instead of stdout",
    )
    args = parser.parse_args()

    if args.spec:
        raw = args.spec
    elif args.file:
        raw = Path(args.file).read_text()
    else:
        raw = sys.stdin.read()

    try:
        spec = json.loads(raw)
    except json.JSONDecodeError as e:
        sys.exit(f"make-agent-create: invalid JSON: {e}")

    try:
        makefile = render(spec)
    except (KeyError, TypeError) as e:
        sys.exit(f"make-agent-create: invalid spec: {e}")

    if args.output:
        Path(args.output).write_text(makefile)
    else:
        sys.stdout.write(makefile)


if __name__ == "__main__":
    main()
