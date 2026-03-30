"""Generate a make-agent Makefile from a structured YAML agent spec.

Usage (stdin)::

    echo '<YAML>' | make-agent-create [-o OUTPUT]

Usage (argument)::

    make-agent-create --spec '<YAML>' [-o OUTPUT]
    make-agent-create --file spec.yaml  [-o OUTPUT]

YAML spec schema::

    system_prompt: |
      You are a specialist that ...
    tools:
      - name: tool-name
        description: What this tool does.
        params:
          - name: PARAM
            type: string
            description: The param purpose
        recipe:
          - "@shell command $(value PARAM)"

``params`` may be omitted for tools that take no arguments.
``type`` must be one of: ``string``, ``number``, ``integer``, or ``boolean``.

The ``system_prompt`` is written as a ``define SYSTEM_PROMPT``/``endef`` block
so it can contain any text including ``$`` signs without escaping.

``recipe`` can be a list of shell command strings or a single multi-line
string.  Each recipe line is tab-indented in the generated Makefile.

In recipes, reference parameters as ``$(PARAM)`` (Make-expanded) or
``$(value PARAM)`` (raw literal, preserves ``$`` and special characters).
Each ``recipe`` entry becomes one tab-indented line in the Makefile target.
"""

from __future__ import annotations

import argparse
import logging
import re
import string
import sys
from pathlib import Path

import yaml

from make_agent.app_dirs import log_file

logger = logging.getLogger(__name__)

_PARAM_LINE = string.Template("# @param ${name} ${type} ${description}\n")

_TOOL_BLOCK = string.Template("# <tool>\n${description}${params}# </tool>\n${name}:\n${recipe}\n")

_MAKEFILE = string.Template("define SYSTEM_PROMPT\n${system_prompt}\nendef\n\n.PHONY: ${phony}\n\n${tools}")


def _comment_lines(text: str) -> str:
    """Prefix every line of *text* with ``# ``, blank lines become bare ``#``."""
    lines = []
    for line in text.splitlines():
        lines.append(f"# {line}" if line.strip() else "#")
    return "\n".join(lines) + "\n"


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
    raw_recipe = tool["recipe"]
    if isinstance(raw_recipe, str):
        lines = raw_recipe.splitlines()
    else:
        lines = list(raw_recipe)
    recipe = "".join(f"\t{line}\n" for line in lines)
    return _TOOL_BLOCK.substitute(
        name=tool["name"],
        description=description,
        params=params,
        recipe=recipe,
    )


def _validate_spec_params(spec: dict) -> None:
    """Raise ``ValueError`` if any tool spec declares a param not used in its recipe.

    Accepts ``$(NAME)``, ``${NAME}``, ``$$NAME``, or the ``$(NAME_FILE)`` form
    (which is always available at runtime for every parameter).
    """
    errors: list[str] = []
    for tool in spec.get("tools", []):
        name = tool.get("name", "<unnamed>")
        raw_recipe = tool.get("recipe", [])
        if isinstance(raw_recipe, str):
            recipe_text = raw_recipe
        else:
            recipe_text = "\n".join(raw_recipe)
        used = set(re.findall(r"\$\(([^)]+)\)|\$\{([^}]+)\}|\$\$(\w+)", recipe_text))
        used_flat = {g for pair in used for g in pair if g}
        for param in tool.get("params", []):
            pname = param["name"]
            file_var = f"{pname}_FILE"
            if pname not in used_flat and file_var not in used_flat:
                errors.append(
                    f"Tool '{name}': @param {pname} declared but never referenced in recipe.\n"
                    f"  Expected $({pname}), ${{{pname}}}, $${pname}, or $({file_var}) in the recipe body."
                )
    if errors:
        raise ValueError("\n".join(errors))


def _init_logging(level: int = logging.DEBUG) -> None:
    handler = logging.FileHandler(log_file())
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)


def _write_output_no_symlink(output_path: Path, content: str) -> None:
    """Write *content* to *output_path* while refusing symlink destinations."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_symlink():
        raise ValueError(f"refusing to overwrite symlink: {output_path}")
    output_path.write_text(content)


def render(spec: dict) -> str:
    """Return a Makefile string rendered from an agent *spec* dict.

    Raises ``KeyError`` if required fields are missing, ``TypeError`` if
    ``tools`` is not a list, ``ValueError`` if any tool declares a param that
    is not referenced in its recipe.
    """
    _validate_spec_params(spec)
    system_prompt: str = spec["system_prompt"]
    tools_list: list[dict] = spec["tools"]
    phony = " ".join(t["name"] for t in tools_list)
    tools = "\n".join(_render_tool(t) for t in tools_list)
    return _MAKEFILE.substitute(
        system_prompt=system_prompt,
        phony=phony,
        tools=tools,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="make-agent-create",
        description="Generate a make-agent Makefile from a YAML agent spec.",
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--spec",
        metavar="YAML",
        help="Agent spec as a YAML string",
    )
    src.add_argument(
        "--file",
        metavar="FILE",
        help="Path to a YAML file containing the agent spec",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="Write output to FILE instead of stdout",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    args = parser.parse_args()

    _init_logging(getattr(logging, args.log_level))

    logger.info("Starting make-agent-create...")

    if args.spec:
        raw = args.spec
    elif args.file:
        raw = Path(args.file).read_text()
    else:
        raw = sys.stdin.read()

    try:
        spec = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        logger.error("Failed to parse YAML: %s", e)
        sys.exit(f"make-agent-create: invalid YAML: {e}")

    try:
        makefile = render(spec)
    except (KeyError, TypeError) as e:
        logger.error("Invalid spec: %s", e)
        sys.exit(f"make-agent-create: invalid spec: {e}")
    except ValueError as e:
        logger.error("Spec validation error: %s", e)
        sys.exit(f"make-agent-create: {e}")

    if args.output:
        try:
            _write_output_no_symlink(Path(args.output), makefile)
        except OSError as e:
            logger.error("Failed to write output file: %s", e)
            sys.exit(f"make-agent-create: failed to write output file: {e}")
        except ValueError as e:
            logger.error("Unsafe output path: %s", e)
            sys.exit(f"make-agent-create: {e}")
    else:
        sys.stdout.write(makefile)

    logger.info("make-agent-create completed successfully.")


if __name__ == "__main__":
    main()
