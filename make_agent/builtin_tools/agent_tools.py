"""Agent management tools: list_agent, validate_agent, create_agent, run_agent."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, NamedTuple

from make_agent.parser import parse, parse_file, validate

_VALID_AGENT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _valid_agent_name(name: str) -> bool:
    return bool(_VALID_AGENT_NAME_RE.fullmatch(name))


def _agent_description(mk_path: Path) -> str:
    """Return the agent's description, or a fallback if absent."""
    try:
        mf = parse_file(mk_path)
    except Exception:
        return "  (could not parse)"

    if mf.description:
        return f"  {mf.description}"
    return "  (no description)"


def list_agent(agents_dir: str, current_agent: str | None = None) -> str:
    """List all available specialist agents with their descriptions."""
    path = Path(agents_dir)
    if not path.exists():
        return "No agents found (directory does not exist)"
    mk_files = sorted(p for p in path.glob("*.mk") if p.stem != current_agent)
    if not mk_files:
        return "No agents found"
    entries = [f"{mk.stem}:\n{_agent_description(mk)}" for mk in mk_files]
    return "\n\n".join(entries)


def validate_agent(name: str, agents_dir: str) -> str:
    """Validate a specialist agent's Makefile and report any errors."""
    if not _valid_agent_name(name):
        return f"Error: invalid agent name {name!r}. Use letters, numbers, hyphens, underscores, and dots only."

    mk_path = Path(agents_dir) / f"{name}.mk"
    if not mk_path.exists():
        return f"Agent '{name}' not found in {agents_dir}"

    try:
        mf = parse_file(mk_path)
    except OSError as e:
        return f"Error: could not read {mk_path}: {e}"

    errors = validate(mf)
    tool_count = sum(1 for r in mf.rules if r.params or r.description)
    if not tool_count:
        errors = ["No tools defined: at least one rule must have a # <tool> annotation block."] + errors
    if errors:
        return "Validation errors:\n" + "\n".join(f"  - {e}" for e in errors)

    return f"OK — {mk_path} ({tool_count} tool(s) valid)"


def _write_output_no_symlink(output_path: Path, content: str) -> None:
    """Write *content* to *output_path* while refusing symlink destinations."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_symlink():
        raise ValueError(f"refusing to overwrite symlink: {output_path}")
    output_path.write_text(content, encoding="utf-8")


def create_agent(name: str, makefile: str, description: str, agents_dir: str) -> str:
    """Create a new specialist agent from a raw Makefile string."""
    if not _valid_agent_name(name):
        return f"Error: invalid agent name {name!r}. Use letters, numbers, hyphens, underscores, and dots only."

    makefile = f"define DESCRIPTION\n{description}\nendef\n\n" + makefile

    try:
        mf = parse(makefile)
    except Exception as e:
        return f"Error: could not parse Makefile: {e}"

    errors = validate(mf)
    if errors:
        return "Validation errors:\n" + "\n".join(f"  - {e}" for e in errors)

    mk_path = Path(agents_dir) / f"{name}.mk"
    try:
        _write_output_no_symlink(mk_path, makefile)
    except OSError as e:
        return f"Error: could not write agent file: {e}"
    except ValueError as e:
        return f"Error: {e}"

    tool_count = sum(1 for r in mf.rules if r.description is not None)
    return f"Created agent '{name}' at {mk_path} ({tool_count} tool(s))"


class _RunAgent(NamedTuple):
    """Sentinel returned by run_agent to trigger an in-process sub-agent call."""

    mk_path: Path
    prompt: str


def run_agent(name: str, prompt: str, agents_dir: str) -> _RunAgent | str:
    """Delegate a task to a named specialist agent and return its output."""
    if not _valid_agent_name(name):
        return f"Error: invalid agent name {name!r}."

    mk_path = Path(agents_dir) / f"{name}.mk"
    if not mk_path.exists():
        return f"Agent '{name}' not found in {agents_dir}"

    return _RunAgent(mk_path=mk_path, prompt=prompt)


AGENT_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_agent",
            "description": ("List all available specialist agents in the library. " "Returns each agent name and its description."),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_agent",
            "description": (
                "Validate a specialist agent's Makefile. "
                "Checks that every declared @param is referenced in its recipe. "
                "Returns 'OK' with the tool count, or a list of validation errors."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "The agent name (without .mk extension, e.g. 'file-search'). " "Use letters, numbers, hyphens, underscores, and dots only."
                        ),
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_agent",
            "description": (
                "Create a new specialist agent by writing a Makefile to the agents library. "
                "Accepts a raw Makefile string with a define SYSTEM_PROMPT block and tool targets "
                "annotated with # <tool> comment blocks. "
                "Validates the Makefile before writing. "
                "Returns 'Created agent ...' on success or an error message."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "The agent name (without .mk extension, e.g. 'file-search'). " "Use letters, numbers, hyphens, underscores, and dots only."
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": "A short human-readable description of what this agent does. Saved as a DESCRIPTION variable in the Makefile.",
                    },
                    "makefile": {
                        "type": "string",
                        "description": (
                            "Raw Makefile content for the agent. Must include a "
                            "'define SYSTEM_PROMPT ... endef' block and one or more tool targets "
                            "preceded by '# <tool> ... # </tool>' comment blocks."
                        ),
                    },
                },
                "required": ["name", "description", "makefile"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_agent",
            "description": (
                "Delegate a task to a specialist agent and return its output. "
                "The specialist runs in-process with its own system prompt and tools; "
                "the current agent's conversation history is preserved."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The agent name (without .mk extension).",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The task or question to delegate to the specialist agent.",
                    },
                },
                "required": ["name", "prompt"],
            },
        },
    },
]
