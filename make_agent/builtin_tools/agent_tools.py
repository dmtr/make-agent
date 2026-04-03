"""Agent management tools: list_agent, validate_agent, create_agent, load_agent."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, NamedTuple

import yaml

from make_agent.create_agent import _write_output_no_symlink, render
from make_agent.parser import parse_file, validate

_VALID_AGENT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _valid_agent_name(name: str) -> bool:
    return bool(_VALID_AGENT_NAME_RE.fullmatch(name))


def _agent_summary(mk_path: Path) -> str:
    """Return a multi-line summary of an agent: system prompt + tool list."""
    try:
        mf = parse_file(mk_path)
    except Exception:
        return "  (could not parse)"

    lines: list[str] = []

    if mf.system_prompt:
        # First non-empty line of the system prompt as the headline.
        for line in mf.system_prompt.splitlines():
            if line.strip():
                lines.append(f"  {line.strip()}")
                break
    else:
        lines.append("  (no description)")

    tools = [r for r in mf.rules if r.description is not None]
    if tools:
        lines.append("  tools:")
        for rule in tools:
            desc = rule.description.splitlines()[0].strip() if rule.description else ""
            params = ", ".join(p.name for p in rule.params)
            param_str = f"({params})" if params else "()"
            lines.append(f"    - {rule.target}{param_str}: {desc}")

    return "\n".join(lines)


def list_agent(agents_dir: str) -> str:
    """List all available specialist agents with their system prompt and tools."""
    path = Path(agents_dir)
    if not path.exists():
        return "No agents found (directory does not exist)"
    mk_files = sorted(path.glob("*.mk"))
    if not mk_files:
        return "No agents found"
    entries = [f"{mk.stem}:\n{_agent_summary(mk)}" for mk in mk_files]
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
    if errors:
        return "Validation errors:\n" + "\n".join(f"  - {e}" for e in errors)

    tool_count = sum(1 for r in mf.rules if r.params or r.description)
    return f"OK — {mk_path} ({tool_count} tool(s) valid)"


def create_agent(name: str, spec: str, agents_dir: str) -> str:
    """Create a new specialist agent Makefile from a YAML spec string."""
    if not _valid_agent_name(name):
        return f"Error: invalid agent name {name!r}. Use letters, numbers, hyphens, underscores, and dots only."

    try:
        parsed_spec = yaml.safe_load(spec)
    except yaml.YAMLError as e:
        return f"Error: invalid YAML spec: {e}"

    try:
        makefile_content = render(parsed_spec)
    except KeyError as e:
        return f"Error: missing required field in spec: {e}"
    except TypeError as e:
        return f"Error: invalid spec structure: {e}"
    except ValueError as e:
        return f"Error: {e}"

    mk_path = Path(agents_dir) / f"{name}.mk"
    try:
        _write_output_no_symlink(mk_path, makefile_content)
    except OSError as e:
        return f"Error: could not write agent file: {e}"
    except ValueError as e:
        return f"Error: {e}"

    tool_count = sum(1 for t in parsed_spec.get("tools", []))
    return f"Created agent '{name}' at {mk_path} ({tool_count} tool(s))"


class _SwapAgent(NamedTuple):
    """Sentinel returned by load_agent to trigger an in-place agent swap."""

    mk_path: Path
    prompt: str


def load_agent(name: str, prompt: str, agents_dir: str) -> _SwapAgent | str:
    """Replace the current agent with the named specialist and process the given prompt."""
    if not _valid_agent_name(name):
        return f"Error: invalid agent name {name!r}."

    mk_path = Path(agents_dir) / f"{name}.mk"
    if not mk_path.exists():
        return f"Agent '{name}' not found in {agents_dir}"

    return _SwapAgent(mk_path=mk_path, prompt=prompt)


AGENT_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_agent",
            "description": ("List all available specialist agents in the library. " "Returns each agent name and a short description of its purpose."),
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
                "Accepts a YAML spec with a system_prompt and a list of tools (each with a "
                "name, description, optional params, and recipe). "
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
                    "spec": {
                        "type": "string",
                        "description": (
                            "YAML string defining the agent. Required fields: "
                            "'system_prompt' (string) and 'tools' (list). "
                            "Each tool needs 'name', 'description', 'recipe' (list of shell commands), "
                            "and optional 'params' (list of {name, type, description})."
                        ),
                    },
                },
                "required": ["name", "spec"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_agent",
            "description": (
                "Replace the current agent with a specialist and immediately process a prompt with it. "
                "The specialist's system prompt and tools become active; conversation history is reset."
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
                        "description": "The task or question to process with the specialist agent.",
                    },
                },
                "required": ["name", "prompt"],
            },
        },
    },
]
