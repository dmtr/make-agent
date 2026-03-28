"""Built-in agent management tools always available to every agent.

These three tools are injected into every agent's tool schema alongside any
Makefile-defined tools, without requiring a Makefile declaration.

- ``list_agent``          — discover available specialist agents
- ``validate_agent``      — validate a specialist agent's Makefile
- ``run_agent``           — delegate a task to a specialist agent via subprocess
- ``search_user_memory``  — FTS5 search over past user messages (when memory enabled)
- ``search_agent_memory`` — FTS5 search over past agent replies (when memory enabled)
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from make_agent.create_agent import render, _write_output_no_symlink, _validate_spec_params
from make_agent.parser import parse_file, validate

_VALID_AGENT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_MEMORY_SEARCH_PARAMS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "FTS5 match expression (e.g. 'file search', 'make OR agent').",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of results to return (default: 10).",
        },
        "from_date": {
            "type": "string",
            "description": "ISO 8601 date string to filter results on or after (e.g. '2026-03-01').",
        },
        "to_date": {
            "type": "string",
            "description": "ISO 8601 date string to filter results on or before (e.g. '2026-03-31').",
        },
    },
    "required": ["query"],
}


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


def run_agent(name: str, prompt: str, agents_dir: str, model: str, debug: bool = False) -> str:
    """Run a specialist agent as a subprocess and return its output."""
    if not _valid_agent_name(name):
        return f"Error: invalid agent name {name!r}."

    mk_path = Path(agents_dir) / f"{name}.mk"
    if not mk_path.exists():
        return f"Agent '{name}' not found in {agents_dir}"

    cmd = [
        sys.executable, "-m", "make_agent.main",
        "-f", str(mk_path),
        "--prompt", prompt,
        "--model", model,
        "--agents-dir", agents_dir,
    ]
    if debug:
        cmd.append("--debug")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as e:
        return f"Error: failed to run agent: {e}"

    if result.returncode != 0:
        parts = [p for p in [result.stdout.strip(), result.stderr.strip()] if p]
        body = "\n".join(parts)
        return f"Error (exit {result.returncode}):\n{body}" if body else f"Error (exit {result.returncode})"

    return result.stdout


# ── OpenAI tool schemas ────────────────────────────────────────────────────────

BUILTIN_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_agent",
            "description": (
                "List all available specialist agents in the library. "
                "Returns each agent name and a short description of its purpose."
            ),
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
                            "The agent name (without .mk extension, e.g. 'file-search'). "
                            "Use letters, numbers, hyphens, underscores, and dots only."
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
                            "The agent name (without .mk extension, e.g. 'file-search'). "
                            "Use letters, numbers, hyphens, underscores, and dots only."
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
            "name": "run_agent",
            "description": (
                "Run a specialist agent with a single task prompt and return its output. "
                "The agent will use its own tools to complete the task."
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
                        "description": "The task or question to send to the agent.",
                    },
                },
                "required": ["name", "prompt"],
            },
        },
    },
]


def get_builtin_tools(agents_dir: str, model: str, debug: bool = False, memory: Any = None) -> dict[str, Any]:
    """Return a name → callable mapping for all built-in tools.

    Each callable accepts only the LLM-provided arguments; ``agents_dir``,
    ``model``, and ``memory`` are pre-bound via closure.
    """
    tools: dict[str, Any] = {
        "list_agent": lambda **_kw: list_agent(agents_dir),
        "validate_agent": lambda name, **_kw: validate_agent(name, agents_dir),
        "create_agent": lambda name, spec, **_kw: create_agent(name, spec, agents_dir),
        "run_agent": lambda name, prompt, **_kw: run_agent(name, prompt, agents_dir, model, debug),
    }
    if memory is not None:
        tools["search_user_memory"] = lambda query, limit=10, from_date=None, to_date=None, **_kw: memory.search_user(query, limit, from_date, to_date)
        tools["search_agent_memory"] = lambda query, limit=10, from_date=None, to_date=None, **_kw: memory.search_agent(query, limit, from_date, to_date)
    return tools


def get_memory_schemas() -> list[dict[str, Any]]:
    """Return the tool schemas for memory search tools.

    These are only injected when memory is enabled.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "search_user_memory",
                "description": (
                    "Search past user messages stored in memory using full-text search (FTS5). "
                    "Returns matching messages with timestamps, ordered by relevance."
                ),
                "parameters": _MEMORY_SEARCH_PARAMS,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_agent_memory",
                "description": (
                    "Search past agent replies stored in memory using full-text search (FTS5). "
                    "Returns matching messages with timestamps, ordered by relevance."
                ),
                "parameters": _MEMORY_SEARCH_PARAMS,
            },
        },
    ]
