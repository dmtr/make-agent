"""Built-in agent management tools always available to every agent.

These three tools are injected into every agent's tool schema alongside any
Makefile-defined tools, without requiring a Makefile declaration.

- ``list_agent``   — discover available specialist agents
- ``create_agent`` — create or overwrite a specialist agent from a YAML spec
- ``run_agent``    — delegate a task to a specialist agent via subprocess
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from make_agent.create_agent import _write_output_no_symlink, render

_VALID_AGENT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _valid_agent_name(name: str) -> bool:
    return bool(_VALID_AGENT_NAME_RE.fullmatch(name))


def _extract_system_prompt_preview(mk_path: Path) -> str:
    """Return the first non-empty line of the SYSTEM_PROMPT define block."""
    in_block = False
    for line in mk_path.read_text().splitlines():
        stripped = line.strip()
        if stripped == "define SYSTEM_PROMPT":
            in_block = True
            continue
        if in_block:
            if stripped == "endef":
                break
            if stripped:
                return stripped
    return "(no description)"


def list_agent(agents_dir: str) -> str:
    """List all available specialist agents, one per line as ``name: description``."""
    path = Path(agents_dir)
    if not path.exists():
        return "No agents found (directory does not exist)"
    mk_files = sorted(path.glob("*.mk"))
    if not mk_files:
        return "No agents found"
    return "\n".join(f"{mk.stem}: {_extract_system_prompt_preview(mk)}" for mk in mk_files)


def create_agent(name: str, spec: str, agents_dir: str) -> str:
    """Create or overwrite a specialist agent from a YAML spec string."""
    if not _valid_agent_name(name):
        return f"Error: invalid agent name {name!r}. Use letters, numbers, hyphens, underscores, and dots only."

    try:
        spec_dict = yaml.safe_load(spec)
    except yaml.YAMLError as e:
        return f"Error: invalid YAML spec: {e}"

    try:
        makefile_content = render(spec_dict)
    except (KeyError, TypeError) as e:
        return f"Error: invalid spec structure: {e}"
    except ValueError as e:
        return f"Error: {e}"

    output_path = Path(agents_dir) / f"{name}.mk"
    try:
        _write_output_no_symlink(output_path, makefile_content)
    except (OSError, ValueError) as e:
        return f"Error: failed to write agent file: {e}"

    return f"Created {output_path}"


def run_agent(name: str, prompt: str, agents_dir: str, model: str) -> str:
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
            "name": "create_agent",
            "description": (
                "Create or overwrite a specialist agent in the library. "
                "Pass a YAML spec defining the agent's system_prompt and tools. "
                "The agent is immediately available for use with run_agent after creation."
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
                            "YAML agent spec. Schema:\n"
                            "  system_prompt: |\n"
                            "    You are a specialist that ...\n"
                            "  tools:\n"
                            "    - name: tool-name\n"
                            "      description: What this tool does.\n"
                            "      params:\n"
                            "        - name: PARAM\n"
                            "          type: string\n"
                            "          description: The param purpose\n"
                            "      recipe:\n"
                            "        - '@shell command $(PARAM)'\n"
                            "Every param MUST appear as $(PARAM) or $(PARAM_FILE) in the recipe."
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


def get_builtin_tools(agents_dir: str, model: str) -> dict[str, Any]:
    """Return a name → callable mapping for all built-in tools.

    Each callable accepts only the LLM-provided arguments; ``agents_dir``
    and ``model`` are pre-bound via closure.
    """
    return {
        "list_agent": lambda **_kw: list_agent(agents_dir),
        "create_agent": lambda name, spec, **_kw: create_agent(name, spec, agents_dir),
        "run_agent": lambda name, prompt, **_kw: run_agent(name, prompt, agents_dir, model),
    }
