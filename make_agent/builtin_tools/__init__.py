"""Built-in tools always available to every agent.

These tools are injected into every agent's tool schema, without requiring
any Makefile or external definition.

Sub-modules:
- ``skill_tools``  — list/read/execute/create/validate skills
- ``file_tools``   — write_file, edit_file (sandboxed to the working directory)

Memory tools (search_user_memory, search_agent_memory, get_recent_messages) are
owned by ``make_agent.memory`` and registered separately via ``get_memory_executors``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from make_agent.builtin_tools.file_tools import FILE_SCHEMAS, edit_file, write_file
from make_agent.builtin_tools.skill_tools import (
    SKILL_SCHEMAS,
    _valid_skill_name,
    create_skill,
    execute_skill,
    list_skills,
    read_skill,
    validate_skill,
)

BUILTIN_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "list_skills",
        "read_skill",
        "execute_skill",
        "create_skill",
        "validate_skill",
        "write_file",
        "edit_file",
        "search_user_memory",
        "search_agent_memory",
        "get_recent_messages",
    }
)

BUILTIN_SCHEMAS: list[dict[str, Any]] = SKILL_SCHEMAS + FILE_SCHEMAS


def get_builtin_tools(
    skills_dir: str,
    disabled: frozenset[str] = frozenset(),
    tool_timeout: int = 600,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Return a name → callable mapping for non-memory built-in tools.

    Each callable accepts only the LLM-provided arguments; ``skills_dir`` and
    ``base_dir`` are pre-bound via closure.  Tools whose names appear in
    *disabled* are omitted.

    *base_dir* is the sandbox root for file tools.  Defaults to
    ``Path.cwd()`` when not provided.

    Memory tools are registered separately — see ``make_agent.memory.get_memory_executors``.
    """
    _base_dir = base_dir if base_dir is not None else Path.cwd()
    tools: dict[str, Any] = {
        "list_skills": lambda **_kw: list_skills(skills_dir),
        "read_skill": lambda name, **_kw: read_skill(name, skills_dir),
        "execute_skill": lambda name, command, **_kw: execute_skill(name, command, skills_dir, tool_timeout),
        "create_skill": lambda name, mk_content, **_kw: create_skill(name, mk_content, skills_dir),
        "validate_skill": lambda name, **_kw: validate_skill(name, skills_dir),
        "write_file": lambda path, content, **_kw: write_file(path, content, _base_dir),
        "edit_file": lambda path, old_text, new_text, **_kw: edit_file(path, old_text, new_text, _base_dir),
    }
    return {name: fn for name, fn in tools.items() if name not in disabled}


__all__ = [
    "BUILTIN_SCHEMAS",
    "BUILTIN_TOOL_NAMES",
    "FILE_SCHEMAS",
    "SKILL_SCHEMAS",
    "_valid_skill_name",
    "create_skill",
    "edit_file",
    "execute_skill",
    "get_builtin_tools",
    "list_skills",
    "read_skill",
    "validate_skill",
    "write_file",
]
