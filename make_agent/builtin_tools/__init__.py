"""Built-in agent management tools always available to every agent.

These tools are injected into every agent's tool schema alongside any
Makefile-defined tools, without requiring a Makefile declaration.

Sub-modules:
- ``agent_tools``   — list/validate/create/load specialist agents
- ``memory_tools``  — FTS5 search and recall over past messages
- ``file_tools``    — sandboxed line-level file read/write access
"""

from __future__ import annotations

from typing import Any

from make_agent.builtin_tools.agent_tools import (
    AGENT_SCHEMAS,
    _agent_summary,
    _RunAgent,
    _valid_agent_name,
    create_agent,
    list_agent,
    run_agent,
    validate_agent,
)
from make_agent.builtin_tools.file_tools import (
    FILE_TOOL_NAMES,
    FILE_TOOL_SCHEMAS,
    get_file_tools,
    insert_lines,
    read_file,
    replace_lines,
)
from make_agent.builtin_tools.memory_tools import MEMORY_SCHEMAS, get_memory_schemas

BUILTIN_TOOL_NAMES: frozenset[str] = (
    frozenset(
        {
            "list_agent",
            "validate_agent",
            "create_agent",
            "run_agent",
            "search_user_memory",
            "search_agent_memory",
            "get_recent_messages",
        }
    )
    | FILE_TOOL_NAMES
)

# The 4 agent-management schemas (memory and file schemas are assembled separately).
BUILTIN_SCHEMAS: list[dict[str, Any]] = AGENT_SCHEMAS


def get_builtin_tools(agents_dir: str, memory: Any = None, disabled: frozenset[str] = frozenset(), tool_timeout: int = 600) -> dict[str, Any]:
    """Return a name → callable mapping for all built-in tools.

    Each callable accepts only the LLM-provided arguments; ``agents_dir``
    and ``memory`` are pre-bound via closure.  Tools whose names appear in
    *disabled* are omitted.
    """
    tools: dict[str, Any] = {
        "list_agent": lambda **_kw: list_agent(agents_dir),
        "validate_agent": lambda name, **_kw: validate_agent(name, agents_dir),
        "create_agent": lambda name, spec, **_kw: create_agent(name, spec, agents_dir),
        "run_agent": lambda name, prompt, **_kw: run_agent(name, prompt, agents_dir),
    }
    if memory is not None:
        tools["search_user_memory"] = lambda query, limit=10, from_date=None, to_date=None, **_kw: memory.search_user(query, limit, from_date, to_date)
        tools["search_agent_memory"] = lambda query, limit=10, from_date=None, to_date=None, **_kw: memory.search_agent(query, limit, from_date, to_date)
        tools["get_recent_messages"] = lambda limit=10, from_date=None, to_date=None, **_kw: memory.recent(limit, from_date, to_date)

    tools.update(get_file_tools(disabled))

    return {name: fn for name, fn in tools.items() if name not in disabled}


__all__ = [
    "AGENT_SCHEMAS",
    "BUILTIN_SCHEMAS",
    "BUILTIN_TOOL_NAMES",
    "FILE_TOOL_NAMES",
    "FILE_TOOL_SCHEMAS",
    "MEMORY_SCHEMAS",
    "_RunAgent",
    "_agent_summary",
    "_valid_agent_name",
    "create_agent",
    "get_builtin_tools",
    "get_file_tools",
    "get_memory_schemas",
    "insert_lines",
    "list_agent",
    "read_file",
    "replace_lines",
    "run_agent",
    "validate_agent",
]
