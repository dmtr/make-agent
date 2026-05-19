"""Helpers for mode-aware built-in tool validation."""

from __future__ import annotations

_COMMON_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "list_skills",
        "read_skill",
        "execute_skill",
        "create_skill",
        "validate_skill",
        "search_user_memory",
        "search_agent_memory",
        "get_recent_messages",
    }
)
_MAKEFILE_ONLY_TOOL_NAMES: frozenset[str] = frozenset({"write_file", "edit_file"})


def builtin_tool_names(mode: str) -> frozenset[str]:
    """Return built-in tool names for a skill mode."""
    if mode == "makefile":
        return _COMMON_TOOL_NAMES | _MAKEFILE_ONLY_TOOL_NAMES
    if mode == "python":
        return _COMMON_TOOL_NAMES
    raise ValueError(f"unknown skill mode: {mode}")


__all__ = ["builtin_tool_names"]
