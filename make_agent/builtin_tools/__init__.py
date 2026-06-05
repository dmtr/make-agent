"""Helpers for built-in tool validation (makefile mode only)."""

from __future__ import annotations

_BUILTIN_TOOL_NAMES: frozenset[str] = frozenset(
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


def builtin_tool_names(mode: str) -> frozenset[str]:
    """Return built-in tool names. Only makefile mode is supported."""
    if mode != "makefile":
        raise ValueError(
            f"unsupported skill mode: {mode!r}. Only 'makefile' is supported."
        )
    return _BUILTIN_TOOL_NAMES


__all__ = ["builtin_tool_names"]
