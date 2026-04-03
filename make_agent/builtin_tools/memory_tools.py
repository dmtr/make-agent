"""Memory search tools: search_user_memory, search_agent_memory, get_recent_messages."""

from __future__ import annotations

from typing import Any

_MEMORY_SEARCH_PARAMS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "FTS5 keyword query. Use individual keywords rather than full sentences — "
                "FTS5 matches on exact tokens, not phrases or semantics. "
                "For example, to find 'the goal of this project', use 'goal project' or 'goal'. "
                "Combine keywords with OR for broader recall: 'goal OR objective OR purpose'. "
                "Avoid stop words (the, of, is, a) as they are not indexed."
            ),
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

MEMORY_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_user_memory",
            "description": (
                "Search past user messages using keyword-based full-text search (FTS5). "
                "Use this proactively to recall context from earlier in the conversation or past sessions. "
                "Query with short keywords — FTS5 does not match full sentences. "
                "If the first query returns no results, retry with broader or alternative keywords."
            ),
            "parameters": _MEMORY_SEARCH_PARAMS,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_agent_memory",
            "description": (
                "Search past agent replies using keyword-based full-text search (FTS5). "
                "Use this to recall what you previously told the user or decisions you made. "
                "Query with short keywords — FTS5 does not match full sentences. "
                "If the first query returns no results, retry with broader or alternative keywords."
            ),
            "parameters": _MEMORY_SEARCH_PARAMS,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_messages",
            "description": (
                "Fetch the N most recent messages from memory, in chronological order. "
                "Each entry shows the timestamp, sender (user or agent), and message text. "
                "Use this to quickly recall recent conversation context without needing keywords."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of recent messages to return (default: 10).",
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
                "required": [],
            },
        },
    },
]


def get_memory_schemas() -> list[dict[str, Any]]:
    """Return the tool schemas for memory search tools."""
    return MEMORY_SCHEMAS
