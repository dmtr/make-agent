"""Tests for the trusted-skill confirmation mechanism in ToolHandler."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from make_agent.tool_handler.handler import ToolHandler, _SKILL_EXECUTION_TOOL
from make_agent.tool_handler.runner import get_tool_result
from make_agent.main import _parse_trusted_skills


# ── _parse_trusted_skills ──────────────────────────────────────────────────────


def test_parse_trusted_skills_none():
    assert _parse_trusted_skills(None) == frozenset()


def test_parse_trusted_skills_empty_string():
    assert _parse_trusted_skills("") == frozenset()


def test_parse_trusted_skills_single():
    assert _parse_trusted_skills("web-fetch") == frozenset(["web-fetch"])


def test_parse_trusted_skills_multiple():
    result = _parse_trusted_skills("web-fetch,search,summarise")
    assert result == frozenset(["web-fetch", "search", "summarise"])


def test_parse_trusted_skills_strips_whitespace():
    result = _parse_trusted_skills(" web-fetch , search ")
    assert result == frozenset(["web-fetch", "search"])


def test_parse_trusted_skills_all():
    assert _parse_trusted_skills("all") == frozenset(["*"])


def test_parse_trusted_skills_all_case_insensitive():
    assert _parse_trusted_skills("ALL") == frozenset(["*"])


# ── ToolHandler._is_trusted ────────────────────────────────────────────────────


def _make_handler(trusted_skills: frozenset[str] = frozenset()) -> ToolHandler:
    backend = MagicMock()
    backend.schemas = []
    backend.executors = {}
    memory = MagicMock()
    memory.store = MagicMock()
    from make_agent.memory import MEMORY_SCHEMAS, get_memory_executors
    backend.schemas = []
    backend.executors = {"execute_skill": AsyncMock(return_value="ok")}
    backend.get_skill_trusted = MagicMock(return_value=None)
    return ToolHandler(backend, memory, trusted_skills=trusted_skills)


def test_is_trusted_empty_set():
    h = _make_handler(frozenset())
    assert not h._is_trusted("web-fetch", "fetch")


def test_is_trusted_specific_match():
    h = _make_handler(frozenset(["web-fetch"]))
    assert h._is_trusted("web-fetch", "fetch")
    assert not h._is_trusted("search", "query")


def test_is_trusted_wildcard():
    h = _make_handler(frozenset(["*"]))
    assert h._is_trusted("web-fetch", "fetch")
    assert h._is_trusted("any-skill", "run")


def test_is_trusted_dot_notation_specific_target():
    h = _make_handler(frozenset(["web.fetch"]))
    assert h._is_trusted("web", "fetch")
    assert not h._is_trusted("web", "search")
    assert not h._is_trusted("other", "fetch")


def test_is_trusted_dot_notation_does_not_trust_whole_skill():
    h = _make_handler(frozenset(["web.fetch"]))
    assert not h._is_trusted("web", "search")


def test_is_trusted_skill_level_trusts_all_targets():
    h = _make_handler(frozenset(["web"]))
    assert h._is_trusted("web", "fetch")
    assert h._is_trusted("web", "search")
    assert not h._is_trusted("other", "fetch")


def test_is_trusted_backend_ast_trust():
    """Backend returning trusted=True from AST grants trust without CLI override."""
    backend = MagicMock()
    backend.schemas = []
    backend.executors = {"execute_skill": AsyncMock(return_value="ok")}
    backend.get_skill_trusted = MagicMock(return_value=True)
    memory = MagicMock()
    memory.store = MagicMock()
    h = ToolHandler(backend, memory, trusted_skills=frozenset())
    assert h._is_trusted("web", "fetch")


def test_is_trusted_backend_ast_untrusted():
    """Backend returning trusted=False (AST untrusted) requires CLI override."""
    backend = MagicMock()
    backend.schemas = []
    backend.executors = {"execute_skill": AsyncMock(return_value="ok")}
    backend.get_skill_trusted = MagicMock(return_value=False)
    memory = MagicMock()
    memory.store = MagicMock()
    h = ToolHandler(backend, memory, trusted_skills=frozenset())
    assert not h._is_trusted("web", "fetch")


def test_is_trusted_backend_ast_none():
    """Backend returning None (Makefile backend) requires CLI override."""
    backend = MagicMock()
    backend.schemas = []
    backend.executors = {"execute_skill": AsyncMock(return_value="ok")}
    backend.get_skill_trusted = MagicMock(return_value=None)
    memory = MagicMock()
    memory.store = MagicMock()
    h = ToolHandler(backend, memory, trusted_skills=frozenset())
    assert not h._is_trusted("web", "fetch")


# ── ToolHandler.execute — skill confirmation ───────────────────────────────────


async def test_execute_skill_trusted_skips_confirm():
    """Trusted skills execute without calling the confirm callback."""
    handler = _make_handler(frozenset(["web-fetch"]))
    confirm = AsyncMock(return_value=False)  # would deny if called
    handler.set_confirm(confirm)

    await handler.execute(
        _SKILL_EXECUTION_TOOL,
        {"name": "web-fetch", "target": "fetch_page", "kwargs": {}},
    )

    confirm.assert_not_called()


async def test_execute_skill_untrusted_confirm_allowed():
    """Untrusted skill executes when confirm returns True."""
    handler = _make_handler(frozenset())
    confirm = AsyncMock(return_value=True)
    handler.set_confirm(confirm)

    result = await handler.execute(
        _SKILL_EXECUTION_TOOL,
        {"name": "web-fetch", "target": "fetch_page", "kwargs": {"url": "http://x"}},
    )

    confirm.assert_awaited_once_with("web-fetch", "fetch_page", {"url": "http://x"})
    assert not result.is_error


async def test_execute_skill_untrusted_confirm_denied():
    """Untrusted skill is blocked when confirm returns False."""
    handler = _make_handler(frozenset())
    confirm = AsyncMock(return_value=False)
    handler.set_confirm(confirm)

    result = await handler.execute(
        _SKILL_EXECUTION_TOOL,
        {"name": "web-fetch", "target": "fetch_page", "kwargs": {}},
    )

    confirm.assert_awaited_once()
    assert result.is_error
    assert "User denied execution" in result.output
    assert "web-fetch/fetch_page" in result.output


async def test_execute_skill_untrusted_no_confirm_callback_denies():
    """When no confirm callback is set, untrusted skills are denied."""
    handler = _make_handler(frozenset())
    # no set_confirm called

    result = await handler.execute(
        _SKILL_EXECUTION_TOOL,
        {"name": "web-fetch", "target": "fetch_page", "kwargs": {}},
    )

    assert result.is_error
    assert "User denied execution" in result.output


async def test_execute_skill_trust_all_skips_confirm():
    """Wildcard trust skips confirmation for any skill."""
    handler = _make_handler(frozenset(["*"]))
    confirm = AsyncMock(return_value=False)
    handler.set_confirm(confirm)

    await handler.execute(
        _SKILL_EXECUTION_TOOL,
        {"name": "anything", "target": "do_it", "kwargs": {}},
    )

    confirm.assert_not_called()


async def test_execute_non_skill_tool_bypasses_trust_check():
    """Non-execute_skill tools are never gated by the trust mechanism."""
    backend = MagicMock()
    backend.schemas = []
    backend.executors = {"list_skills": MagicMock(return_value="skill-list")}
    backend.get_skill_trusted = MagicMock(return_value=None)
    memory = MagicMock()
    memory.store = MagicMock()
    handler = ToolHandler(backend, memory, trusted_skills=frozenset())
    confirm = AsyncMock(return_value=False)
    handler.set_confirm(confirm)

    result = await handler.execute("list_skills", {})

    confirm.assert_not_called()
    assert not result.is_error
