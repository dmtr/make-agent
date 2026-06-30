"""ToolHandler — owns tool schemas, executor map, and dispatch for a single agent session."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from make_agent.builtin_tools.file_tools import FILE_SCHEMAS, edit_file, write_file
from make_agent.builtin_tools.skill_tools import SKILL_SCHEMAS as MAKEFILE_SKILL_SCHEMAS
from make_agent.builtin_tools.skill_tools import create_skill as create_makefile_skill
from make_agent.builtin_tools.skill_tools import execute_skill as execute_makefile_skill
from make_agent.builtin_tools.skill_tools import list_skills as list_makefile_skills
from make_agent.builtin_tools.skill_tools import read_skill as read_makefile_skill
from make_agent.builtin_tools.skill_tools import (
    validate_skill as validate_makefile_skill,
)
from make_agent.memory import MEMORY_SCHEMAS, Memory, get_memory_executors

from .runner import ToolExecutionResult, get_tool_result

logger = logging.getLogger(__name__)


class ToolHandler:
    """Owns tool schemas, executor map, and dispatch for a single agent session."""

    def __init__(
        self,
        skills_dir: str,
        memory: Memory,
        tool_timeout: int = 600,
        base_dir: Path | None = None,
        enabled_skills: frozenset[str] | None = None,
        disabled: frozenset[str] = frozenset(),
        trusted_skills: frozenset[str] = frozenset(),
    ) -> None:
        _base_dir = base_dir if base_dir is not None else Path.cwd()
        self._skills_dir = skills_dir

        skill_executors: dict[str, Any] = {
            "list_skills": lambda **_kw: list_makefile_skills(
                skills_dir, enabled_skills
            ),
            "read_skill": lambda name, **_kw: read_makefile_skill(name, skills_dir),
            "execute_skill": lambda name, command, **_kw: execute_makefile_skill(
                name, command, skills_dir, tool_timeout
            ),
            "create_skill": lambda name, mk_content, **_kw: create_makefile_skill(
                name, mk_content, skills_dir
            ),
            "validate_skill": lambda name, **_kw: validate_makefile_skill(
                name, skills_dir
            ),
            "write_file": lambda path, content, **_kw: write_file(
                path, content, _base_dir
            ),
            "edit_file": lambda path, old_text, new_text, **_kw: edit_file(
                path, old_text, new_text, _base_dir
            ),
        }
        all_schemas = MAKEFILE_SKILL_SCHEMAS + FILE_SCHEMAS + MEMORY_SCHEMAS
        all_executors = {**skill_executors, **get_memory_executors(memory)}

        self._schemas: list[dict] = [
            s for s in all_schemas if s["function"]["name"] not in disabled
        ]
        self._executors: dict[str, Any] = {
            name: executor
            for name, executor in all_executors.items()
            if name not in disabled
        }
        self._trusted_skills = trusted_skills

    get_tool_result = staticmethod(get_tool_result)

    def is_skill_trusted(self, skill_name: str, target: str) -> bool:
        """Return True if *skill_name*/*target* may execute without user confirmation."""
        if "*" in self._trusted_skills:
            return True
        if f"{skill_name}.{target}" in self._trusted_skills:
            return True
        return skill_name in self._trusted_skills

    @property
    def schemas(self) -> list[dict]:
        """Tool schema list passed to the LLM as the ``tools`` parameter."""
        return self._schemas

    @property
    def tool_names(self) -> set[str]:
        """Set of tool names known to this handler."""
        return {tool["function"]["name"] for tool in self._schemas}

    @property
    def llm_tool_kwargs(self) -> dict:
        """Returns ``{"tools": ..., "tool_choice": "auto"}`` when tools exist, else ``{}``."""
        if self._schemas:
            return {"tools": self._schemas, "tool_choice": "auto"}
        return {}

    async def execute(
        self,
        name: str,
        arguments: dict,
        max_output: int = 0,
    ) -> ToolExecutionResult:
        """Route *name* to its executor and return a :class:`ToolExecutionResult`."""
        if name not in self._executors:
            return get_tool_result("", f"unknown tool: {name}", None)
        try:
            raw = self._executors[name](**arguments)
            if asyncio.iscoroutine(raw):
                raw = await raw
            return get_tool_result(str(raw), "", 0, max_output)
        except TypeError as e:
            logger.error("argument type error when running tool %s: %s", name, e)
            return get_tool_result("", f"argument type error: {e}", None)
        except Exception as e:
            logger.error("unexpected error when running tool %s: %s", name, e)
            return get_tool_result("", f"unexpected error: {e}", None)
