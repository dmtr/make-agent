"""ToolHandler — owns tool schemas, executor map, and dispatch for a single agent session."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from make_agent.memory import MEMORY_SCHEMAS, Memory, get_memory_executors
from make_agent.skill_backend import SkillBackend

from .runner import ToolExecutionResult, get_tool_result

logger = logging.getLogger(__name__)

_SKILL_EXECUTION_TOOL = "execute_skill"


class ToolHandler:
    """Owns tool schemas, executor map, and dispatch for a single agent session."""

    def __init__(
        self,
        backend: SkillBackend,
        memory: Memory,
        disabled: frozenset[str] = frozenset(),
        trusted_skills: frozenset[str] = frozenset(),
    ) -> None:
        active_backend_schemas = [
            schema
            for schema in backend.schemas
            if schema["function"]["name"] not in disabled
        ]
        active_memory_schemas = [
            schema
            for schema in MEMORY_SCHEMAS
            if schema["function"]["name"] not in disabled
        ]
        self._schemas: list[dict] = active_backend_schemas + active_memory_schemas
        self._executors: dict[str, Any] = {
            **{
                name: executor
                for name, executor in backend.executors.items()
                if name not in disabled
            },
            **{
                name: executor
                for name, executor in get_memory_executors(memory).items()
                if name not in disabled
            },
        }
        self._backend = backend
        self._trusted_skills = trusted_skills

    get_tool_result = staticmethod(get_tool_result)

    def is_skill_trusted(self, skill_name: str, target: str) -> bool:
        """Return True if *skill_name*/*target* may execute without user confirmation."""
        if "*" in self._trusted_skills:
            return True
        if f"{skill_name}.{target}" in self._trusted_skills:
            return True
        if skill_name in self._trusted_skills:
            return True
        backend_trusted = self._backend.get_skill_trusted(skill_name)
        return backend_trusted is True

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

    async def setup(self, model: str) -> None:
        await self._backend.setup(model)

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
