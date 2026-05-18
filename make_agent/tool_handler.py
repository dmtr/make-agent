"""ToolHandler — owns tool schemas, executor map, and dispatch for a single agent session."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from make_agent.builtin_tools import BUILTIN_SCHEMAS, get_builtin_tools, get_memory_schemas
from make_agent.memory import Memory
from make_agent.tools import ToolExecutionResult, get_tool_result

logger = logging.getLogger(__name__)


class ToolHandler:
    """Owns tool schemas, executor map, and dispatch for a single agent session.

    Assembles the full list of tool schemas (built-ins + memory tools, minus any
    disabled names) and the corresponding executor callables.  Call :meth:`execute`
    to dispatch a named tool call and receive a :class:`ToolExecutionResult`.
    """

    def __init__(
        self,
        memory: Memory,
        skills_dir: str,
        disabled: frozenset[str] = frozenset(),
        tool_timeout: int = 600,
        base_dir: Path | None = None,
    ) -> None:
        _base_dir = base_dir if base_dir is not None else Path.cwd()
        memory_schemas = get_memory_schemas()
        active_builtin_schemas = [s for s in BUILTIN_SCHEMAS if s["function"]["name"] not in disabled]
        active_memory_schemas = [s for s in memory_schemas if s["function"]["name"] not in disabled]
        self._schemas: list[dict] = active_builtin_schemas + active_memory_schemas
        self._executors: dict[str, Any] = get_builtin_tools(
            skills_dir, memory, disabled, tool_timeout, base_dir=_base_dir
        )

    @property
    def schemas(self) -> list[dict]:
        """Tool schema list passed to the LLM as the ``tools`` parameter."""
        return self._schemas

    @property
    def tool_names(self) -> set[str]:
        """Set of tool names known to this handler."""
        return {t["function"]["name"] for t in self._schemas}

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
        """Route *name* to its executor and return a :class:`ToolExecutionResult`.

        Handles unknown tool names, argument type errors, and unexpected exceptions,
        returning an error result in each case rather than propagating.
        """
        if name not in self._executors:
            return get_tool_result("", f"unknown tool: {name}", None)
        try:
            raw = self._executors[name](**arguments)
            return get_tool_result(str(raw), "", 0, max_output)
        except TypeError as e:
            logger.error("argument type error when running tool %s: %s", name, e)
            return get_tool_result("", f"argument type error: {e}", None)
        except Exception as e:
            logger.error("unexpected error when running tool %s: %s", name, e)
            return get_tool_result("", f"unexpected error: {e}", None)
