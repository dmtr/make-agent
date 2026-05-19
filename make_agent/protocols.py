"""Structural protocols for the make-agent package.

Defines the minimal interfaces consumed by the Agent so implementations
(Memory, ToolHandler) can be swapped or mocked without inheriting from
concrete classes.
"""

from __future__ import annotations

from typing import Iterable, Protocol

from make_agent.tool_handler.runner import ToolExecutionResult


class MemoryProtocol(Protocol):
    """Minimal memory interface consumed by Agent."""

    def store(self, sender: str, message: str) -> None: ...

    def record_token_usage(
        self,
        session_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None: ...

    def get_session_stats(self, session_id: str) -> dict: ...


class ToolHandlerProtocol(Protocol):
    """Minimal tool-dispatch interface consumed by Agent."""

    @property
    def tool_names(self) -> Iterable[str]: ...

    @property
    def llm_tool_kwargs(self) -> dict: ...

    async def execute(
        self,
        name: str,
        arguments: dict,
        max_output: int,
    ) -> ToolExecutionResult: ...

    def get_tool_result(
        self,
        stdout: str,
        stderr: str,
        exit_code: int | None,
        max_output: int = 0,
    ) -> ToolExecutionResult: ...
