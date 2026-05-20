"""Top-level entry point for the agent shell."""

from __future__ import annotations

from typing import Optional

from make_agent.agent_core import (
    _DEFAULT_COMPACT_CONTEXT_WINDOW,
    _DEFAULT_COMPACT_MAX_THRESHOLD,
    _DEFAULT_COMPACT_MIN_THRESHOLD,
    _DEFAULT_COMPACT_THRESHOLD_RATIO,
    _DEFAULT_MAX_RETRIES,
    _DEFAULT_MAX_TOKENS,
    _DEFAULT_MAX_TOOL_OUTPUT,
    _DEFAULT_REASONING_EFFORT,
    _DEFAULT_TOOL_TIMEOUT,
    _DEFAULT_USE_PROMPT_CACHE,
    AgentConfig,
    AgentManager,
)
from make_agent.app_dirs import project_dir
from make_agent.memory import Memory
from make_agent.tool_handler import ToolHandler

from .shell import MakeAgentShell


async def run(
    system_prompt: str,
    model: str,
    memory: Memory,
    tool_handler: ToolHandler,
    prompt: Optional[str] = None,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    tool_timeout: int = _DEFAULT_TOOL_TIMEOUT,
    max_tool_output: int = _DEFAULT_MAX_TOOL_OUTPUT,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    reasoning_effort: str = _DEFAULT_REASONING_EFFORT,
    compact_threshold: int | None = None,
    compact_threshold_ratio: float = _DEFAULT_COMPACT_THRESHOLD_RATIO,
    compact_min_threshold: int = _DEFAULT_COMPACT_MIN_THRESHOLD,
    compact_max_threshold: int = _DEFAULT_COMPACT_MAX_THRESHOLD,
    compact_context_window: int = _DEFAULT_COMPACT_CONTEXT_WINDOW,
    use_prompt_cache: bool = _DEFAULT_USE_PROMPT_CACHE,
) -> None:
    """Start the interactive shell (or send a single prompt and return)."""
    await tool_handler.setup(model)
    agent_config = AgentConfig(
        system_prompt=system_prompt,
        model=model,
        max_retries=max_retries,
        tool_timeout=tool_timeout,
        max_tool_output=max_tool_output,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        project_dir=project_dir(),
        compact_threshold=compact_threshold,
        compact_threshold_ratio=compact_threshold_ratio,
        compact_min_threshold=compact_min_threshold,
        compact_max_threshold=compact_max_threshold,
        compact_context_window=compact_context_window,
        use_prompt_cache=use_prompt_cache,
    )
    agent_manager = AgentManager(memory, tool_handler)
    session_id = agent_manager.create_session(agent_config)
    if system_prompt:
        print("System prompt loaded.")
    else:
        print("No system prompt — using built-in defaults.")

    if prompt:
        print("Sending initial prompt...\n")
        print(await agent_manager.arun_agent(session_id, prompt))
        return

    shell = MakeAgentShell(agent_manager, session_id)
    try:
        await shell.run()
    except KeyboardInterrupt:
        print()
