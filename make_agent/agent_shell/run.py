"""Top-level entry point for the agent shell."""

from __future__ import annotations

import logging
from typing import Optional

from make_agent.agent_core import (
    DEFAULT_COMPACT_MODE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TOOL_OUTPUT,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_TOOL_TIMEOUT,
    DEFAULT_USE_PROMPT_CACHE,
    AgentConfig,
    AgentManager,
    SessionMiddleware,
)
from make_agent.app_dirs import project_dir
from make_agent.memory import Memory
from make_agent.tool_handler import ToolHandler

from .shell import MakeAgentShell

logger = logging.getLogger(__name__)


async def run(
    system_prompt: str,
    model: str,
    memory: Memory,
    tool_handler: ToolHandler,
    prompt: Optional[str] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    tool_timeout: int = DEFAULT_TOOL_TIMEOUT,
    max_tool_output: int = DEFAULT_MAX_TOOL_OUTPUT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    use_prompt_cache: bool = DEFAULT_USE_PROMPT_CACHE,
    compact_mode: str = DEFAULT_COMPACT_MODE,
) -> None:
    """Start the interactive shell (or send a single prompt and return)."""
    agent_config = AgentConfig(
        system_prompt=system_prompt,
        model=model,
        max_retries=max_retries,
        tool_timeout=tool_timeout,
        max_tool_output=max_tool_output,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        project_dir=project_dir(),
        use_prompt_cache=use_prompt_cache,
        compact_mode=compact_mode,
    )
    agent_manager = AgentManager(
        tool_handler,
        middlewares=[SessionMiddleware(memory)],
    )
    session_id = agent_manager.create_session(agent_config)
    if system_prompt:
        print("System prompt loaded.")
    else:
        print("No system prompt — using built-in defaults.")

    if prompt:
        print("Sending initial prompt...\n")
        print(await agent_manager.arun_agent(session_id, prompt))
        return

    shell = MakeAgentShell(
        agent_manager,
        session_id,
        model=model,
        history_path=project_dir() / "history",
        memory=memory,
    )
    try:
        await shell.run()
    except KeyboardInterrupt:
        print()
