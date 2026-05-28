"""Top-level entry point for the agent shell."""

from __future__ import annotations

import logging
from typing import Optional

import litellm
from make_agent.agent_core import (
    DEFAULT_COMPACT_MAX_THRESHOLD,
    DEFAULT_COMPACT_MIN_THRESHOLD,
    DEFAULT_COMPACT_THRESHOLD_RATIO,
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


def _compute_compact_threshold(
    model: str,
    context_window: int,
    threshold_ratio: float,
    min_threshold: int,
    max_threshold: int,
) -> int:
    """Compute the auto-compact threshold in tokens.

    Uses *context_window* if non-zero; otherwise auto-detects via
    ``litellm.get_model_info``.  Returns 0 (disabled) when the context window
    cannot be determined.
    """
    max_input = context_window
    if not max_input:
        try:
            info = litellm.get_model_info(model)
            max_input = info.get("max_input_tokens") or info.get("max_tokens") or 0
        except Exception:
            logger.exception("Could not get model info for %r; auto-compact disabled", model)
            return 0
    if not max_input:
        return 0
    threshold = int(max_input * threshold_ratio)
    return max(min_threshold, min(max_threshold, threshold))


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
    compact_context_window: int = 0,
    compact_threshold_ratio: float = DEFAULT_COMPACT_THRESHOLD_RATIO,
    compact_min_threshold: int = DEFAULT_COMPACT_MIN_THRESHOLD,
    compact_max_threshold: int = DEFAULT_COMPACT_MAX_THRESHOLD,
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
        use_prompt_cache=use_prompt_cache,
    )
    compact_threshold = _compute_compact_threshold(
        model=model,
        context_window=compact_context_window,
        threshold_ratio=compact_threshold_ratio,
        min_threshold=compact_min_threshold,
        max_threshold=compact_max_threshold,
    )
    logger.info("Auto-compact threshold set to %d tokens", compact_threshold)
    agent_manager = AgentManager(
        tool_handler,
        middlewares=[SessionMiddleware(memory)],
        compact_threshold=compact_threshold,
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
    )
    try:
        await shell.run()
    except KeyboardInterrupt:
        print()
