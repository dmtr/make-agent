"""make-agent: an AI agent powered by skills."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from make_agent.agent_core import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TOOL_OUTPUT,
    DEFAULT_USE_PROMPT_CACHE,
)
from make_agent.agent_shell import run
from make_agent.app_dirs import (
    default_skills_dir,
    ensure_mode_system_prompt,
    log_file,
    mode_dir,
    mode_memory_path,
)
from make_agent.builtin_tools import builtin_tool_names
from make_agent.memory import Memory
from make_agent.skill_backend import MakefileSkillBackend, PythonSkillBackend
from make_agent.tool_handler import ToolHandler

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT_FILE = "SYSTEM.md"
_REASONING_EFFORT_VALUES = ("none", "minimal", "low", "medium", "high", "xhigh")
_SKILL_MODES = ("makefile", "python")


def _init_logging(loglevel: str) -> None:
    level = getattr(logging, loglevel.upper(), logging.INFO)
    logging.basicConfig(filename=log_file(), level=level, format="%(asctime)s %(levelname)s %(message)s")


def _resolve_system_prompt(args: argparse.Namespace) -> str:
    """Resolve the system prompt from CLI args or SYSTEM.md file discovery.

    Priority: --system > --system-file > cwd/SYSTEM.md > ~/.make-agent/<project>/<mode>/SYSTEM.md
    Returns an empty string when none is found.
    """
    if getattr(args, "system", None):
        return args.system

    if getattr(args, "system_file", None):
        try:
            return Path(args.system_file).read_text(encoding="utf-8")
        except OSError as e:
            sys.exit(f"make-agent: {e}")

    cwd_system = Path(_DEFAULT_SYSTEM_PROMPT_FILE)
    if cwd_system.exists():
        return cwd_system.read_text(encoding="utf-8")

    project_system = mode_dir(getattr(args, "skill_mode", "python")) / _DEFAULT_SYSTEM_PROMPT_FILE
    if project_system.exists():
        return project_system.read_text(encoding="utf-8")

    return ""


def _parse_disabled_tools(value: str | None, mode: str) -> frozenset[str]:
    """Parse the --disable-builtin-tools value into a frozenset of tool names."""
    available = builtin_tool_names(mode)
    if not value:
        return frozenset()
    if value.strip().lower() == "all":
        return available
    names = frozenset(name.strip() for name in value.split(",") if name.strip())
    unknown = names - available
    if unknown:
        sys.exit("make-agent: unknown built-in tool(s): " f"{', '.join(sorted(unknown))}. Valid names for {mode}: {', '.join(sorted(available))}")
    return names


def _parse_trusted_skills(value: str | None) -> frozenset[str]:
    """Parse --trusted-skills into a frozenset. 'all' maps to {'*'} (trust everything)."""
    if not value:
        return frozenset()
    if value.strip().lower() == "all":
        return frozenset(["*"])
    return frozenset(name.strip() for name in value.split(",") if name.strip())


def _build_backend(skill_mode: str, skills_dir: str, tool_timeout: int):
    if skill_mode == "makefile":
        return MakefileSkillBackend(skills_dir, tool_timeout, Path.cwd())
    return PythonSkillBackend(skills_dir, tool_timeout)


def _cmd_run(args: argparse.Namespace) -> None:
    if args.model is None:
        sys.exit("make-agent: --model is required")

    prompt = args.prompt
    if args.prompt_file is not None:
        try:
            prompt = Path(args.prompt_file).read_text(encoding="utf-8")
        except OSError as e:
            sys.exit(f"make-agent run: {e}")

    ensure_mode_system_prompt(args.skill_mode)
    system_prompt = _resolve_system_prompt(args)
    disabled = _parse_disabled_tools(args.disable_builtin_tools, args.skill_mode)
    if args.skills_dir:
        skills_dir = str(Path(args.skills_dir) / args.skill_mode)
    else:
        skills_dir = str(default_skills_dir(args.skill_mode))

    memory = Memory(mode_memory_path(args.skill_mode))
    backend = _build_backend(args.skill_mode, skills_dir, args.tool_timeout)
    trusted_skills = _parse_trusted_skills(getattr(args, "trusted_skills", None))
    tool_handler = ToolHandler(backend, memory, disabled, trusted_skills)

    asyncio.run(
        run(
            system_prompt=system_prompt,
            model=args.model,
            memory=memory,
            tool_handler=tool_handler,
            prompt=prompt,
            max_retries=args.max_retries,
            tool_timeout=args.tool_timeout,
            max_tool_output=args.max_tool_output,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
            use_prompt_cache=args.prompt_cache,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="make-agent",
        description="An AI agent powered by skills.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_p = subparsers.add_parser("run", help="Start the interactive agent (default)")
    run_p.add_argument("--model", default=None, metavar="MODEL", help="litellm model string (required)")
    system_g = run_p.add_mutually_exclusive_group()
    system_g.add_argument(
        "--system",
        default=None,
        metavar="PROMPT",
        help="System prompt string (overrides SYSTEM.md discovery)",
    )
    system_g.add_argument(
        "--system-file",
        default=None,
        metavar="FILE",
        help="Read system prompt from FILE (overrides SYSTEM.md discovery)",
    )
    run_prompt_g = run_p.add_mutually_exclusive_group()
    run_prompt_g.add_argument(
        "--prompt",
        default=None,
        metavar="PROMPT",
        help="Skip interactive mode and send this prompt to the model",
    )
    run_prompt_g.add_argument(
        "--prompt-file",
        default=None,
        metavar="FILE",
        help="Skip interactive mode and read the prompt from FILE",
    )
    run_p.add_argument(
        "--loglevel",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        metavar="LEVEL",
        help="Set logging level (default: INFO)",
    )
    run_p.add_argument(
        "--max-retries",
        type=int,
        default=5,
        metavar="N",
        help="Max retry attempts on rate limit (default: 5)",
    )
    run_p.add_argument(
        "--tool-timeout",
        type=int,
        default=600,
        metavar="SECONDS",
        help="Timeout in seconds for each tool call (default: 600)",
    )
    run_p.add_argument(
        "--skill-mode",
        choices=_SKILL_MODES,
        default="python",
        metavar="MODE",
        help="Skill backend mode to use (default: python)",
    )
    run_p.add_argument(
        "--skills-dir",
        default=None,
        metavar="DIR",
        help="Directory for skills (default: ~/.make-agent/<project>/<mode>/skills/)",
    )
    run_p.add_argument(
        "--max-tool-output",
        type=int,
        default=DEFAULT_MAX_TOOL_OUTPUT,
        metavar="CHARS",
        help=f"Max characters of stdout kept from each tool call; 0 = unlimited (default: {DEFAULT_MAX_TOOL_OUTPUT})",
    )
    run_p.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        metavar="N",
        help=f"Max tokens in model response (default: {DEFAULT_MAX_TOKENS})",
    )
    run_p.add_argument(
        "--disable-builtin-tools",
        default=None,
        metavar="TOOLS",
        help="Comma-separated built-in tool names to disable, or 'all'. Valid names depend on --skill-mode.",
    )
    run_p.add_argument(
        "--reasoning-effort",
        choices=_REASONING_EFFORT_VALUES,
        default="medium",
        metavar="EFFORT",
        help=f"Reasoning effort level ({'/'.join(_REASONING_EFFORT_VALUES)}, default: auto)",
    )
    run_p.add_argument(
        "--trusted-skills",
        default=None,
        metavar="SKILLS",
        help="Comma-separated skill names that run without confirmation, or 'all'. "
        "Use 'skill.target' to trust a specific target (Python mode only). "
        "Unspecified skills prompt the user before each execution.",
    )
    run_p.add_argument(
        "--prompt-cache",
        action="store_true",
        default=DEFAULT_USE_PROMPT_CACHE,
        help="Enable prompt caching for the system prompt (Anthropic models only)",
    )
    parser.add_argument("--model", default=None, metavar="MODEL", help=argparse.SUPPRESS)
    legacy_system_g = parser.add_mutually_exclusive_group()
    legacy_system_g.add_argument("--system", default=None, metavar="PROMPT", help=argparse.SUPPRESS)
    legacy_system_g.add_argument("--system-file", default=None, metavar="FILE", help=argparse.SUPPRESS)
    legacy_prompt_g = parser.add_mutually_exclusive_group()
    legacy_prompt_g.add_argument("--prompt", default=None, metavar="PROMPT", help=argparse.SUPPRESS)
    legacy_prompt_g.add_argument("--prompt-file", default=None, metavar="FILE", help=argparse.SUPPRESS)
    parser.add_argument(
        "--loglevel",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        metavar="LEVEL",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--max-retries", type=int, default=5, metavar="N", help=argparse.SUPPRESS)
    parser.add_argument(
        "--tool-timeout",
        type=int,
        default=600,
        metavar="SECONDS",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--skill-mode",
        choices=_SKILL_MODES,
        default="python",
        metavar="MODE",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--skills-dir", default=None, metavar="DIR", help=argparse.SUPPRESS)
    parser.add_argument(
        "--max-tool-output",
        type=int,
        default=DEFAULT_MAX_TOOL_OUTPUT,
        metavar="CHARS",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        metavar="N",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--disable-builtin-tools", default=None, metavar="TOOLS", help=argparse.SUPPRESS)
    parser.add_argument(
        "--reasoning-effort",
        choices=_REASONING_EFFORT_VALUES,
        default="medium",
        metavar="EFFORT",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--trusted-skills", default=None, metavar="SKILLS", help=argparse.SUPPRESS)
    parser.add_argument(
        "--prompt-cache",
        action="store_true",
        default=DEFAULT_USE_PROMPT_CACHE,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    _init_logging(args.loglevel)
    _cmd_run(args)


if __name__ == "__main__":
    main()
