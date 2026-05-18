"""make-agent: an AI agent powered by skills."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from make_agent.agent import _DEFAULT_MAX_TOKENS, _DEFAULT_MAX_TOOL_OUTPUT
from make_agent.agent_shell import run
from make_agent.app_dirs import log_file, project_dir
from make_agent.builtin_tools import BUILTIN_TOOL_NAMES
from make_agent.settings import load_settings

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT_FILE = "SYSTEM.md"
_REASONING_EFFORT_VALUES = ("none", "minimal", "low", "medium", "high", "xhigh", "auto")


def _init_logging(loglevel: str) -> None:
    level = getattr(logging, loglevel.upper(), logging.INFO)
    logging.basicConfig(filename=log_file(), level=level, format="%(asctime)s %(levelname)s %(message)s")


def _resolve_system_prompt(args: argparse.Namespace) -> str:
    """Resolve the system prompt from CLI args or SYSTEM.md file discovery.

    Priority: --system > --system-file > cwd/SYSTEM.md > ~/.make-agent/<project>/SYSTEM.md
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

    project_system = project_dir() / _DEFAULT_SYSTEM_PROMPT_FILE
    if project_system.exists():
        return project_system.read_text(encoding="utf-8")

    return ""


def _resolve_run_args(args: argparse.Namespace) -> argparse.Namespace:
    """Apply settings.yaml defaults.

    Priority: CLI flag > settings.yaml > code default.
    """
    settings = load_settings() or {}

    if args.model is None:
        args.model = settings.get("model")

    if getattr(args, "reasoning_effort", None) is None:
        raw = settings.get("reasoning_effort", "auto")
        if raw not in _REASONING_EFFORT_VALUES:
            raise ValueError(
                f"Invalid reasoning_effort in settings.yaml: {raw!r}. "
                f"Valid values: {', '.join(_REASONING_EFFORT_VALUES)}"
            )
        args.reasoning_effort = raw

    return args


def _parse_disabled_tools(value: str | None) -> frozenset[str]:
    """Parse the --disable-builtin-tools value into a frozenset of tool names.

    Accepts ``"all"`` or a comma-separated list of known built-in tool names.
    Exits with an error on unknown names.
    """
    if not value:
        return frozenset()
    if value.strip().lower() == "all":
        return BUILTIN_TOOL_NAMES
    names = frozenset(n.strip() for n in value.split(",") if n.strip())
    unknown = names - BUILTIN_TOOL_NAMES
    if unknown:
        sys.exit(f"make-agent: unknown built-in tool(s): {', '.join(sorted(unknown))}. " f"Valid names: {', '.join(sorted(BUILTIN_TOOL_NAMES))}")
    return names


def _cmd_run(args: argparse.Namespace) -> None:
    args = _resolve_run_args(args)

    if args.model is None:
        sys.exit("make-agent: model is required — pass --model or set 'model' in settings.yaml")

    prompt = args.prompt
    if args.prompt_file is not None:
        try:
            prompt = Path(args.prompt_file).read_text(encoding="utf-8")
        except OSError as e:
            sys.exit(f"make-agent run: {e}")

    system_prompt = _resolve_system_prompt(args)

    asyncio.run(
        run(
            system_prompt=system_prompt,
            model=args.model,
            prompt=prompt,
            max_retries=args.max_retries,
            tool_timeout=args.tool_timeout,
            max_tool_output=args.max_tool_output,
            max_tokens=args.max_tokens,
            skills_dir=args.skills_dir,
            disabled_builtin_tools=_parse_disabled_tools(args.disable_builtin_tools),
            reasoning_effort=args.reasoning_effort,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="make-agent",
        description="An AI agent powered by skills.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── run (default) ────────────────────────────────────────────────────────
    run_p = subparsers.add_parser("run", help="Start the interactive agent (default)")
    run_p.add_argument("--model", default=None, metavar="MODEL", help="any-llm model string (required if not set in settings.yaml)")
    system_g = run_p.add_mutually_exclusive_group()
    system_g.add_argument("--system", default=None, metavar="PROMPT", help="System prompt string (overrides SYSTEM.md discovery)")
    system_g.add_argument("--system-file", default=None, metavar="FILE", help="Read system prompt from FILE (overrides SYSTEM.md discovery)")
    run_prompt_g = run_p.add_mutually_exclusive_group()
    run_prompt_g.add_argument("--prompt", default=None, metavar="PROMPT", help="Skip interactive mode and send this prompt to the model")
    run_prompt_g.add_argument("--prompt-file", default=None, metavar="FILE", help="Skip interactive mode and read the prompt from FILE")
    run_p.add_argument("--loglevel", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", metavar="LEVEL", help="Set logging level (default: INFO)")
    run_p.add_argument("--max-retries", type=int, default=5, metavar="N", help="Max retry attempts on rate limit (default: 5)")
    run_p.add_argument("--tool-timeout", type=int, default=600, metavar="SECONDS", help="Timeout in seconds for each tool call (default: 600)")
    run_p.add_argument("--skills-dir", default=None, metavar="DIR", help="Directory for skills (default: ~/.make-agent/<project>/skills/)")
    run_p.add_argument(
        "--max-tool-output",
        type=int,
        default=_DEFAULT_MAX_TOOL_OUTPUT,
        metavar="CHARS",
        help=f"Max characters of stdout kept from each tool call; 0 = unlimited (default: {_DEFAULT_MAX_TOOL_OUTPUT})",
    )
    run_p.add_argument(
        "--max-tokens",
        type=int,
        default=_DEFAULT_MAX_TOKENS,
        metavar="N",
        help=f"Max tokens in model response (default: {_DEFAULT_MAX_TOKENS})",
    )
    run_p.add_argument(
        "--disable-builtin-tools",
        default=None,
        metavar="TOOLS",
        help=f"Comma-separated built-in tool names to disable, or 'all'. Valid names: {', '.join(sorted(BUILTIN_TOOL_NAMES))}",
    )
    run_p.add_argument(
        "--reasoning-effort",
        choices=_REASONING_EFFORT_VALUES,
        default=None,
        metavar="EFFORT",
        help=f"Reasoning effort level ({'/'.join(_REASONING_EFFORT_VALUES)}, default: auto)",
    )

    # ── legacy: no subcommand → behave as "run" ──────────────────────────────
    parser.add_argument("--model", default=None, metavar="MODEL", help=argparse.SUPPRESS)
    legacy_system_g = parser.add_mutually_exclusive_group()
    legacy_system_g.add_argument("--system", default=None, metavar="PROMPT", help=argparse.SUPPRESS)
    legacy_system_g.add_argument("--system-file", default=None, metavar="FILE", help=argparse.SUPPRESS)
    legacy_prompt_g = parser.add_mutually_exclusive_group()
    legacy_prompt_g.add_argument("--prompt", default=None, metavar="PROMPT", help=argparse.SUPPRESS)
    legacy_prompt_g.add_argument("--prompt-file", default=None, metavar="FILE", help=argparse.SUPPRESS)
    parser.add_argument("--loglevel", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", metavar="LEVEL", help=argparse.SUPPRESS)
    parser.add_argument("--max-retries", type=int, default=5, metavar="N", help=argparse.SUPPRESS)
    parser.add_argument("--tool-timeout", type=int, default=600, metavar="SECONDS", help=argparse.SUPPRESS)
    parser.add_argument("--skills-dir", default=None, metavar="DIR", help=argparse.SUPPRESS)
    parser.add_argument("--max-tool-output", type=int, default=_DEFAULT_MAX_TOOL_OUTPUT, metavar="CHARS", help=argparse.SUPPRESS)
    parser.add_argument("--max-tokens", type=int, default=_DEFAULT_MAX_TOKENS, metavar="N", help=argparse.SUPPRESS)
    parser.add_argument("--disable-builtin-tools", default=None, metavar="TOOLS", help=argparse.SUPPRESS)
    parser.add_argument("--reasoning-effort", choices=_REASONING_EFFORT_VALUES, default=None, metavar="EFFORT", help=argparse.SUPPRESS)

    args = parser.parse_args()
    _init_logging(args.loglevel)
    _cmd_run(args)


if __name__ == "__main__":
    main()
