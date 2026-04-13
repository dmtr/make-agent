"""make-agent: an AI agent driven by a Makefile."""

import argparse
import logging
import sys
from pathlib import Path

from make_agent.agent import _DEFAULT_MAX_TOKENS, _DEFAULT_MAX_TOOL_OUTPUT
from make_agent.agent_shell import run
from make_agent.app_dirs import default_agents_dir, log_file
from make_agent.builtin_tools import BUILTIN_TOOL_NAMES
from make_agent.settings import load_settings, run_setup_wizard

logger = logging.getLogger(__name__)

_DEFAULT_MAKEFILE = "Makefile"
_REASONING_EFFORT_VALUES = ("none", "minimal", "low", "medium", "high", "xhigh", "auto")


def _init_logging(loglevel: str) -> None:
    level = getattr(logging, loglevel.upper(), logging.INFO)
    logging.basicConfig(filename=log_file(), level=level, format="%(asctime)s %(levelname)s %(message)s")


def _find_makefile(name: str = _DEFAULT_MAKEFILE) -> str | None:
    """Search for *name* in order: cwd, then agents dir.

    Returns the path string of the first existing file, or ``None`` if not found.
    """
    candidates = [
        Path(name),
        Path(default_agents_dir()) / Path(name).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _resolve_run_args(args: argparse.Namespace) -> argparse.Namespace:
    """Apply settings.yaml defaults and run the setup wizard when appropriate.

    Priority: CLI flag > settings.yaml > auto-discovered Makefile > code default.

    When a makefile name comes from settings.yaml but is not found in the cwd,
    the agents directory is also searched before giving up.

    The wizard is triggered only when ``-f`` was not explicitly provided,
    no ``settings.yaml`` exists for this project, *and* no Makefile is found
    in the cwd or the project agents directory.
    """
    file_explicit = args.file is not None
    model_explicit = args.model is not None

    settings = load_settings()

    if not file_explicit:
        if settings is not None and "makefile" in settings:
            # Search cwd then agents dir for the name from settings
            found = _find_makefile(settings["makefile"])
            args.file = found if found is not None else settings["makefile"]
        else:
            found = _find_makefile()
            if found is not None:
                args.file = found
            elif settings is None:
                settings = run_setup_wizard()
                args.file = settings.get("makefile", _DEFAULT_MAKEFILE)
            else:
                args.file = settings.get("makefile", _DEFAULT_MAKEFILE)

    if settings is None:
        settings = {}

    if not model_explicit:
        args.model = settings.get("model")

    # Memory: CLI flag takes precedence, then settings.yaml
    if not getattr(args, "with_memory", False):
        args.with_memory = bool(settings.get("memory", False))

    # Reasoning effort: CLI flag takes precedence, then settings.yaml, then default
    if getattr(args, "reasoning_effort", None) is None:
        raw = settings.get("reasoning_effort", "auto")
        if raw not in _REASONING_EFFORT_VALUES:
            raise ValueError(f"Invalid reasoning_effort in settings.yaml: {raw!r}. " f"Valid values: {', '.join(_REASONING_EFFORT_VALUES)}")
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

    run(
        makefile_path=Path(args.file),
        model=args.model,
        prompt=prompt,
        max_retries=args.max_retries,
        tool_timeout=args.tool_timeout,
        max_tool_output=args.max_tool_output,
        max_tokens=args.max_tokens,
        agents_dir=args.agents_dir,
        with_memory=args.with_memory,
        disabled_builtin_tools=_parse_disabled_tools(args.disable_builtin_tools),
        reasoning_effort=args.reasoning_effort,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="make-agent",
        description="An AI agent that reads its system prompt and tools from a Makefile.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── run (default) ────────────────────────────────────────────────────────
    run_p = subparsers.add_parser("run", help="Start the interactive agent (default)")
    run_p.add_argument("-f", "--file", default=None, metavar="FILE", help="Makefile to load (default: ./Makefile or value from settings.yaml)")
    run_p.add_argument("--model", default=None, metavar="MODEL", help="any-llm model string (required if not set in settings.yaml)")
    run_prompt_g = run_p.add_mutually_exclusive_group()
    run_prompt_g.add_argument("--prompt", default=None, metavar="PROMPT", help="Skip interactive mode and send this prompt to the model")
    run_prompt_g.add_argument("--prompt-file", default=None, metavar="FILE", help="Skip interactive mode and read the prompt from FILE")
    run_p.add_argument("--loglevel", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", metavar="LEVEL", help="Set logging level (default: INFO)")
    run_p.add_argument("--max-retries", type=int, default=5, metavar="N", help="Max retry attempts on rate limit (default: 5)")
    run_p.add_argument("--tool-timeout", type=int, default=600, metavar="SECONDS", help="Timeout in seconds for each tool call (default: 600)")
    run_p.add_argument("--agents-dir", default=None, metavar="DIR", help="Directory for specialist agent .mk files (default: ~/.make-agent/<project>/agents/)")
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
        "--with-memory", action="store_true", default=False, help="Enable persistent conversation memory (stored in ~/.make-agent/<project>/memory.db)"
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
    parser.add_argument("-f", "--file", default=None, metavar="FILE", help=argparse.SUPPRESS)
    parser.add_argument("--model", default=None, metavar="MODEL", help=argparse.SUPPRESS)
    legacy_prompt_g = parser.add_mutually_exclusive_group()
    legacy_prompt_g.add_argument("--prompt", default=None, metavar="PROMPT", help=argparse.SUPPRESS)
    legacy_prompt_g.add_argument("--prompt-file", default=None, metavar="FILE", help=argparse.SUPPRESS)
    parser.add_argument("--loglevel", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", metavar="LEVEL", help=argparse.SUPPRESS)
    parser.add_argument("--max-retries", type=int, default=5, metavar="N", help=argparse.SUPPRESS)
    parser.add_argument("--tool-timeout", type=int, default=600, metavar="SECONDS", help=argparse.SUPPRESS)
    parser.add_argument("--agents-dir", default=None, metavar="DIR", help=argparse.SUPPRESS)
    parser.add_argument("--max-tool-output", type=int, default=_DEFAULT_MAX_TOOL_OUTPUT, metavar="CHARS", help=argparse.SUPPRESS)
    parser.add_argument("--max-tokens", type=int, default=_DEFAULT_MAX_TOKENS, metavar="N", help=argparse.SUPPRESS)
    parser.add_argument("--with-memory", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--disable-builtin-tools", default=None, metavar="TOOLS", help=argparse.SUPPRESS)
    parser.add_argument("--reasoning-effort", choices=_REASONING_EFFORT_VALUES, default=None, metavar="EFFORT", help=argparse.SUPPRESS)

    args = parser.parse_args()
    _init_logging(args.loglevel)
    _cmd_run(args)


if __name__ == "__main__":
    main()
