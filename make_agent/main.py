"""make-agent: an AI agent driven by a Makefile."""

import argparse
import sys
from pathlib import Path

from make_agent.agent import run
from make_agent.parser import parse_file, validate


def _cmd_run(args: argparse.Namespace) -> None:
    run(
        makefile_path=Path(args.file),
        model=args.model,
        prompt=args.prompt,
        debug=args.debug,
        max_retries=args.max_retries,
        tool_timeout=args.tool_timeout,
    )


def _cmd_validate(args: argparse.Namespace) -> None:
    """Load *args.file* and report any @param/recipe mismatches."""
    try:
        mf = parse_file(args.file)
    except OSError as e:
        sys.exit(f"make-agent validate: {e}")

    errors = validate(mf)
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        sys.exit(1)

    tool_count = sum(1 for r in mf.rules if r.params or r.description)
    print(f"OK — {args.file} ({tool_count} tool(s) validated)")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="make-agent",
        description="An AI agent that reads its system prompt and tools from a Makefile.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── run (default) ────────────────────────────────────────────────────────
    run_p = subparsers.add_parser("run", help="Start the interactive agent (default)")
    run_p.add_argument("-f", "--file", default="Makefile", metavar="FILE",
                       help="Makefile to load (default: ./Makefile)")
    run_p.add_argument("--model", default="anthropic/claude-haiku-4-5-20251001",
                       metavar="MODEL", help="litellm model string")
    run_p.add_argument("--prompt", default=None, metavar="PROMPT",
                       help="Skip interactive mode and send this prompt to the model")
    run_p.add_argument("--debug", action="store_true", default=False,
                       help="Log all messages to make-agent.log")
    run_p.add_argument("--max-retries", type=int, default=5, metavar="N",
                       help="Max retry attempts on rate limit (default: 5)")
    run_p.add_argument("--tool-timeout", type=int, default=600, metavar="SECONDS",
                       help="Timeout in seconds for each tool call (default: 600)")

    # ── validate ─────────────────────────────────────────────────────────────
    val_p = subparsers.add_parser(
        "validate",
        help="Check that every @param variable is referenced in its recipe",
    )
    val_p.add_argument("-f", "--file", default="Makefile", metavar="FILE",
                       help="Makefile to validate (default: ./Makefile)")

    # ── legacy: no subcommand → behave as "run" ──────────────────────────────
    parser.add_argument("-f", "--file", default="Makefile", metavar="FILE",
                        help=argparse.SUPPRESS)
    parser.add_argument("--model", default="anthropic/claude-haiku-4-5-20251001",
                        metavar="MODEL", help=argparse.SUPPRESS)
    parser.add_argument("--prompt", default=None, metavar="PROMPT",
                        help=argparse.SUPPRESS)
    parser.add_argument("--debug", action="store_true", default=False,
                        help=argparse.SUPPRESS)
    parser.add_argument("--max-retries", type=int, default=5, metavar="N",
                        help=argparse.SUPPRESS)
    parser.add_argument("--tool-timeout", type=int, default=600, metavar="SECONDS",
                        help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.command == "validate":
        _cmd_validate(args)
    else:
        # "run" subcommand or legacy invocation (no subcommand)
        _cmd_run(args)


if __name__ == "__main__":
    main()
