"""make-agent: an AI agent driven by a Makefile."""

import argparse
from pathlib import Path

from make_agent.agent import run


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="make-agent",
        description="An AI agent that reads its system prompt and tools from a Makefile.",
    )
    parser.add_argument(
        "-f",
        "--file",
        default="Makefile",
        metavar="FILE",
        help="Makefile to load (default: ./Makefile)",
    )
    parser.add_argument(
        "--model",
        default="anthropic/claude-haiku-4-5-20251001",
        metavar="MODEL",
        help="litellm model string (default: anthropic/claude-haiku-4-5-20251001)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        metavar="PROMPT",
        help="If set skip interactive mode and send this prompt to the model (default: None)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Log all messages to make-agent.log",
    )
    args = parser.parse_args()

    run(makefile_path=Path(args.file), model=args.model, prompt=args.prompt, debug=args.debug)


if __name__ == "__main__":
    main()
