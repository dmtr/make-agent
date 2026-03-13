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
        default="openai/gpt-4o",
        metavar="MODEL",
        help="litellm model string (default: openai/gpt-4o)",
    )
    args = parser.parse_args()

    run(makefile_path=Path(args.file), model=args.model)


if __name__ == "__main__":
    main()
