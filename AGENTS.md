# AGENTS.md

Guidance for AI coding assistants working on this repository.

## What this project does

**make-agent** is a Python framework for building AI agents defined entirely in Makefiles. Each Makefile target annotated with a `# <tool>` comment block becomes an LLM-callable tool. A `define SYSTEM_PROMPT` block sets the agent's system prompt. The agent invokes tools by running `make <target> KEY=value …`.

## Repository layout

```
make_agent/          # Source package
  main.py            # CLI entry point (argparse, settings resolution)
  agent.py           # Agent class — conversation loop, LLM calls
  agent_shell.py     # Interactive REPL (cmd.Cmd wrapper)
  parser.py          # GNU Make parser (variables, rules, tool blocks)
  tools.py           # Tool schema builder and subprocess executor
  builtin_tools.py   # Built-in tools (list/create/validate/run_agent + memory)
  create_agent.py    # make-agent-create CLI: YAML spec → Makefile
  memory.py          # SQLite + FTS5 persistent memory
  settings.py        # Per-project settings (~/.make-agent/<slug>/settings.yaml)
  app_dirs.py        # Path helpers (~/.make-agent/)
  templates/         # Bundled orchestra.mk template (copied on first run)
examples/            # Example Makefiles
tests/               # Pytest test suite
```

## Development commands

```bash
uv run pytest                  # Run all tests (281 collected)
uv run pytest --e2e            # Include end-to-end tests (call real LLM API)
uv run ruff check make_agent/  # Lint
uv run ruff format make_agent/ # Format
```

All tests live in `tests/`. End-to-end tests are marked `@pytest.mark.e2e` and skipped by default.

## Key conventions

- **Python 3.11+** required.
- Dependency management via `uv`. The lockfile is `uv.lock`; update it with `uv lock` after changing `pyproject.toml`.
- The project uses `any-llm-sdk` (not litellm) for LLM access.
- Two CLI entry points: `make_agent` (run agent) and `make-agent-create` (YAML → Makefile).
- Per-project data lives in `~/.make-agent/<project-slug>/` — never write to the repo at runtime.
- Ruff is the linter and formatter. Rule `E741` (ambiguous variable names) is ignored.
- Always run `uv run pytest` and `uv run ruff check make_agent/` before finishing a change.
