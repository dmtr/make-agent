# AGENTS.md

Guidance for AI coding assistants working on this repository.


## Development commands

```bash
uv run pytest                  # Run all tests (265 collected)
uv run pytest --e2e            # Include end-to-end tests (call real LLM API)
uv run ruff check make_agent/  # Lint
uv run ruff format make_agent/ # Format
```

All tests live in `tests/`. End-to-end tests are marked `@pytest.mark.e2e` and skipped by default.

## Key conventions

- **Python 3.11+** required.
- Dependency management via `uv`. The lockfile is `uv.lock`; update it with `uv lock` after changing `pyproject.toml`.
- The project uses `litellm` (not any-llm-sdk) for LLM access.
- One CLI entry point: `make_agent` (the `run` subcommand is the default).
- Two skill modes: `python` (default) and `makefile`, selected via `--skill-mode`.
- Per-project data lives in `~/.make-agent/<project-slug>/<mode>/` — never write to the repo at runtime.
- Bundled system-prompt templates live in `make_agent/templates/<mode>/SYSTEM.md` and are copied to the mode dir on first run.
- `builtin_tool_names(mode)` returns the tool set for a given mode; file tools (`write_file`, `edit_file`) are makefile-only.
- Ruff is the linter and formatter. Rule `E741` (ambiguous variable names) is ignored.
- Always run `uv run pytest` and `uv run ruff check make_agent/` before finishing a change.
- Always use CONSTANTS name with ALL_CAPS for global constants, no underscores at the constant beginning or end. For example: `MAX_RETRIES = 5`.

