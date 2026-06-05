# AGENTS.md

Guidance for AI coding assistants working on this repository.


## Development commands

```bash
uv run pytest                  # Run all tests
uv run pytest --e2e            # Include end-to-end tests (call real LLM API)
uv run ruff check make_agent/  # Lint
uv run ruff format make_agent/ # Format
```

All tests live in `tests/`. End-to-end tests are marked `@pytest.mark.e2e` and skipped by default.

## Key conventions

- **Python 3.11+** required.
- Dependency management via `uv`. The lockfile is `uv.lock`; update it with `uv lock` after changing `pyproject.toml`.
- The project uses Anthropic and OpenAI SDKs (not LiteLLM) for LLM access.
- One CLI entry point: `make_agent` (the `run` subcommand is the default).
- Only one skill mode: `makefile`. Skills are directories containing a `skill.mk` file.
- Per-project data lives in `~/.make-agent/<project-slug>/makefile/` — never write to the repo at runtime.
- Bundled system-prompt template lives in `make_agent/templates/makefile/SYSTEM.md` and is copied to the mode dir on first run.
- `builtin_tool_names(mode)` returns the tool set; only `"makefile"` is a valid mode.
- Ruff is the linter and formatter. Rule `E741` (ambiguous variable names) is ignored.
- Always run `uv run pytest` and `uv run ruff check make_agent/` before finishing a change.
- Always use CONSTANTS name with ALL_CAPS for global constants, no underscores at the constant beginning or end. For example: `MAX_RETRIES = 5`.
