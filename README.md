# make-agent

An AI agent powered by skills. Skills extend the agent with domain-specific instructions and optional tools. Two skill modes are supported: **python** (default) and **makefile**.

## Installation

```
pip install makefile-agent
```

Requires Python 3.11+ and a working `make` binary (for makefile mode). Uses [LiteLLM](https://github.com/BerriAI/litellm) for model access — set the appropriate provider API key in your environment (for example `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`).

## Usage

```
make_agent [run] --model MODEL [--skill-mode MODE] [--prompt PROMPT | --prompt-file FILE]
```

| Flag | Default | Description |
|---|---|---|
| `--model MODEL` | — (required) | LiteLLM model string |
| `--skill-mode MODE` | `python` | Skill backend: `python` or `makefile` |
| `--skills-dir DIR` | `~/.make-agent/<project>/<mode>/skills/` | Directory containing skills |
| `--system PROMPT` | — | System prompt string (overrides SYSTEM.md discovery) |
| `--system-file FILE` | — | Read system prompt from FILE (overrides SYSTEM.md discovery) |
| `--prompt PROMPT` | — | Send a single prompt and exit (non-interactive) |
| `--prompt-file FILE` | — | Read a single prompt from FILE and exit |
| `--disable-builtin-tools TOOLS` | — | Comma-separated built-in tool names to disable, or `all` |
| `--trusted-skills SKILLS` | — | Comma-separated trusted skills, `skill.target`, or `all`; trusted skills run without confirmation |
| `--max-tool-output CHARS` | 32000 | Truncate tool output; `0` = unlimited |
| `--max-tokens N` | 4096 | Max tokens in the model response |
| `--reasoning-effort EFFORT` | `high` | `none\|minimal\|low\|medium\|high\|xhigh` |
| `--prompt-cache` | disabled | Enable Anthropic system-prompt caching |
| `--compact-context-window TOKENS` | 0 | Known context window used for adaptive threshold (`0` = unknown) |
| `--compact-threshold-ratio RATIO` | 0.75 | Auto-compact trigger as a fraction of context window |
| `--compact-target-ratio RATIO` | 0.5 | Hysteresis lower bound after compaction |
| `--max-retries N` | 5 | Max retries on rate-limit errors |
| `--tool-timeout SECONDS` | 600 | Timeout per tool call |
| `--loglevel LEVEL` | `INFO` | `DEBUG\|INFO\|WARNING\|ERROR\|CRITICAL` |

Without `--prompt`, the agent starts an interactive full-screen REPL. Press Ctrl-D, `/exit`, or `/quit` to leave.

Interactive commands: `/help`, `/export` (save conversation to HTML), `/stats` (token totals), `/exit`, `/quit`.

Useful keys: `Alt+Enter` inserts a newline, `Ctrl-C` cancels an in-flight turn, `Ctrl-T` focuses transcript view, then `Ctrl-P`/`Ctrl-N` moves between turns.

## Project data

All per-project data lives under `~/.make-agent/`:

```
~/.make-agent/
└── <project-slug>/          # e.g. Users_alice_proj_myapp
    ├── history              # shell input history
    ├── python/
    │   ├── SYSTEM.md        # default system prompt (copied from template on first run)
    │   ├── skills/          # skill directories
    │   └── memory.db        # conversation history
    ├── makefile/
    │   ├── SYSTEM.md
    │   ├── skills/
    │   └── memory.db
    └── logs/
        └── make-agent.log
```

The **project slug** is the absolute working-directory path with the leading `/` stripped and remaining `/` replaced by `_`.

### System prompt discovery

Priority order (first match wins):

1. `--system PROMPT` flag
2. `--system-file FILE` flag
3. `SYSTEM.md` in the current working directory
4. `~/.make-agent/<project>/<mode>/SYSTEM.md` (created from a bundled template on first run)

## Skill modes

### Python mode (default)

Each skill is a directory containing:

- **`skill.md`** — instructions the agent reads before acting (required)
- **`skill.py`** — optional Python module with `@target`-decorated tool functions

```python
# skill.py
from make_agent import target

@target
def search_files(pattern: str, directory: str) -> str:
    """Search files for a pattern and return matching lines.

    :param pattern: Text pattern to search for
    :param directory: Directory to search in
    """
    import subprocess
    result = subprocess.run(["grep", "-rn", pattern, directory], capture_output=True, text=True)
    return result.stdout or "No matches found"
```

```markdown
<!-- skill.md -->
---
description: "Searches files for a text pattern."
---

# File Search

Use this skill to search file contents.

## Available tools

- `search_files(pattern, directory)` — search recursively for a pattern
```

Rules:
- Every tool function must be decorated with `@target`.
- Annotate parameters with Python type hints (`str`, `int`, `float`, `bool`).
- Document parameters with `:param name: description` in the docstring.
- Functions must be synchronous.
- Call `validate_skill` after creating a skill with tools — it runs syntax + AST trust checks and reports whether the skill is trusted.

### Makefile mode

Each skill is a directory containing a single **`skill.mk`** file.

```makefile
define DESCRIPTION
Searches files for a text pattern.
endef

.PHONY: search-files

search-files:
	@grep -rn "$$PATTERN" "$$DIR" || echo "No matches found"
```

The agent invokes targets via `make`, passing parameters as environment variables (`$$PARAM` in a recipe becomes `$PARAM` for the shell). The `define DESCRIPTION … endef` block is required and shown by `list_skills`.

### Skill trust and approvals

`execute_skill` requires confirmation when a skill is not trusted. In interactive mode, the shell prompts `[Y] approve / [N] deny`.

Trust sources:

- `--trusted-skills all` trusts all skills.
- `--trusted-skills skill_name` trusts all targets in that skill.
- `--trusted-skills skill_name.target_name` trusts one Python target.
- Python skills can also be auto-trusted when AST trust checks pass.

## Built-in tools

All modes include these skill management tools:

| Tool | What it does |
|---|---|
| `list_skills` | List available skills with descriptions |
| `read_skill` | Return a skill's instructions |
| `execute_skill` | Run a tool from a skill |
| `create_skill` | Create or overwrite a skill |
| `validate_skill` | Validate a skill (python: `skill.md` + `skill.py` syntax/trust; makefile: structure check) |

Makefile mode additionally includes:

| Tool | What it does |
|---|---|
| `write_file` | Write content to a file |
| `edit_file` | Replace a string in a file |

Use `--disable-builtin-tools` to turn off specific tools (or `all`).

Safety constraints:

- `write_file` and `edit_file` are sandboxed to paths inside the current working directory.
- Makefile `execute_skill` blocks dangerous make options (`-f/--file`, `-C/--directory`, `-I/--include-dir`, `--eval`).

## Memory

Every conversation turn can be persisted to a local SQLite database (`memory.db`) and searched in future sessions.

Memory is always on by default — the database is created per mode at `~/.make-agent/<project>/<mode>/memory.db`.

Three additional built-in tools are available:

| Tool | What it does |
|---|---|
| `get_recent_messages(limit, from_date, to_date)` | Return recent messages in chronological order |
| `search_user_memory(query, limit, from_date, to_date)` | FTS5 keyword search over past user messages |
| `search_agent_memory(query, limit, from_date, to_date)` | FTS5 keyword search over past agent replies |

**FTS5 search tips** — the search is keyword-based, not semantic:

- Use short keywords: `"goal project"` not `"what is the goal of this project"`
- Use `OR` for broader recall: `"goal OR objective OR purpose"`
- Stop words (`the`, `of`, `is`, `a`) are not indexed — omit them
- Fall back to `get_recent_messages` when you don't know which keywords to search for

## Development checks

```
uv run pytest
uv run pytest --e2e
uv run ruff check make_agent/
uv run ruff format make_agent/
```
