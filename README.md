# make-agent

An AI agent powered by Makefile skills. Skills extend the agent with domain-specific instructions and shell tools.

## Installation

```
pip install makefile-agent
```

Requires Python 3.11+ and a working `make` binary. Uses Anthropic and OpenAI SDKs for model access — set the appropriate provider API key in your environment (for example `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`).

## Usage

```
make_agent [run] --model MODEL [--prompt PROMPT | --prompt-file FILE]
```

| Flag | Default | Description |
|---|---|---|
| `--model MODEL` | — (required) | Model string (e.g. `claude-opus-4-5`, `gpt-4o`) |
| `--skills-dir DIR` | `~/.make-agent/<project>/makefile/skills/` | Directory containing skills |
| `--system PROMPT` | — | System prompt string (overrides SYSTEM.md discovery) |
| `--system-file FILE` | — | Read system prompt from FILE (overrides SYSTEM.md discovery) |
| `--prompt PROMPT` | — | Send a single prompt and exit (non-interactive) |
| `--prompt-file FILE` | — | Read a single prompt from FILE and exit |
| `--disable-builtin-tools TOOLS` | — | Comma-separated built-in tool names to disable, or `all` |
| `--trusted-skills SKILLS` | — | Comma-separated trusted skills or `all`; trusted skills run without confirmation |
| `--max-tool-output CHARS` | 32000 | Truncate tool output; `0` = unlimited |
| `--max-tokens N` | 4096 | Max tokens in the model response |
| `--reasoning-effort EFFORT` | `medium` | `none\|minimal\|low\|medium\|high\|xhigh` |
| `--prompt-cache` | disabled | Enable Anthropic system-prompt caching |
| `--compact-mode MODE` | `drop` | Context compaction strategy when context window is exceeded: `drop` removes oldest turns; `summarize` replaces turns with LLM-generated summaries |
| `--max-retries N` | 5 | Max retries on rate-limit errors |
| `--tool-timeout SECONDS` | 600 | Timeout per tool call |
| `--loglevel LEVEL` | `INFO` | `DEBUG\|INFO\|WARNING\|ERROR\|CRITICAL` |

Without `--prompt`, the agent starts an interactive full-screen REPL. Press Ctrl-D, `/exit`, or `/quit` to leave.

Interactive commands: `/help`, `/export` (save conversation to HTML), `/stats` (token usage totals), `/exit`, `/quit`.

Useful keys: `Alt+Enter` inserts a newline, `Ctrl-C` cancels an in-flight turn, `Ctrl-T` focuses transcript view, then `Ctrl-P`/`Ctrl-N` moves between turns.

## Project data

All per-project data lives under `~/.make-agent/`:

```
~/.make-agent/
└── <project-slug>/          # e.g. Users_alice_proj_myapp
    ├── history              # shell input history
    ├── makefile/
    │   ├── SYSTEM.md        # default system prompt (copied from template on first run)
    │   ├── skills/          # skill directories
    │   └── memory.db        # conversation history
    └── logs/
        └── make-agent.log
```

The **project slug** is the absolute working-directory path with the leading `/` stripped and remaining `/` replaced by `_`.

### System prompt discovery

Priority order (first match wins):

1. `--system PROMPT` flag
2. `--system-file FILE` flag
3. `~/.make-agent/<project>/makefile/SYSTEM.md` (created from a bundled template on first run)

## Skills

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

## Built-in tools

| Tool | What it does |
|---|---|
| `list_skills` | List available skills with descriptions |
| `read_skill` | Return the raw `skill.mk` content |
| `execute_skill(name, command)` | Run a make command against a skill's `skill.mk` |
| `create_skill` | Create or overwrite a skill (single `skill.mk` file) |
| `validate_skill` | Validate that a `skill.mk` exists and has a `DESCRIPTION` block |
| `write_file` | Write content to a file (sandboxed to the current working directory) |
| `edit_file` | Replace a string in a file (sandboxed to the current working directory) |

Use `--disable-builtin-tools` to turn off specific tools (or `all`).

Safety constraints:

- `write_file` and `edit_file` are sandboxed to paths inside the current working directory.
- `execute_skill` blocks dangerous make options (`-f/--file`, `-C/--directory`, `-I/--include-dir`, `--eval`).

## Memory

Every conversation turn is persisted to a local SQLite database (`memory.db`), enabled by default.

The database is created at `~/.make-agent/<project>/makefile/memory.db`.

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

## Context compaction

When the context window is exceeded the agent automatically compacts conversation history and retries. Two strategies are available via `--compact-mode`:

- **`drop`** (default) — removes the oldest turns, keeping the most recent two.
- **`summarize`** — replaces prior turns with LLM-generated summaries, preserving more context at the cost of an extra API call.

Up to 3 compaction attempts are made before the request is aborted.

## Development checks

```
uv run pytest
uv run ruff check make_agent/
uv run ruff format make_agent/
```
