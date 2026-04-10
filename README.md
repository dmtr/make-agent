# make-agent

An AI agent whose system prompt and tools are defined in a Makefile.

Each Makefile target annotated with a `# <tool>` comment block becomes a callable tool. The agent invokes targets via `make`, passing parameters as `KEY=value` arguments. A `define SYSTEM_PROMPT` block sets the agent's system prompt.

## Installation

```
pip install makefile-agent
```

Requires Python 3.11+ and a working `make` binary. Uses [any-llm-sdk](https://pypi.org/project/any-llm-sdk/) for model access, so any API keys (e.g. `ANTHROPIC_API_KEY`) must be set in the environment.

## Usage

```
ANTHROPIC_API_KEY=<key> uv run make_agent [run] [-f FILE] [--model MODEL] [--prompt PROMPT | --prompt-file FILE] [--with-memory]
```

- `-f FILE` — Makefile to load. Searched in the current directory first, then `~/.make-agent/<project>/agents/`. Defaults to the value in `settings.yaml`, or `./Makefile` if not set.
- `--model MODEL` — any-llm model string. Defaults to the value in `settings.yaml`.
- `--prompt PROMPT` — send a single prompt and exit instead of entering the interactive shell
- `--prompt-file FILE` — send a single prompt read from `FILE` and exit
- `--agents-dir DIR` — directory for specialist `.mk` files (default: `~/.make-agent/<project>/agents/`)
- `--agent-model MODEL` — model used when running specialist agents via `run_agent` (default: same as `--model`)
- `--with-memory` — enable persistent conversation memory (see [Memory](#memory))
- `--disable-builtin-tools TOOLS` — comma-separated built-in tool names to disable, or `all`
- `--max-tool-output CHARS` — truncate tool stdout to this many characters; `0` = unlimited (default: 20000)
- `--max-tokens N` — max tokens in the model response (default: 4096)
- `--max-retries N` — max retry attempts on rate limit errors (default: 5)
- `--tool-timeout SECONDS` — timeout for each tool subprocess (default: 600)
- `--debug` — write all messages to `~/.make-agent/<project>/logs/make-agent.log`

Without `--prompt`, the agent starts an interactive REPL. Type `exit`, `quit`, or press Ctrl-D to leave.

### First run — setup wizard

If no `settings.yaml` exists for the project and no Makefile is found automatically, the agent prompts you to create one:

```
No settings.yaml found for this project.
Let's create one. Press Enter to accept the default shown in brackets.

  Makefile path [Makefile]: ./my-agent.mk
  Model [anthropic/claude-haiku-4-5-20251001]:

Saved settings to ~/.make-agent/Users_alice_proj_myapp/settings.yaml
```

## Project settings

All per-project data is stored under `~/.make-agent/`:

```
~/.make-agent/
└── <project-slug>/          # e.g. Users_alice_proj_myapp
    ├── settings.yaml        # default model and Makefile
    ├── memory.db            # conversation history (when memory is enabled)
    ├── agents/              # specialist agent .mk files
    └── logs/
        └── make-agent.log   # debug log (written when --debug is set)
```

The **project slug** is the absolute path of the working directory with the leading `/` stripped and remaining `/` replaced by `_`.

### settings.yaml

```yaml
model: anthropic/claude-haiku-4-5-20251001
makefile: ./my-agent.mk
memory: true          # optional — enable persistent memory
agent_model: anthropic/claude-opus-4-5  # optional — model for run_agent calls
```

All fields are optional. CLI flags always take precedence over `settings.yaml` values.

## Makefile format

```makefile
define SYSTEM_PROMPT
You are a filesystem assistant.
endef

.PHONY: list-files greet

# <tool>
# List files in a directory.
# @param DIR string The directory path to list
# </tool>
list-files:
	@ls -la $(DIR)

# <tool>
# Greet someone.
# @param NAME string The name to greet
# </tool>
greet:
	@echo "Hello, $(NAME)!"
```

### Special comment blocks

- `define SYSTEM_PROMPT ... endef` sets the system prompt passed to the model. The content is raw text — no `#` prefix needed. `endef` must be on its own line with no indentation.
- `# <tool> ... # </tool>` marks the following target as an LLM-callable tool. Lines starting with `# @param NAME type description` declare parameters (JSON Schema primitives: `string`, `number`, `integer`, `boolean`). All other lines form the tool description.

### Parameters and `$$PARAM`

Every declared parameter is injected as an environment variable.  Recipes
access it with shell syntax — `$$PARAM` in a Make recipe becomes `$PARAM` for
the shell:

- `$$PARAM` — canonical form, works for both single-line and multiline values.
- `$(PARAM)` — also works for simple single-line values (Make auto-imports env vars).

```makefile
# <tool>
# Write content to a file.
# @param FILE_PATH string Destination file path
# @param CONTENT string Content to write (may be multiline)
# </tool>
write-file:
	@printf '%s' "$$CONTENT" > "$$FILE_PATH"
```

Targets without a `# <tool>` block are invisible to the model.

## Built-in tools

Every agent automatically receives four built-in tools alongside its Makefile-defined tools — no Makefile declaration needed:

| Tool | What it does |
|---|---|
| `list_agent` | Scan the agents directory and return each specialist's name and description |
| `validate_agent` | Parse and validate a named specialist's Makefile, reporting any errors |
| `create_agent` | Generate a new `.mk` file from a YAML spec and save it to the agents directory |
| `run_agent` | Delegate a task to a specialist agent and return its output |

The agents directory defaults to `~/.make-agent/<project>/agents/` and can be changed with `--agents-dir`.

## Memory

Agents can persist every conversation turn to a local SQLite database and search it in future sessions.

### Enabling memory

```bash
# One-time flag
make_agent --with-memory -f my-agent.mk

# Always on for this project (settings.yaml)
memory: true
```

The database is stored at `~/.make-agent/<project-slug>/memory.db`. Every user message and final agent reply is written automatically.

The project directory layout becomes:

```
~/.make-agent/
└── <project-slug>/
    ├── settings.yaml
    ├── memory.db        # conversation history (created on first use)
    ├── agents/
    └── logs/
```

### Memory tools

When memory is enabled, three additional built-in tools are injected:

| Tool | What it does |
|---|---|
| `get_recent_messages(limit)` | Return the N most recent messages in chronological order |
| `search_user_memory(query, limit, from_date, to_date)` | FTS5 keyword search over past user messages |
| `search_agent_memory(query, limit, from_date, to_date)` | FTS5 keyword search over past agent replies |

**FTS5 search tips** — the search is keyword-based, not semantic:

- Use short keywords, not sentences: `"goal project"` not `"what is the goal of this project"`
- Use `OR` for broader recall: `"goal OR objective OR purpose"`
- Stop words (`the`, `of`, `is`, `a`) are not indexed — omit them
- If a search returns nothing, retry with broader or alternative keywords
- Use `get_recent_messages` when you need recent context and don't know which keywords to search for

### Orchestrator pattern

`examples/orchestra.mk` shows how to use the built-in tools to build a self-managing agent that creates and improves specialist agents at runtime:

```bash
# Interactive session
make_agent -f examples/orchestra.mk

# Single prompt
make_agent -f examples/orchestra.mk --prompt "Summarise the git log for the last week"
```

For every task the orchestrator:

1. Calls `list_agent` to discover available specialists.
2. Delegates to an existing specialist with `run_agent`, or designs and saves a new one with `create_agent` first.
3. Improves any specialist by calling `create_agent` with the same name — it overwrites the previous version.

### YAML spec for `create_agent`

```yaml
system_prompt: "You are a specialist that searches source code for patterns."
tools:
  - name: search-files
    description: Search files for a text pattern and return matching lines.
    params:
      - name: PATTERN
        type: string
        description: The text pattern to search for
      - name: DIR
        type: string
        description: The directory to search in
    recipe:
      - '@grep -rn "$(PATTERN)" "$(DIR)" || echo "No matches found"'
```

`make-agent-create` converts this spec into a standard `make-agent` Makefile.

## Example

`examples/orchestra.mk` is the orchestrator agent — it manages specialist agents using the built-in tools and also exposes three simple no-parameter tools for system information:

```bash
make_agent -f examples/orchestra.mk
```

| Tool | Recipe |
|---|---|
| `current-dir` | `pwd` |
| `os-info` | `uname -a` |
| `current-date` | `date` |

## Running tests

```
uv run pytest
```
