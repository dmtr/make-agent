# make-agent

An AI agent whose system prompt and tools are defined in a Makefile.

Each Makefile target annotated with a `# <tool>` comment block becomes a callable tool. The agent invokes targets via `make`, passing parameters as `KEY=value` arguments. A `define SYSTEM_PROMPT` block sets the agent's system prompt.

## Installation

```
uv pip install .
```

Requires Python 3.10+ and a working `make` binary. Uses [litellm](https://github.com/BerriAI/litellm) for model access, so any API keys (e.g. `ANTHROPIC_API_KEY`) must be set in the environment.

## Usage

```
ANTHROPIC_API_KEY=<key> uv run make_agent [-f FILE] [--model MODEL] [--prompt PROMPT | --prompt-file FILE]
```

- `-f FILE` — Makefile to load (default: `./Makefile`)
- `--model MODEL` — litellm model string (default: `anthropic/claude-haiku-4-5-20251001`)
- `--prompt PROMPT` — send a single prompt and exit instead of entering the interactive shell
- `--prompt-file FILE` — send a single prompt read from `FILE` and exit

Without `--prompt`, the agent starts an interactive REPL. Type `exit`, `quit`, or press Ctrl-D to leave.

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

### Parameters and `$(PARAM_FILE)`

Every declared parameter automatically gets two Make variables injected at call time:

- `$(PARAM)` — the value as a Make variable with shell-escaped value. Convenient for single-line values.
- `$(PARAM_FILE)` — path to a temp file containing the full, unescaped value. Use this for multiline content or when the value might contain shell metacharacters.

```makefile
# <tool>
# Write content to a file.
# @param FILE_PATH string Destination file path
# @param CONTENT string Content to write (may be multiline)
# </tool>
write-file:
	@cat "$(CONTENT_FILE)" > "$(FILE_PATH)"
```

Targets without a `# <tool>` block are invisible to the model.

See `Makefile.sample` for a working example.

## Self-improving agents

`examples/self-improving/` shows an orchestrator pattern where one agent builds and manages a library of specialist agents at runtime.

### How it works

The **orchestrator** (`orchestrator.mk`) exposes four tools to the model:

| Tool | What it does |
|---|---|
| `list-agents` | Scans `./agents/` and returns each specialist's name and purpose |
| `read-agent` | Prints the full Makefile of a named specialist |
| `create-agent` | Writes a YAML spec to a temp file and generates a new `.mk` in `./agents/` |
| `run-agent` | Runs a specialist via `make-agent -f agents/<name>.mk --prompt-file <task-file>` |

For every task the orchestrator follows this loop:

1. Call `list-agents` to discover available specialists.
2. If a suitable specialist exists, delegate with `run-agent`.
3. If none fits, design a new specialist, call `create-agent` to save it, then `run-agent` to execute it.
4. To improve a specialist, call `create-agent` with the same name — it overwrites the previous version.

The `.agents/` directory grows over time into a reusable, self-curated library.

### Running the orchestrator

```bash
cd examples/self-improving

# Interactive session
make-agent -f orchestrator.mk

# Single prompt
printf '%s' "Find all TODO comments in the ../.. directory" > /tmp/task.txt
make-agent -f orchestrator.mk --prompt-file /tmp/task.txt
```

### YAML spec for specialist creation

When the orchestrator designs a new specialist it passes a YAML spec to `create-agent`:

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

`make-agent-create` converts this spec into a standard `make-agent` Makefile and saves it to `agents/<name>.mk`.

## Running tests

```
uv run pytest
```
