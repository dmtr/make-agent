# make-agent

An AI agent whose system prompt and tools are defined in a Makefile.

Each Makefile target annotated with a `# <tool>` comment block becomes a callable tool. The agent invokes targets via `make`, passing parameters as `KEY=value` arguments. A `# <system>` block sets the agent's system prompt.

## Installation

```
uv pip install .
```

Requires Python 3.10+ and a working `make` binary. Uses [litellm](https://github.com/BerriAI/litellm) for model access, so any API keys (e.g. `ANTHROPIC_API_KEY`) must be set in the environment.

## Usage

```
ANTHROPIC_API_KEY=<key> uv run make_agent [-f FILE] [--model MODEL] [--prompt PROMPT]
```

- `-f FILE` — Makefile to load (default: `./Makefile`)
- `--model MODEL` — litellm model string (default: `anthropic/claude-haiku-4-5-20251001`)
- `--prompt PROMPT` — send a single prompt and exit instead of entering the interactive shell

Without `--prompt`, the agent starts an interactive REPL. Type `exit`, `quit`, or press Ctrl-D to leave.

## Makefile format

```makefile
# <system>
# You are a filesystem assistant.
# </system>

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

- `# <system> ... # </system>` sets the system prompt passed to the model.
- `# <tool> ... # </tool>` marks the following target as an LLM-callable tool. Lines starting with `# @param NAME type description` declare parameters (JSON Schema primitives: `string`, `number`, `integer`, `boolean`). All other lines form the tool description.

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
| `run-agent` | Runs a specialist via `make-agent -f agents/<name>.mk --prompt "..."` |

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
make-agent -f orchestrator.mk --prompt "Find all TODO comments in the ../.. directory"
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
