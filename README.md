# make-agent

An AI agent whose system prompt and tools are defined in a Makefile.

Each Makefile target annotated with a `# <tool>` comment block becomes a callable tool. The agent invokes targets via `make`, passing parameters as `KEY=value` arguments. A `# <system>` block sets the agent's system prompt.

## Installation

```
uv pip install .
```

Requires Python 3.10+ and a working `make` binary. Uses [litellm](https://github.com/BerriAI/litellm) for model access, so any API keys (e.g. `OPENAI_API_KEY`) must be set in the environment.

## Usage

```
OPENAI_API_KEY=<key> uv run make_agent [-f FILE] [--model MODEL] [--prompt PROMPT]
```

- `-f FILE` — Makefile to load (default: `./Makefile`)
- `--model MODEL` — litellm model string (default: `openai/gpt-4o`)
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

## Running tests

```
uv run pytest
```
