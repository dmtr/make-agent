# Self-Improving Agents

An orchestrator agent that builds and maintains a library of specialist agents at runtime.

## How it works

The **orchestrator** (`orchestrator.mk`) has four tools:

| Tool | What it does |
|---|---|
| `list-agents` | Scans `./agents/` and returns each agent's name + purpose |
| `read-agent` | Prints the full Makefile of a named agent |
| `create-agent` | Writes a new `.mk` file to `./agents/<name>.mk` |
| `run-agent` | Runs a specialist via `make-agent -f agents/<name>.mk --prompt "..."` |

When given a task the orchestrator will:
1. Discover existing specialists via `list-agents`
2. Delegate to an existing specialist if one fits, or
3. Design and create a new specialist, then run it
4. To improve a specialist, call `create-agent` with the same name — it overwrites

The `agents/` directory grows over time into a reusable library of specialists.

## Usage

```bash
cd examples/self-improving

# Interactive session
make-agent -f orchestrator.mk

# Single prompt
make-agent -f orchestrator.mk --prompt "Find all TODO comments in the ../.. directory"
```

## Writing specialist agents

Specialists are standard `make-agent` Makefiles. Example:

```makefile
# <system>
# You are a specialist that searches source code for patterns.
# </system>

.PHONY: search-files

# <tool>
# Search files for a text pattern and return matching lines.
# @param PATTERN string The text pattern to search for
# @param DIR string The directory to search in
# </tool>
search-files:
	@grep -rn "$(PATTERN)" "$(DIR)" || echo "No matches found"
```

The orchestrator will create files like this automatically when needed.

## Directory layout

```
examples/self-improving/
  orchestrator.mk     ← the orchestrator
  agents/             ← specialist library (created at runtime)
    *.mk              ← specialist agents (created by the orchestrator)
  README.md           ← this file
```
