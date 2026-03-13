# make-agent design

## Problem

Build an AI agent that reads its system prompt and tool definitions from a
Makefile, and uses `make <target> KEY=val …` to invoke tools.

## Architecture

```
Makefile (on disk)
    ↓  parse_file()
Makefile dataclass  ──→  system prompt + rules with @params
    ↓  build_tools()
Tool schemas (OpenAI JSON Schema)
    ↓  litellm.completion()
LLM  ←→  Agent loop  ←→  User REPL
    ↓  tool call dispatch
make -f <path> <target> KEY=val  →  stdout  →  tool result
```

### New / changed files

| File | Change |
|------|--------|
| `src/parser.py` | Add `Param` dataclass; parse `@param` lines from `# <tool>` blocks |
| `src/tools.py` | Build LLM tool schemas from rules; execute `make` subprocess |
| `src/agent.py` | Interactive REPL loop with litellm + tool dispatch |
| `main.py` | CLI entry point with `-f/--file` and `--model` flags |

---

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
	ls $(DIR)

# <tool>
# Greet someone.
# @param NAME string The name to greet
# @param GREETING string The greeting word to use
# </tool>
greet:
	echo "$(GREETING), $(NAME)!"

# No <tool> block → not exposed to the LLM
clean:
	rm -rf dist/
```

### `@param` syntax

```
# @param <NAME> <type> <description>
```

- `NAME` — Make variable name, passed as `NAME=value` on the command line.
- `type` — JSON Schema primitive type: `string`, `number`, `boolean`, `integer`.
- `description` — Free-form text sent to the LLM.
- All declared params are **required**.
- Non-`@param` lines in the block become the tool description.
- Rules without a `# <tool>` block are not exposed to the LLM.

---

## Parser changes (`src/parser.py`)

Add a `Param` dataclass:

```python
@dataclass
class Param:
    name: str
    type: str          # JSON Schema primitive
    description: str
```

Update `Rule`:

```python
@dataclass
class Rule:
    ...
    description: str | None = None   # non-@param lines only
    params: list[Param] = field(default_factory=list)
```

Parse `@param` lines inside `_State.TOOL_BLOCK` with:

```
^@param\s+(\w+)\s+(\w+)\s+(.+)
```

---

## Tool adapter (`src/tools.py`)

### Schema builder

For each rule with a description, produce an OpenAI function tool dict:

```python
{
    "type": "function",
    "function": {
        "name": rule.target,          # e.g. "greet"
        "description": rule.description,
        "parameters": {
            "type": "object",
            "properties": {
                p.name: {"type": p.type, "description": p.description}
                for p in rule.params
            },
            "required": [p.name for p in rule.params],
        },
    },
}
```

### Executor

```python
def run_tool(target: str, arguments: dict[str, str], makefile: Path) -> str:
    args = ["make", "-f", str(makefile), target] + [f"{k}={v}" for k, v in arguments.items()]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        return f"Error (exit {result.returncode}):\n{result.stderr}"
    return result.stdout
```

---

## Agent loop (`src/agent.py`)

```
1. parse_file(path) → system prompt + tool schemas
2. print prompt "make-agent> ", read user input (Ctrl-D to exit)
3. append user message to history
4. inner loop:
   a. litellm.completion(model, messages=history, tools=schemas)
   b. if tool_calls in response:
        for each tool call:
          run_tool(name, args, makefile)
          append tool result message to history
      go to 4a
   c. else:
        print assistant content
        append assistant message to history
        break
5. go to 2
```

---

## CLI (`main.py`)

```
usage: make-agent [-f FILE] [--model MODEL]

options:
  -f, --file FILE    Makefile to load (default: ./Makefile)
  --model MODEL      LLM model string for litellm (default: openai/gpt-4o)
```
