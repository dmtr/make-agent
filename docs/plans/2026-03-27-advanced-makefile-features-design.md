# Advanced Makefile Features for Agent Definition

Extend the make-agent parser to support five GNU Make features that reduce
boilerplate, enable reuse, and make agents more expressive.

## Problem

Agent Makefiles are repetitive.  `codebase-archaeology.mk` has 11 tools that
follow a handful of patterns, yet each repeats the full `# <tool>` /
`# @param` / recipe ceremony.  The orchestrator duplicates a 4-line validation
block across every tool.  There is no way to share tools between agents, no
way to make parameters optional, and no way to conditionally include tools.

## Approach

Extend the existing custom parser (Approach A) incrementally.  Each feature is
an independent PR with its own tests.  The `# <tool>` comment blocks stay —
they express metadata Make cannot — but everything *around* them shrinks.

Implementation order is easiest-to-hardest so value ships early.

---

## Phase 1 — `define` / `endef` (multi-line variables)

### What it enables

- System prompts as real variables instead of comment blocks.
- Reusable recipe fragments (e.g. shared validation logic).

### Syntax

```makefile
define SYSTEM_PROMPT
You are a senior software archaeologist.
You can explore directory trees and read files.
endef

define VALIDATE_NAME
@set -e; \
case "$(NAME)" in \
    ""|*[!A-Za-z0-9._-]*) echo "ERROR: invalid NAME" >&2; exit 2 ;; \
esac
endef
```

### Semantics

- `define VAR` starts collection; `endef` ends it.
- Stored as a `Variable` with `flavor='define'` (recursively expanded).
- `SYSTEM_PROMPT` is a recognized variable name: its value becomes the
  agent's system prompt.  The `# <system>` comment block remains supported
  for backward compatibility but the variable takes precedence.

### Parser change

Add a `DEFINE` state to the state machine.  Collect lines between `define`
and `endef`.  ~30 lines.

### Migration

Replace `# <system>` blocks with `define SYSTEM_PROMPT` in both example
files.  Extract the orchestrator's validation block into `VALIDATE_NAME`.

---

## Phase 2 — Target-specific variables (optional params with defaults)

### What it enables

- Parameters with default values.
- Optional parameters in the JSON schema (not in `required` array).

### Syntax

```makefile
# <tool>
# Show directory tree.
# @param DIR string Root directory
# @param DEPTH integer Maximum depth
# </tool>
show-structure: DIR = .
show-structure: DEPTH = 3
show-structure:
	@find $(DIR) -maxdepth $(DEPTH) -type f | head -200
```

### Semantics

- `TARGET: VAR = value` sets a default for VAR when TARGET runs.
- If a param has a target-specific default, it is **not required** in the
  JSON schema.  The LLM may omit it.
- `run_tool()` only passes params the LLM supplied; Make uses the
  target-specific default for the rest.

### Parser change

Detect `TARGET: VAR = value` (distinguished from prerequisites by the `=`).
Store defaults on the `Rule` dataclass.  ~20 lines.

### Impact on tools.py

`build_tools()` reads defaults from the rule and excludes those params from
`required`.  `run_tool()` filters arguments accordingly.

### Migration

Add defaults to `codebase-archaeology.mk`: `DIR = .`, `DEPTH = 3`,
`LIMIT = 20`, `TARGET = ` (empty = run all tests).

---

## Phase 3 — `include` (composable tool libraries)

### What it enables

- Shared tool definitions across agents.
- A standard library of common tools (`lib/search-tools.mk`,
  `lib/git-tools.mk`, etc.).

### Syntax

```makefile
include lib/search-tools.mk
include lib/git-tools.mk

define SYSTEM_PROMPT
You are a code reviewer.
endef
```

### Semantics

- `include path` reads and parses the file, merging variables and rules
  into the current `Makefile` dataclass.
- Paths are resolved relative to the including file's directory.
- Circular includes are detected and raise an error.
- `-include path` (optional include) silently ignores missing files.

### Parser change

When the parser encounters `include`, call `parse()` recursively on the
included file and merge results.  Track an include stack for cycle
detection.  ~40 lines.

### Impact on create_agent.py

Generated Makefiles can emit `include` directives instead of inlining
common tools.  The YAML spec gains an optional `includes` list:

```yaml
includes:
  - lib/search-tools.mk
system_prompt: "You are a specialist."
tools:
  - name: custom-tool
    ...
```

### Migration

Extract `search-symbol`, `read-file`, and `show-structure` from
`codebase-archaeology.mk` into `lib/` modules.  Rebuild the agent as a
thin file that includes them plus agent-specific tools.

---

## Phase 4 — `ifdef` / `ifeq` conditionals

### What it enables

- Conditional tool availability (e.g. read-only vs. read-write mode).
- Conditional system prompt sections.
- Platform-adaptive recipes.

### Syntax

```makefile
ifdef ENABLE_WRITE_OPS
# <tool>
# Delete files matching pattern.
# @param PATTERN string Glob pattern
# </tool>
delete-files:
	@rm -f $(PATTERN)
endif

ifeq ($(MODE),strict)
SYSTEM_PROMPT += Never modify files without confirmation.
endif
```

### Semantics

- `ifdef VAR` / `ifndef VAR` — test whether a variable is defined.
- `ifeq (a,b)` / `ifneq (a,b)` — test string equality.
- `else` and `endif` work as expected.
- Conditions are evaluated at parse time using current variable state.
- Lines inside a false branch are skipped entirely (not parsed).

### Parser change

Maintain a conditional nesting stack.  On `ifdef`/`ifeq`, push
true/false.  On `else`, flip top of stack.  On `endif`, pop.  Skip
lines when top of stack is false.  ~60 lines.

### Migration

Add `ifdef ENABLE_WRITE_OPS` guard around any dangerous tools in examples.
Add `ifeq` platform check for `search-files` (macOS `mdfind` vs.
Linux `find`).

---

## Phase 5 — `$(call)` + `$(eval)` (tool templates)

### What it enables

- Define a tool pattern once, stamp out N tools from it.
- The single biggest boilerplate reduction.

### Syntax

```makefile
define GREP_TOOL
# <tool>
# Search for patterns in $(2) files.
# @param PATTERN string The pattern to search for
# @param DIR string The directory to search in
# </tool>
search-$(1): DIR = .
search-$(1):
	@grep -rn "$$(PATTERN)" --include="*.$(3)" "$$(DIR)"
endef

$(eval $(call GREP_TOOL,python,Python,py))
$(eval $(call GREP_TOOL,js,JavaScript,js))
$(eval $(call GREP_TOOL,yaml,YAML,yaml))
```

Three tools from one template.

### Semantics

- `$(call VAR,arg1,arg2,...)` expands VAR with `$(1)` → arg1,
  `$(2)` → arg2, etc.  `$$` becomes `$` (deferred expansion).
- `$(eval TEXT)` re-parses TEXT as Makefile content and merges results.

### Parser change

Implement a function expansion engine.  `call` performs positional
substitution.  `eval` invokes `parse()` on the expanded text and merges.
~80 lines — the most complex addition.

### Impact on create_agent.py

Generated Makefiles can reference template libraries:

```makefile
include lib/templates/grep-tools.mk
$(eval $(call GREP_TOOL,python,Python,py))
```

The YAML spec gains a `templates` section:

```yaml
includes:
  - lib/templates/grep-tools.mk
templates:
  - call: GREP_TOOL
    args: [python, Python, py]
  - call: GREP_TOOL
    args: [js, JavaScript, js]
```

### Migration

Refactor `codebase-archaeology.mk` grep-based tools into a template.
Refactor orchestrator CRUD tools into a template with shared validation.

---

## Summary of parser additions

| Phase | Feature | Est. lines | Depends on |
|-------|---------|-----------|------------|
| 1 | `define`/`endef` | ~30 | — |
| 2 | Target-specific variables | ~20 | — |
| 3 | `include` | ~40 | — |
| 4 | `ifdef`/`ifeq` | ~60 | — |
| 5 | `$(call)`+`$(eval)` | ~80 | Phase 1 |

Phases 1–4 are independent and can be done in any order.  Phase 5 depends
on Phase 1 (`define` must work before `call` can reference defined
variables).

## What stays the same

- `# <tool>` / `# @param` / `# </tool>` comment blocks — these express
  metadata (tool descriptions, param types) that Make syntax cannot.
- The tool execution model (`make -f FILE TARGET KEY=value`).
- The YAML spec as the programmatic creation interface.
- `content`-type params and the `_FILE` indirection.

## Out of scope

- `export` directive — useful but lower priority; agents rarely need
  environment propagation beyond what the shell already provides.
- Make string functions (`$(subst)`, `$(patsubst)`, `$(wildcard)`, etc.)
  — can be added later if demand arises, but `call`/`eval` covers most
  use cases.
- Recursive make (`$(MAKE) -f sub.mk`) — already works at the recipe
  level; no parser changes needed.
- Pattern rules (`%.o: %.c`) — not relevant to agent definitions.
