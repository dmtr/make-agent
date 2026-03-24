# Replace JSON with YAML for Agent Spec Format

## Problem

LLMs struggle to generate valid JSON agent specs when using `make-agent-create`,
particularly in `orchestrator.mk`. The main pain points are:

- Escaping quotes inside `recipe` shell commands (e.g. `\"$(VAR)\"` in JSON)
- Nested structures with mixed quoting (`{`, `[`, `"` delimiters all in play)
- No inline comments to guide generation

YAML eliminates all three: shell commands are plain strings, structure uses
indentation, and `#` comments are native.

## Approach

Replace JSON parsing with YAML parsing throughout the agent spec pipeline.
The internal `spec` dict structure is **unchanged** — only the wire format changes.

This is a clean break: JSON support is removed entirely (no dual-format fallback).

## Spec Format Comparison

**Before (JSON):**
```json
{
  "system_prompt": "You are a filesystem assistant.",
  "tools": [
    {
      "name": "search-files",
      "description": "Search files for a pattern.",
      "params": [
        {"name": "PATTERN", "type": "string", "description": "Search pattern (regex)"},
        {"name": "DIR",     "type": "string", "description": "Directory to search in"}
      ],
      "recipe": ["@grep -rn \"$(PATTERN)\" \"$(DIR)\" || echo \"No matches found\""]
    }
  ]
}
```

**After (YAML):**
```yaml
system_prompt: "You are a filesystem assistant."
tools:
  - name: search-files
    description: Search files for a pattern.
    params:
      - name: PATTERN
        type: string
        description: Search pattern (regex)
      - name: DIR
        type: string
        description: Directory to search in
    recipe:
      - '@grep -rn "$(PATTERN)" "$(DIR)" || echo "No matches found"'
```

No backslash escaping needed in recipe lines.

## Implementation Scope

### 1. `pyproject.toml`
Add `pyyaml>=6` to `[project.dependencies]`.

### 2. `make_agent/create_agent.py`
- `import yaml` (remove `import json`)
- Replace `json.loads(raw)` → `yaml.safe_load(raw)`
- Update `--spec` CLI help: "Agent spec as a YAML string"
- Update `--file` CLI help: "Path to YAML spec file"
- Update `render()` docstring to show YAML spec format

### 3. `tests/test_create_agent.py`
- Replace `json.dumps(spec)` → `yaml.dump(spec)` in CLI-level tests
- Update any inline JSON strings to YAML

### 4. `examples/self-improving/orchestrator.mk`
Two sub-changes:

**System prompt (lines 14–44):** Replace the JSON schema description and examples
with YAML equivalents so the LLM knows to emit YAML.

**`create-agent` tool:**
- Rename tmpfile from `.json` → `.yaml`
- Update `SPEC` param description from "JSON agent spec" to "YAML agent spec"

### 5. `README.md` / docs
Update any JSON spec examples to YAML.

## YAML Schema (unchanged fields, new format)

```yaml
system_prompt: "..."      # required; string
tools:                     # required; list of tool objects
  - name: tool-name        # required; string (used as Makefile target)
    description: "..."     # required; string
    params:                # optional; omit for tools with no arguments
      - name: PARAM_NAME   # required; string (used as $(PARAM_NAME) in recipe)
        type: string       # required; one of: string, number, integer, boolean
        description: "..." # required; string
    recipe:                # required; list of shell command strings
      - "@command $(PARAM_NAME)"
```

**Constraint:** every `params[].name` must appear as `$(NAME)` in at least one
`recipe` line (validated by `render()` — unchanged).

## Dependencies

- Add: `pyyaml>=6`
- Remove: `json` stdlib (no longer used in `create_agent.py`)

## Error Handling

`yaml.safe_load` raises `yaml.YAMLError` on invalid input. This propagates
naturally to the caller (same behavior as the current `json.JSONDecodeError`).
No explicit catch needed unless we want a friendlier error message.

## Testing

Existing 47 tests in `test_create_agent.py` cover the full spec surface.
They require only mechanical changes (`json.dumps` → `yaml.dump`, inline strings
updated) — no new test cases needed, no logic changes.
