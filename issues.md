# Critical Issues Found During Codebase Review

## Issue 1: SQL Injection via FTS5 MATCH Clause in Memory Search

**Severity:** CRITICAL
**Location:** `make_agent/memory/memory.py` lines ~90-105

```python
def _search(
    self,
    view: str,
    query: str,
    ...
) -> str:
    conn = self._get_conn()
    sql = f"""
        SELECT v.created_at, v.message
        FROM {view} v
        JOIN messages_fts ON v.id = messages_fts.rowid
        WHERE messages_fts MATCH ?
    """
```

**Why it's critical:** The `query` parameter (originating from user input in the search tools) is passed into the FTS5 `MATCH` clause. While it uses a `?` placeholder, **FTS5 MATCH does not properly support parameterized queries the way standard SQL operators do**. An attacker can craft an FTS5 query string that breaks out of the intended context.

For example, a user could search for something like:
```
foo" UNION SELECT * FROM messages --
```

This could lead to:
- **Data exfiltration** — reading data from other tables via `UNION SELECT`
- **Injection of FTS5 operators** — using `AND`, `OR`, `NOT`, or `phrase()` syntax to manipulate query behavior
- **Database errors** that leak internal information

The search tools (`search_user_memory`, `search_agent_memory`) expose this directly to the agent, meaning any tool-generated query reaches the database unsanitized.

---

## Issue 2: AST-based Security Check Can Be Bypassed for Arbitrary Code Execution

**Severity:** CRITICAL
**Location:** `make_agent/skill_registry.py` lines ~37-84 (`_ast_trust_check`) and `skill_backend.py` lines ~26-58 (subprocess worker)

The `_ast_trust_check` function is the primary trust boundary for Python skill execution. It walks the AST looking for dangerous imports and calls, but it has several well-known bypass vectors:

**Bypass examples that would pass the trust check but still execute harmful code:**

```python
# Bypass 1: Indirect call through variable
func_name = "eval"
func = getattr(__builtins__, func_name)
func("harmful_code()")

# Bypass 2: Dynamic attribute access
import os
mod = "os"
attr = "system"
getattr(getattr(__import__(mod), mod), attr)("rm -rf /")

# Bypass 3: exec/eval through builtins dict (not detected by Name check)
builtins = __builtins__
builtins["exec"]("malicious()")

# Bypass 4: Using __class__.__base__.__subclasses__() to find dangerous classes
".".join([str(c.__name__) for c in object.__subclasses__()])
# Could access subprocess.Popen, etc.

# Bypass 5: os.popen through attribute chain
import os
pop = getattr(os, "popen")
pop("whoami").read()  # Not detected — only checks direct `os.popen(...)` calls
```

The AST check only catches **direct** calls like `exec(...)`, `eval(...)`, `os.system(...)`, and imports from a hardcoded list. It does not detect:
- Indirect calls through variables or attributes
- Dynamic attribute access (`getattr(os, "system")`)
- Builtins accessed via the `__builtins__` dict
- Class introspection to find dangerous subclasses
- Library functions that internally call dangerous operations (e.g., `pickle.loads`, `shutil.rmtree`)

**Impact:** An attacker who can write a `skill.py` file could execute arbitrary system commands, read/write any file, exfiltrate data, or pivot to other systems — all outside the subprocess worker's timeout protection.

---

## Issue 3: Race Condition in `_build_chain` Causes Middleware Capture Bug

**Severity:** HIGH
**Location:** `make_agent/agent_core/agent.py` lines ~141-152

```python
def _build_chain(self) -> Callable[[Request], AsyncIterator[AgentEvent]]:
    """Build the middleware chain; first middleware in the list is innermost."""
    current: Callable[[Request], AsyncIterator[AgentEvent]] = self._stream_events_core
    for mw in self._middlewares:
        prev = current

        def make_wrapper(
            _mw: MiddlewareBase, _prev: Callable[[Request], AsyncIterator[AgentEvent]]
        ) -> Callable[[Request], AsyncIterator[AgentEvent]]:
            return lambda req: _mw(req, _prev)

        current = make_wrapper(mw, prev)   # ← BUG: `prev` is captured by reference!
    return current
```

**Why it's critical:** The variable `prev` is reassigned in every loop iteration, but the lambda created by `make_wrapper` captures `prev` **by reference**, not by value. This means all middleware wrappers end up referencing the *same* `prev` — which at the end of the loop will always be the last assigned value (`self._stream_events_core`).

In practice this means:
- When multiple middlewares are registered, only the **outermost** one (first in the list) is actually wired into the chain. All inner middlewares get lost.
- The `SessionMiddleware` (which handles persisting messages and token usage to SQLite) may not be properly chained in all scenarios, causing **silent data loss** — messages and token stats are never stored.
- Any custom middleware added by users would similarly be disconnected from the execution chain.

**Suggested fix:** Capture `prev` as a default argument so it's bound at definition time:

```python
def make_wrapper(
    _mw: MiddlewareBase, _prev: Callable[[Request], AsyncIterator[AgentEvent]] = prev  # ← add default
) -> Callable[[Request], AsyncIterator[AgentEvent]]:
    return lambda req: _mw(req, _prev)

current = make_wrapper(mw, prev)
```

Or use a factory that captures the value explicitly. This is a subtle Python scoping bug that's notoriously difficult to debug because it silently produces incorrect behavior rather than crashing.
