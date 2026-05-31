# Test Failure Analysis and Suggested Fixes

## Summary of Failures
The test suite failed with 12 errors across two main areas: Shell State Management and Security AST Analysis.

---

## 1. Shell State Management (`tests/test_agent_shell.py`)
**Issue:** `AgentStatus.STREAMING` is not being correctly restored or maintained during specific lifecycle events, leading to incorrect status reporting (e.g., showing `IDLE` when it should be `STREAMING`).

### Suggested Fixes:
* **Handle `HistoryCompacted` event:** In the logic handling history compaction, add a check to ensure that if a turn is currently active, the shell status is explicitly set to/maintained as `STREAMING`.
* **Event-Driven Status Restoration:** Update the event consumer (`_consume_turn_events`) so that `TurnStarted` and `StatusChanged(is_busy=True)` events trigger an automatic transition back to `STREAMING`.
* **Robust State Machine:** Implement state guards to prevent the status from dropping to `IDLE` until a terminal turn event is received.

---

## 2. Security AST Analysis (`tests/test_skill_registry.py`)
**Issue:** The `_ast_trust_check` function fails to detect several obfuscation techniques, allowing potentially dangerous code to be marked as "trusted".

### Suggested Fixes:
* **Implement Symbol Tracking:** Track assignments of dangerous functions (e.g., `x = eval`). If a call is made using a name found in this "tainted" set, mark the code as untrusted.
* **Strict `getattr` Validation:** Flag any `getattr(obj, attr)` call where `attr` is not a constant string literal.
* **Blacklist Dunder Methods:** Explicitly block access to sensitive dunder attributes like `__subclasses__`, `__getattribute__`, and `__import__`.
* **Detect String Obfuscation:** Flag non-constant string expressions (like concatenation) used in function calls or attribute lookups.
* **Chain Analysis:** Ensure the analyzer inspects entire attribute chains to catch nested dangerous calls like `os.__getattribute__('system')`.
