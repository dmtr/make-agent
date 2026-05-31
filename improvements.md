# Improvements for Auto-Compact Implementation

Following the technical review of the `AgenticLoop` implementation in `make_agent/agent_core/loop.py`, the following three improvements are suggested to enhance robustness, maintainability, and intelligence.

## 1. Refactor the Monolithic `_run` Method
The current `_run` method is a "God Method"—it handles streaming chunks, manages context overflow/retries, processes tool call sequences, tracks consecutive failures, and manages conversation state. This makes it difficult to unit test specific behaviors (like how tool errors are handled) without simulating an entire LLM stream.

**Proposed Changes:**
- Decompose `_run` into smaller, private helper methods:
    - `_stream_provider_response(...)`: Encapsulates the provider's async streaming logic and returns a structured result (content, tool calls, usage).
    - `_handle_context_overflow(...)`: Isolates the snapshot/revert/compact/retry logic.
    - `_execute_tool_turn(...)`: Manages the sequence of executing multiple tool calls and appending results to history.

**Benefit:** Improved readability, easier debugging, and the ability to write targeted unit tests for error recovery.

## 2. Implement Token-Aware (Content) Pruning
The current `compact_history` implementation is strictly **turn-based**. It counts how many "turns" exist and drops them. This creates a major edge case: if a single message (e.g., a tool output containing a large file's content) is so massive that it exceeds the context window on its own, `compact_history` will see only one turn and return `dropped=0`. This triggers a `RuntimeError`.

**Proposed Changes:**
- Implement a tiered compaction strategy:
    1. **Tier 1 (Turn Pruning):** Drop oldest turns (current implementation).
    2. **Tier 2 (Content Truncation):** If Tier 1 fails to reduce size, truncate the `content` of the oldest non-system messages (e.g., keeping only the first/last 500 characters) rather than deleting them entirely.

**Benefit:** Prevents "dead-end" failures when dealing with large data inputs or long tool outputs.

## 3. Make Compaction Policies Configurable
The current policy is hardcoded: `keep = 2 if len(turns) > 2 else 1`. This is quite aggressive and nukes most of the conversation history. For many tasks, an agent might only need to drop a few messages to fit within the window; keeping more context would significantly improve performance.

**Proposed Changes:**
- Move compaction parameters into `AgentConfig` using a `CompactionPolicy` dataclass.
- Parameters should include `keep_turns`, `truncate_content`, and `max_retries`.

**Benefit:** Allows developers to tune the balance between "context richness" and "stability" depending on the model being used (e.g., small vs. large models).
