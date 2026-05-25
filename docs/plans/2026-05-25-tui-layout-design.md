# TUI Layout Design

## Context

The existing `tui.py` prototype and the original UI mockup define a five-region layout:
header → alert bar → prompt → skill confirmation → output. This design challenges that
layout and arrives at a simpler, three-region structure that is more consistent with
terminal conventions while preserving the deliberate design choice to keep the composer
at the top.

## Layout

Three regions. No fixed rows for alerts or skill confirmation.

```
┌──────────────────────────────────────────────┐
│ GEMMA │ tokens: 4,521 │ ● STREAMING          │  ← header
├──────────────────────────────────────────────┤
│ > _                                          │  ← composer
├──────────────────────────────────────────────┤
│                                              │
│  transcript (scrollable)                     │  ← transcript
│                                              │
└──────────────────────────────────────────────┘
```

### Header (1 line)

Always visible. Owns live agent state.

```
│ GEMMA │ tokens: 4,521 │ ● STREAMING          │
```

Fields:
- **Model name** — fixed for the session.
- **Tokens** — cumulative total for the session, updated after each turn.
- **Status** — live indicator with a colored bullet:
  - `● STREAMING` — agent is generating text.
  - `◉ TOOL: shell.execute` — a tool is running; shows the tool name.
  - `⏸ AWAITING APPROVAL` — an approval card is pending in the transcript.
  - `○ IDLE` — ready for input.
  - `✗ FAILED` — last turn ended in an error.

No alert counter. Alerts are inline events in the transcript.

### Composer (1–3 lines)

Always at the top, below the header. The user reads the transcript below and types at
the top — a command-bar interaction model rather than a chat model.

```
│ > run the tests and fix any failures         │
```

States:
- **Idle** — cursor at `> `, ready for input. Ghost text hints fade in after a pause:
  `/exit  /stats  /export`.
- **Working** — frame dims; prompt changes to `● working…`. Input is accepted but queued.
  Ctrl-C cancels the running turn.
- **Post-cancel** — brief `✗ cancelled` message, then resets.

Multiline input: Alt+Enter inserts a newline. Enter submits.

Slash commands are typed in the composer and dispatched before the input is sent to the
agent: `/exit`, `/stats`, `/export`, `/help`. Slash-command completion is provided by
`prompt_toolkit`'s `WordCompleter`.

### Transcript (remaining height, scrollable)

The single surface for everything: user turns, agent responses, tool rows, inline alerts,
and inline approval cards. Auto-scrolls to the latest line at all times.

## Transcript structure

### Turn block

Each user/agent exchange renders as a self-contained block:

```
  ╔══════════════════════════════════════════╗
  ║ You: run the tests and fix any failures  ║
  ╚══════════════════════════════════════════╝

  Running pytest now...

  ▶ shell.execute  pytest tests/  ··· 2.1s ✓
    3 failures found. Fixing now...

  ▶ write_file  tests/test_foo.py  ··· 0.3s ✓

  Done. 3 tests fixed, all 265 passing.
  ─────────────────── 18s │ 1,204 tokens ───
```

**User message header** — boxed so it is easy to find when scrolling back.

**Agent response** — plain text, streams character-by-character.

**Tool row** — one line per tool invocation:
```
  ▶ <name>  <args summary>  ··· <elapsed> <state>
```
States: `···` (running) → `✓` (done) → `✗` (failed). Tool output is hidden by default.
A `[+]` expander can reveal the raw output on demand.

**Turn footer** — dim separator with elapsed time and token delta for the turn.

### Inline alert

Alerts appear as styled events in the stream, not in a separate bar.

```
  ⚠  HINT  auto-compact triggered (82%)
```

Color coding: yellow gutter for `WARNING`, blue for `HINT`/`INFO`, red for `ERROR`.

## Inline approval card

When a skill requires approval the agent pauses and an approval card appears in the
transcript at the point where the agent stopped:

```
  ┌─ ⚠ Approval required ──────────────────────┐
  │  shell.execute  rm -rf /tmp/build           │
  │                                             │
  │  [Y] approve   [N] deny                     │
  └─────────────────────────────────────────────┘
```

**Focus model** — while a card is visible, the header shows `⏸ AWAITING APPROVAL` and
key bindings `y` and `n` are intercepted by the card. The composer dims. Resolving the
card restores normal focus.

**Resolved card** — collapses to a single line and stays in the transcript as an audit
record:
```
  ✓ shell.execute approved
  ✗ shell.execute denied
```

## What changed from the original mockup

| Original mockup | This design | Reason |
|---|---|---|
| Fixed alert bar (row between prompt and output) | Alerts inline in transcript | Avoids a persistent dead row; keeps alerts contextual |
| Fixed skill confirmation panel | Approval cards inline in transcript | Same reason; cards appear where the agent paused |
| `ALERTS: 2` badge in header | Removed | Redundant when alerts are inline |
| Prompt between header and output | Prompt directly below header | Cleaner; upholds command-bar model |
| OUTPUT at the bottom with least space | Transcript takes all remaining height | Most important content gets most space |

## Non-goals

- Session management and session resume are out of scope.
- A command palette widget is not required; ghost-text hints and `WordCompleter` cover
  discoverability without the complexity.
- Workspace mode (multi-pane) is not addressed here; this design provides the event
  model and primitives it would build on.

## Implementation order

1. Three-region `prompt_toolkit` layout (header, composer, transcript pane).
2. Structured turn rendering with user header box and turn footer.
3. Tool rows with state transitions.
4. Inline alert rendering.
5. Inline approval cards with focus interception.
6. Composer state (idle / working / cancelled) and ghost-text hints.
