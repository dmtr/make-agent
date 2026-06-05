# Agent shell UI/UX design

## Context

The current shell is a prompt-toolkit REPL with multiline input, streaming model output,
and inline tool events. It is functional, but long turns can become hard to scan because
assistant text, tool activity, approvals, and status updates all share the same linear
output surface.

This design defines a clearer shell UX that keeps the default experience lightweight
while creating a path toward a richer terminal workspace.

## Goals

- Make each turn scannable in a few seconds.
- Keep the default shell approachable for first-time users.
- Preserve fast keyboard-driven workflows for regular users.
- Reduce transcript noise from tool execution.
- Surface agent state clearly: responding, running tools, awaiting approval, failed,
  cancelled, or done.
- Use one interaction model that can scale from a simple REPL to a richer full-screen UI.

## Non-goals

- Session resume and session management are not in scope for the first design slice.
- A full terminal IDE or agent cockpit is not the immediate target.
- Workspace mode is not required for the first implementation slice.

## Design principles

1. Progressive disclosure: start simple, reveal structure only when the user needs it.
2. Minimal noise by default: summarize first, expand details on demand.
3. Stable layout: a turn should keep the same visual anatomy while streaming.
4. One session focus: optimize for the live conversation before adding session management.
5. Command-bar model: the composer lives at the top; the transcript flows below it.

## Layout

The shell uses a single three-region layout:

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

**Header** (1 line): model name, cumulative token count, live status indicator.
Status values: `○ IDLE`, `● STREAMING`, `◉ TOOL: <name>`, `⏸ AWAITING APPROVAL`,
`✗ FAILED`, `✗ CANCELLED`. No separate alert counter.

**Composer** (1–3 lines): input at the top, below the header. This is a command-bar
interaction model — the user issues tasks from the top and reads results below. Supports
multiline entry via Alt+Enter. Slash-command ghost text fades in when idle (e.g.
`/exit  /stats  /export`). While the agent is running the composer dims and shows
`● working…`; Ctrl-C cancels the current turn.

**Transcript** (remaining height, scrollable): the single surface for all events —
user turns, agent responses, tool rows, inline alerts, and approval cards. Auto-scrolls
to the latest line at all times.

## First implementation slice

The recommended scope for the first shipping slice:

- Three-region `prompt_toolkit` layout.
- Structured turn rendering.
- Better composer behavior and command discovery.
- Clear agent state in the shell chrome.
- Inline approval cards instead of raw yes/no prompts.

## Structured turn rendering

Structured turn rendering is the foundation for the rest of the shell. A turn should no
longer behave like a raw terminal log. Instead, every turn should render as one stable
container with four ordered regions:

1. **User message header**: compact and always visible.
2. **Assistant response body**: the primary reading area; streamed tokens render here.
3. **Tool activity block**: visible only if tools are used; stays secondary to the
   assistant response.
4. **Turn footer**: outcome state and metadata such as elapsed time.

The turn should keep this structure while streaming. New events should update the
existing container instead of spraying unstructured lines into the scrollback.

### Turn states

Visible states should stay minimal:

- Streaming
- Running tool
- Awaiting approval
- Completed
- Failed
- Cancelled

State should be shown once per turn rather than repeated across multiple lines.

## Tool activity block

The tool activity block should optimize for minimal noise by default. Each tool
invocation should render as a compact tool row with:

- Tool name
- Short description
- Current state
- Elapsed time
- One-line result summary after completion

Tool rows should transition cleanly through queued, running, completed, failed, or
cancelled states.

Verbose tool output should be collapsed by default. The transcript should show a short
preview only when it materially helps comprehension. Failed tools should remain compact,
but they should display a stronger summary and make details easy to inspect.

This keeps the main reading path focused on the assistant response while preserving
observability when the user wants it.

## Composer and command discovery

The input area should feel more like a small editor than a plain input line. The
composer should preserve multiline entry while adding:

- Draft persistence for the current session
- Placeholder hints for key shortcuts
- Better slash-command discovery via `WordCompleter`

Hints should be contextual and unobtrusive. New users need discoverability, but regular
users should not be slowed down by persistent instructional noise. A command palette
widget is not required; ghost-text hints and completion cover the discoverability problem
with far less complexity.

## Approvals and safety prompts

The shell should replace raw `Allow X? [y/N]` prompts with lightweight inline approval
cards. Each card appears in the transcript at the point where the agent paused. It shows:

- Requested skill or action

Clear actions: approve (`y`) or deny (`n`).

While a card is visible the header shows `⏸ AWAITING APPROVAL` and the `y`/`n` key
bindings are intercepted by the card. The composer dims. Resolving the card restores
normal focus.

Resolved cards collapse to a single audit line in the transcript:
`✓ shell.execute approved` or `✗ shell.execute denied`.

Alerts (harness warnings, auto-compact notices, etc.) appear as inline events in the
transcript stream — not in a fixed alert bar. This keeps the layout free of persistent
dead rows and keeps alerts contextual to where they occurred.

## Onboarding and confidence signals

The shell should make the current state of the interaction obvious at all times. A user
should not have to guess whether the agent is still responding, is waiting on a tool, is
blocked on approval, or has finished.

The shell should include:

- A compact header with model, token count, and current shell state
- Elapsed-time feedback in the turn footer for long-running work
- Explicit messaging when a turn is waiting, retrying, cancelled, or done

## UI architecture

The shell UI should be built from reusable primitives:

- Composer
- Transcript
- Tool Feed
- Approval Queue
- Status Bar

Both shell modes should consume the same normalized event stream. Useful event types
include:

- token appended
- tool started
- tool updated
- tool finished
- approval requested
- approval resolved
- turn cancelled
- turn completed

The UI should render from structured turn state derived from these events rather than
printing directly from the raw stream. That will make future features such as folding,
pinning, filtering, and alternate layouts much easier to add.

## Error handling

The shell should distinguish between:

- Agent failure
- Tool failure
- User interruption

These should not all look the same in the transcript. Each case should present an
appropriate recovery path, such as retry turn, inspect tool output, or dismiss.

## Rollout plan

Recommended implementation order:

1. Three-region `prompt_toolkit` layout (header, composer, transcript pane).
2. Structured turn rendering with user header box and turn footer.
3. Compact tool rows with state transitions.
4. Inline alert rendering.
5. Inline approval cards with focus interception.
6. Composer state (idle / working / cancelled) and ghost-text hints.

## Success criteria

The shell is successful if:

- A new user can understand what the shell is doing without relying on `/help`.
- A regular user can follow long turns with less scroll fatigue.
- Tool-heavy turns stay readable because tool output is summarized first.
- Approval prompts feel safe and informative instead of disruptive.
