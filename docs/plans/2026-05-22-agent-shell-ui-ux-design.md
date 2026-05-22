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
5. Shared primitives: Quick mode and Workspace mode should reuse the same components and
   event model.

## Product shape

The shell should evolve into a dual-mode experience:

- **Quick mode** is the default. It remains a single-column REPL, but with clearer turn
  structure, better input ergonomics, stronger state visibility, and improved approvals.
- **Workspace mode** is an expanded full-screen view for longer or tool-heavy sessions.
  It should use the same underlying session and the same interaction semantics as Quick
  mode, but with more screen real estate and more explicit secondary surfaces.

This keeps the shell accessible to new users while giving advanced users a denser and
more inspectable interface when they need it.

## Quick mode v1

Quick mode v1 should be the first shipping slice. It should improve the current REPL
without turning it into a full terminal UI. The recommended scope is:

- Structured turn rendering.
- Better composer behavior and command discovery.
- Clear agent state in the shell chrome.
- Inline approval cards instead of raw yes/no prompts.
- A lightweight command palette for high-value actions.

Quick mode v1 should remain single-column and transcript-first. Panes, side rails,
timeline views, and advanced workspace layout should come later.

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

The input area should feel more like a small editor than a plain input line. Quick mode
should preserve multiline entry while adding:

- Draft persistence for the current session
- Placeholder hints for key shortcuts
- Better slash-command discovery
- A command palette for high-value actions such as help, export, stats, search current
  transcript, and jump to the latest tool result

Hints should be contextual and unobtrusive. New users need discoverability, but regular
users should not be slowed down by persistent instructional noise.

## Approvals and safety prompts

The shell should replace raw `Allow X? [y/N]` prompts with lightweight approval cards.
Each approval should show:

- Requested skill or action
- Relevant arguments or scope
- A simple risk label when applicable
- Clear actions such as approve once, always trust, deny, and view details

In Quick mode this remains inline. In Workspace mode the same approval model can move
into a dedicated queue without changing the semantics.

## Onboarding and confidence signals

The shell should make the current state of the interaction obvious at all times. A user
should not have to guess whether the agent is still responding, is waiting on a tool, is
blocked on approval, or has finished.

Quick mode should include:

- A clearer startup surface with model, skill mode, key shortcuts, and the main commands
- A compact status line with model, active tool, and current shell state
- Elapsed-time feedback for long-running work
- Explicit messaging when a turn is waiting, retrying, cancelled, or done

This is more important than session resume for now. The immediate UX priority is helping
users stay oriented inside the live conversation.

## Workspace mode direction

Workspace mode should build on the same primitives as Quick mode rather than becoming a
separate shell. It should expand the current session into a richer terminal layout with:

- Transcript pane
- Tool activity pane
- Approval queue
- Status and context area
- Space for pinned items or turn navigation within the current session

Workspace mode should only be introduced after Quick mode has a solid event-driven
rendering model and stronger interaction primitives.

## UI architecture

The shell UI should be built from reusable primitives:

- Composer
- Transcript
- Tool Feed
- Approval Queue
- Status Bar
- Command Palette

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

1. Introduce event-driven turn rendering and stable turn containers.
2. Add compact tool rows and collapsed output behavior.
3. Improve the composer and add command discovery.
4. Replace raw confirmations with approval cards.
5. Polish Quick mode chrome and state visibility.
6. Add Workspace mode on top of the same primitives.

## Success criteria

Quick mode v1 is successful if:

- A new user can understand what the shell is doing without relying on `/help`.
- A regular user can follow long turns with less scroll fatigue.
- Tool-heavy turns stay readable because tool output is summarized first.
- Approval prompts feel safe and informative instead of disruptive.
- The shell gains the structural primitives needed for Workspace mode later.
