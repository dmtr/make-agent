"""MakeAgentShell — interactive full-screen REPL for the make-agent.

Three-region layout:
  1. Header  (1 line)   — model | tokens | live status
  2. Composer (3 lines) — prompt input at the top; Alt+Enter for newlines
  3. Transcript (rest)  — scrollable; structured turn blocks rendered inline
"""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from prompt_toolkit.application import Application
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.filters import Condition
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame, TextArea

from make_agent.agent_core import (
    AgentManager,
    ApprovalRequested,
    ApproveSkill,
    CancelTurn,
    DenySkill,
    ManagerError,
    ShellCommand,
    ShellEvent,
    Shutdown,
    StartTurn,
    StatusChanged,
    ToolFinished,
    ToolStarted,
    TokenEmitted,
    TurnCancelled,
    TurnFinished,
    TurnStarted,
)

MAX_MESSAGES_TO_DISPLAY = 10
TRANSCRIPT_SEPARATOR = "  " + "─" * 72

# ── status / enums ──────────────────────────────────────────────────────────────


class AgentStatus(str, Enum):
    IDLE = "IDLE"
    STREAMING = "STREAMING"
    TOOL = "TOOL"
    AWAITING_APPROVAL = "AWAITING APPROVAL"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


STATUS_INDICATOR = {
    AgentStatus.IDLE: "○",
    AgentStatus.STREAMING: "●",
    AgentStatus.TOOL: "◉",
    AgentStatus.AWAITING_APPROVAL: "⏸",
    AgentStatus.FAILED: "✗",
    AgentStatus.CANCELLED: "✗",
}

STATUS_STYLE = {
    AgentStatus.IDLE: "class:status.idle",
    AgentStatus.STREAMING: "class:status.streaming",
    AgentStatus.TOOL: "class:status.tool",
    AgentStatus.AWAITING_APPROVAL: "class:status.approval",
    AgentStatus.FAILED: "class:status.error",
    AgentStatus.CANCELLED: "class:status.error",
}

# ── transcript data classes ─────────────────────────────────────────────────────


@dataclass
class ToolRow:
    """One tool invocation in the tool activity block."""

    name: str
    args: dict[str, Any]
    state: str = "running"  # running | done | failed
    elapsed: float = 0.0
    start_time: float = field(default_factory=time.time)
    output_preview: str = ""

    def render(self) -> str:
        if self.state == "running":
            elapsed_str = "···"
            icon = "◌"
        elif self.state == "done":
            elapsed_str = f"{self.elapsed:.1f}s"
            icon = "✓"
        else:
            elapsed_str = f"{self.elapsed:.1f}s"
            icon = "✗"
        args_short = " ".join(f"{k}={str(v)[:24]!r}" for k, v in list(self.args.items())[:2])
        line = f"  {icon} {self.name}  {args_short}  {elapsed_str}"
        if self.output_preview and self.state != "running":
            preview = self.output_preview[:80].replace("\n", "↵")
            line += f"\n    └ {preview}"
        return line


@dataclass
class ApprovalCard:
    """Inline approval card rendered in the transcript."""

    skill_name: str
    target: str
    kwargs: dict[str, Any]
    request_id: str
    resolved: Optional[bool] = None  # None = pending; True = approved; False = denied

    def render(self) -> str:
        if self.resolved is not None:
            mark = "✓" if self.resolved else "✗"
            word = "approved" if self.resolved else "denied"
            return f"  {mark} {self.skill_name}/{self.target} {word}"
        args_repr = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())[:38]
        width = 44
        pad = width - 2
        return "\n".join([
            "  ┌─ ⚠ Approval required " + "─" * (width - 22) + "┐",
            f"  │  {(self.skill_name + '/' + self.target + '(' + args_repr + ')')[:pad]:<{pad}}│",
            "  │" + " " * (width) + "│",
            f"  │  [Y] approve   [N] deny{' ' * (pad - 24)}│",
            "  └" + "─" * width + "┘",
        ])


_ALERT_ICONS: dict[str, str] = {"INFO": "ℹ", "HINT": "ℹ", "WARNING": "⚠", "ERROR": "✗"}


@dataclass
class InlineAlert:
    """An inline alert event rendered in the transcript stream."""

    level: str
    message: str

    def render(self) -> str:
        icon = _ALERT_ICONS.get(self.level, "·")
        return f"  {icon} {self.level}  {self.message}"


@dataclass
class TurnBlock:
    """One agent turn rendered as a stable container in the transcript."""

    user_message: str
    response_lines: list[str] = field(default_factory=list)
    tools: list[ToolRow] = field(default_factory=list)
    approval: Optional[ApprovalCard] = None
    state: str = "streaming"  # streaming | done | failed | cancelled
    elapsed: float = 0.0
    tokens: int = 0
    start_time: float = field(default_factory=time.time)

    def render(self) -> str:
        parts: list[str] = []

        # Assistant response body
        for line in self.response_lines:
            parts.append(f"  {line}")

        # Tool activity block
        if self.tools:
            parts.append("")
            for tool in self.tools:
                parts.append(tool.render())

        # Inline approval card
        if self.approval:
            parts.append("")
            parts.append(self.approval.render())

        # Turn footer — elapsed time, token count, outcome
        if self.state in ("done", "failed", "cancelled"):
            elapsed = self.elapsed or (time.time() - self.start_time)
            state_label = self.state.upper()
            sep = f"  {'─' * 8} {elapsed:.0f}s │ {self.tokens} tok │ {state_label} {'─' * 8}"
            parts.append("")
            parts.append(sep)

        parts.append("")
        return "\n".join(parts)


# ── shell state ─────────────────────────────────────────────────────────────────


class ShellState:
    """Mutable application state consumed by the shell UI."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.total_tokens: int = 0
        self.status = AgentStatus.IDLE
        self.active_tool: Optional[str] = None
        self.transcript: list[TurnBlock | InlineAlert] = []
        self.current_turn: Optional[TurnBlock] = None
        self.pending_approval: Optional[ApprovalCard] = None
        self._approval_future: Optional[asyncio.Future[bool]] = None
        self.transcript_focused: bool = False
        self.selected_message_idx: int = 0

    def new_turn(self, message: str) -> TurnBlock:
        turn = TurnBlock(user_message=message)
        self.transcript.append(turn)
        self.current_turn = turn
        self.status = AgentStatus.STREAMING
        return turn

    def finish_turn(
        self,
        turn: TurnBlock,
        *,
        failed: bool = False,
        cancelled: bool = False,
    ) -> None:
        if cancelled:
            turn.state = "cancelled"
        elif failed:
            turn.state = "failed"
        else:
            turn.state = "done"
        turn.elapsed = time.time() - turn.start_time
        self.status = AgentStatus.IDLE
        self.current_turn = None

    def add_alert(self, level: str, message: str) -> None:
        self.transcript.append(InlineAlert(level=level, message=message))


# ── MakeAgentShell ──────────────────────────────────────────────────────────────


class MakeAgentShell:
    """Interactive full-screen shell backed by an :class:`AgentManager`."""

    def __init__(
        self,
        agent_manager: AgentManager,
        session_id: str,
        model: str,
        history_path: Path,
        max_messages_to_display: int = MAX_MESSAGES_TO_DISPLAY,
    ) -> None:
        self._agent_manager = agent_manager
        self._session_id = session_id
        self._history_path = history_path
        self._max_messages_to_display = max_messages_to_display
        self._command_queue: asyncio.Queue[ShellCommand] = asyncio.Queue()
        self._event_queue: asyncio.Queue[ShellEvent] = asyncio.Queue()
        self._state = ShellState(model=model)
        self._app: Optional[Application] = None
        self._transcript_area: Optional[TextArea] = None
        self._commands: dict[str, Any] = {
            "exit": self._cmd_exit,
            "quit": self._cmd_exit,
            "export": self._cmd_export,
            "stats": self._cmd_stats,
            "help": self._cmd_help,
        }

    # ── rendering ───────────────────────────────────────────────────────────────

    def _header_text(self) -> list[tuple[str, str]]:
        s = self._state
        label = s.status.value
        if s.status == AgentStatus.TOOL and s.active_tool:
            label = f"TOOL: {s.active_tool}"
        indicator = STATUS_INDICATOR[s.status]
        status_style = STATUS_STYLE[s.status]
        return [
            ("class:header", f" {s.model}"),
            ("class:header.sep", " │ "),
            ("class:header", f"tokens: {s.total_tokens}"),
            ("class:header.sep", " │ "),
            (status_style, f"{indicator} {label} "),
        ]

    def _render_transcript(self) -> str:
        items = list(reversed(self._state.transcript[-self._max_messages_to_display:]))
        sep = "\n" + TRANSCRIPT_SEPARATOR + "\n"
        parts = []
        for i, item in enumerate(items):
            rendered = item.render()
            if self._state.transcript_focused and i == self._state.selected_message_idx:
                lines = rendered.split("\n")
                for j, line in enumerate(lines):
                    if line.startswith("  "):
                        lines[j] = "► " + line[2:]
                        break
                rendered = "\n".join(lines)
            parts.append(rendered)
        return sep.join(parts)

    def _transcript_cursor_offset(self, items: list) -> int:
        """Character offset of the selected message in the rendered transcript."""
        sep = "\n" + TRANSCRIPT_SEPARATOR + "\n"
        idx = min(self._state.selected_message_idx, max(0, len(items) - 1))
        offset = 0
        for i in range(idx):
            offset += len(items[i].render()) + len(sep)
        return offset

    def _refresh(self) -> None:
        """Update the transcript area text and invalidate the app."""
        if self._transcript_area is not None:
            text = self._render_transcript()
            self._transcript_area.text = text
            if self._state.transcript_focused:
                items = list(reversed(self._state.transcript[-self._max_messages_to_display:]))
                self._transcript_area.buffer.cursor_position = self._transcript_cursor_offset(items)
            else:
                self._transcript_area.buffer.cursor_position = 0
        if self._app is not None:
            self._app.invalidate()

    # ── layout ──────────────────────────────────────────────────────────────────

    def _build_app(self) -> Application:
        state = self._state

        # Transcript pane — scrollable, read-only
        transcript_area = TextArea(
            text="",
            scrollbar=True,
            focusable=False,
            read_only=True,
            wrap_lines=True,
        )
        self._transcript_area = transcript_area

        # Composer input — multiline with slash-command completion and history
        completer = WordCompleter(
            ["/" + name for name in self._commands],
            sentence=True,
        )
        composer_input = TextArea(
            height=2,
            prompt="> ",
            multiline=True,
            completer=completer,
            history=FileHistory(str(self._history_path)),
            wrap_lines=False,
        )
        self._composer_input = composer_input

        # Contextual hint line below the composer
        approval_active = Condition(
            lambda: state._approval_future is not None
            and not (state._approval_future.done() if state._approval_future else True)
        )
        transcript_focus_active = Condition(lambda: state.transcript_focused)
        hint_control = FormattedTextControl(
            lambda: (
                [("class:hint.transcript", "  ► TRANSCRIPT  Ctrl+P prev   Ctrl+N next   Ctrl+T back to input")]
                if state.transcript_focused
                else (
                    [("class:hint.approval", "  [Y] approve   [N] deny")]
                    if approval_active()
                    else (
                        [("class:hint.busy", "  ● working…  (Ctrl-C cancels the current turn)")]
                        if state.status != AgentStatus.IDLE
                        else [
                            (
                                "class:hint",
                                "  /exit  /stats  /export  /help"
                                "   │  Alt+Enter for newlines  │  Ctrl+T transcript  │  Ctrl-C exits",
                            )
                        ]
                    )
                )
            )
        )
        hint_window = Window(height=1, content=hint_control)

        # Header — 1 line: model | tokens | status
        header_window = Window(
            height=1,
            content=FormattedTextControl(self._header_text),
        )

        # Three-region root layout
        root = HSplit([
            header_window,
            Frame(body=HSplit([composer_input, hint_window]), title=""),
            Frame(body=transcript_area, title="TRANSCRIPT"),
        ])

        layout = Layout(root, focused_element=composer_input)

        # Key bindings
        kb = KeyBindings()

        @kb.add("enter")
        def _on_enter(event) -> None:
            if state.status != AgentStatus.IDLE or state.transcript_focused:
                return
            text = composer_input.text.strip()
            if not text:
                return
            composer_input.text = ""
            if text.startswith("/"):
                should_exit = self._dispatch_command(text[1:])
                if should_exit:
                    event.app.exit()
                else:
                    self._refresh()
            else:
                asyncio.ensure_future(self._run_turn(text))

        @kb.add("escape", "enter")
        def _on_alt_enter(event) -> None:
            if state.status == AgentStatus.IDLE and not state.transcript_focused:
                composer_input.buffer.insert_text("\n")

        @kb.add("c-c", eager=True)
        def _on_ctrl_c(event) -> None:
            if state.transcript_focused:
                state.transcript_focused = False
                self._refresh()
            elif state.status != AgentStatus.IDLE:
                asyncio.ensure_future(self._command_queue.put(CancelTurn()))
            elif not event.app.current_buffer.text:
                event.app.exit()
            else:
                # Buffer has text: clear it (readline/bash behaviour for first press)
                event.app.current_buffer.reset()

        @kb.add("c-d", eager=True)
        def _on_ctrl_d(event) -> None:
            if not event.app.current_buffer.text:
                event.app.exit()
            else:
                # Buffer has text: delete character under cursor (emacs C-d behaviour)
                event.app.current_buffer.delete()

        @kb.add("c-t")
        def _on_ctrl_t(event) -> None:
            if state.transcript_focused:
                state.transcript_focused = False
            else:
                items = list(reversed(state.transcript[-self._max_messages_to_display:]))
                if items:
                    state.transcript_focused = True
                    state.selected_message_idx = 0
            self._refresh()

        @kb.add("c-n", filter=transcript_focus_active)
        def _on_next_message(event) -> None:
            items = list(reversed(state.transcript[-self._max_messages_to_display:]))
            state.selected_message_idx = min(
                state.selected_message_idx + 1, max(0, len(items) - 1)
            )
            self._refresh()

        @kb.add("c-p", filter=transcript_focus_active)
        def _on_prev_message(event) -> None:
            state.selected_message_idx = max(0, state.selected_message_idx - 1)
            self._refresh()

        @kb.add("y", filter=approval_active)
        def _on_y(event) -> None:
            fut = state._approval_future
            if fut and not fut.done():
                fut.set_result(True)

        @kb.add("n", filter=approval_active)
        def _on_n(event) -> None:
            fut = state._approval_future
            if fut and not fut.done():
                fut.set_result(False)

        app_style = Style.from_dict({
            "header": "bg:#1a1a2e #c0c0d0",
            "header.sep": "bg:#1a1a2e #555566",
            "status.idle": "bg:#1a1a2e #666677",
            "status.streaming": "bg:#1a1a2e #00cc66 bold",
            "status.tool": "bg:#1a1a2e #00aaff bold",
            "status.approval": "bg:#1a1a2e #ffaa00 bold",
            "status.error": "bg:#1a1a2e #ff4444 bold",
            "frame.border": "#444466",
            "hint": "#555566 italic",
            "hint.busy": "#00aaff italic",
            "hint.approval": "#ffaa00 bold",
            "hint.transcript": "#00aaff bold",
        })

        return Application(
            layout=layout,
            key_bindings=kb,
            full_screen=True,
            style=app_style,
            mouse_support=False,
        )

    # ── command handlers ─────────────────────────────────────────────────────────

    def _cmd_exit(self) -> bool:
        return True

    def _cmd_export(self) -> bool:
        path = self._agent_manager.export_conversation(self._session_id)
        if path:
            self._state.add_alert("INFO", f"Conversation exported to {path}")
        else:
            self._state.add_alert("INFO", "Nothing to export yet.")
        return False

    def _cmd_stats(self) -> bool:
        stats = self._agent_manager.get_token_stats(self._session_id)
        if not stats:
            self._state.add_alert(
                "INFO", "No token usage stats available (memory not enabled or no LLM calls yet)."
            )
        else:
            self._state.add_alert(
                "INFO",
                f"Tokens: {stats['total_tokens']} total"
                f"  ({stats['input_tokens']} in / {stats['output_tokens']} out)"
                f"  models: {', '.join(stats['models'])}",
            )
        return False

    def _cmd_help(self) -> bool:
        cmds = "  ".join(f"/{name}" for name in self._commands)
        self._state.add_alert("INFO", f"Commands: {cmds}")
        self._state.add_alert(
            "INFO", "Any other input is sent to the agent. Alt+Enter inserts a newline."
        )
        return False

    def _dispatch_command(self, line: str) -> bool:
        """Dispatch a /command. Returns True if the shell should exit."""
        name, *_ = line.strip().split(None, 1)
        handler = self._commands.get(name)
        if handler is None:
            self._state.add_alert("WARNING", f"Unknown command: /{name}  (type /help for a list)")
            return False
        return handler()

    # ── event consumption ────────────────────────────────────────────────────────

    async def _consume_turn_events(self, turn: TurnBlock, turn_id: str) -> None:  # noqa: ARG002
        """Read events from the bridge until the current turn ends."""
        while True:
            event = await self._event_queue.get()

            if isinstance(event, TokenEmitted):
                # Accumulate tokens into the last response line, splitting on newlines
                combined = ("\n".join(turn.response_lines) if turn.response_lines else "") + event.text
                turn.response_lines = combined.split("\n")
                self._refresh()

            elif isinstance(event, ToolStarted):
                self._state.active_tool = event.name
                self._state.status = AgentStatus.TOOL
                turn.tools.append(ToolRow(name=event.name, args=event.args))
                self._refresh()

            elif isinstance(event, ToolFinished):
                self._state.active_tool = None
                self._state.status = AgentStatus.STREAMING
                for tr in reversed(turn.tools):
                    if tr.name == event.name and tr.state == "running":
                        tr.state = "failed" if event.is_error else "done"
                        tr.elapsed = (event.duration_ms or 0) / 1000.0
                        tr.output_preview = (event.output or "")[:120]
                        break
                self._refresh()

            elif isinstance(event, ApprovalRequested):
                card = ApprovalCard(
                    skill_name=event.skill_name,
                    target=event.target,
                    kwargs=event.kwargs,
                    request_id=event.request_id,
                )
                turn.approval = card
                self._state.pending_approval = card
                self._state.status = AgentStatus.AWAITING_APPROVAL
                future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
                self._state._approval_future = future
                self._refresh()

                approved = await future

                card.resolved = approved
                self._state.pending_approval = None
                self._state._approval_future = None
                self._state.status = AgentStatus.STREAMING
                if approved:
                    await self._command_queue.put(ApproveSkill(request_id=event.request_id))
                else:
                    await self._command_queue.put(DenySkill(request_id=event.request_id))
                self._refresh()

            elif isinstance(event, TurnFinished):
                stats = self._agent_manager.get_token_stats(self._session_id)
                if stats:
                    turn.tokens = stats["total_tokens"] - self._state.total_tokens
                    self._state.total_tokens = stats["total_tokens"]
                self._state.finish_turn(turn)
                self._refresh()
                break

            elif isinstance(event, TurnCancelled):
                self._state.finish_turn(turn, cancelled=True)
                self._refresh()
                break

            elif isinstance(event, ManagerError):
                self._state.add_alert("ERROR", event.message)
                self._state.finish_turn(turn, failed=True)
                self._refresh()
                break

            elif isinstance(event, (TurnStarted, StatusChanged)):
                pass  # informational; state managed locally

    # ── agent turn ───────────────────────────────────────────────────────────────

    async def _run_turn(self, message: str) -> None:
        """Send one turn to the bridge and consume events until it finishes."""
        turn_id = str(uuid4())
        turn = self._state.new_turn(message)
        self._refresh()
        await self._command_queue.put(StartTurn(message=message, turn_id=turn_id))
        try:
            await self._consume_turn_events(turn, turn_id)
        except Exception as e:
            self._state.add_alert("ERROR", str(e))
            self._state.finish_turn(turn, failed=True)
            self._refresh()

    # ── main loop ────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Start the interactive full-screen shell."""
        bridge_task = asyncio.create_task(
            self._agent_manager.run_shell_bridge(
                self._session_id, self._command_queue, self._event_queue
            )
        )
        self._app = self._build_app()
        self._state.add_alert(
            "INFO",
            "Type your message and press Enter.  Alt+Enter inserts a newline.  /help for commands.",
        )
        self._refresh()
        try:
            await self._app.run_async()
        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            await self._command_queue.put(Shutdown())
            try:
                await asyncio.wait_for(bridge_task, timeout=3.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                bridge_task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await bridge_task
