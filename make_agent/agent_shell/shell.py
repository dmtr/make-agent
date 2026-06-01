"""MakeAgentShell — interactive full-screen REPL for the make-agent.

Five-region layout:
  1. Header     (1 line)   — model | tokens | live status | alert (right)
  2. Composer   (3 lines)  — prompt input; Alt+Enter for newlines
  3. Turn N     (rest)     — response pane (top) + tools pane (bottom)
  4. Footer     (1 line)   — session start time
"""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from prompt_toolkit.application import Application
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.filters import Condition
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame, TextArea

from make_agent.agent_core import (
    AgentManager,
    ApprovalRequested,
    ApproveSkill,
    CancelTurn,
    DenySkill,
    HistoryCompacted,
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
        args_short = " ".join(f"{k}={str(v)[:60]!r}" for k, v in self.args.items())
        line = f"  {icon} {self.name}  {args_short}  {elapsed_str}"
        if self.output_preview and self.state != "running":
            preview = self.output_preview[:120].replace("\n", "↵")
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
        # Ensure kwargs is a dict before calling .items()
        kwargs_dict = self.kwargs if isinstance(self.kwargs, dict) else {}
        args_repr = ", ".join(f"{k}={v!r}" for k, v in kwargs_dict.items())[:38]
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

    def render_response(self) -> str:
        """Render only the user message header + LLM answer lines + turn footer."""
        parts: list[str] = []

        # User message header
        parts.append(f"  ▶ {self.user_message}")
        parts.append("  " + "╌" * 36)

        # Assistant response body
        for line in self.response_lines:
            parts.append(f"  {line}")

        # Turn footer — always shown for finished turns
        if self.state != "streaming":
            elapsed = self.elapsed or (time.time() - self.start_time)
            state_label = "" if self.state == "done" else f" {self.state.upper()}"
            sep = f"  {'─' * 8} {elapsed:.0f}s │ {self.tokens} tok{state_label} {'─' * 8}"
            parts.append("")
            parts.append(sep)

        parts.append("")
        return "\n".join(parts)

    def render_tools(self) -> str:
        """Render only tool rows and any approval card."""
        parts: list[str] = []

        if self.approval:
            parts.append(self.approval.render())

        for tool in self.tools:
            parts.append(tool.render())

        return "\n".join(parts)

    def render(self) -> str:
        parts: list[str] = []

        # User message header
        parts.append(f"  ▶ {self.user_message}")
        parts.append("  " + "╌" * 36)

        # Assistant response body
        for line in self.response_lines:
            parts.append(f"  {line}")

        # Tool activity block — show every tool individually, never collapse
        if self.tools:
            parts.append("")
            for tool in self.tools:
                parts.append(tool.render())

        # Turn footer — always shown for finished turns
        if self.state != "streaming":
            elapsed = self.elapsed or (time.time() - self.start_time)
            state_label = "" if self.state == "done" else f" {self.state.upper()}"
            sep = f"  {'─' * 8} {elapsed:.0f}s │ {self.tokens} tok{state_label} {'─' * 8}"
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
        self.transcript: list[TurnBlock] = []
        self.current_turn: Optional[TurnBlock] = None
        self.pending_approval: Optional[ApprovalCard] = None
        self._approval_future: Optional[asyncio.Future[bool]] = None
        self.transcript_focused: bool = False
        self.viewed_turn_index: Optional[int] = None  # None = always follow latest
        self.current_alert: Optional[InlineAlert] = None

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
        self.current_alert = InlineAlert(level=level, message=message)


# ── MakeAgentShell ──────────────────────────────────────────────────────────────


class MakeAgentShell:
    """Interactive full-screen shell backed by an :class:`AgentManager`."""

    def __init__(
        self,
        agent_manager: AgentManager,
        session_id: str,
        model: str,
        history_path: Path,
    ) -> None:
        self._agent_manager = agent_manager
        self._session_id = session_id
        self._history_path = history_path
        self._command_queue: asyncio.Queue[ShellCommand] = asyncio.Queue()
        self._event_queue: asyncio.Queue[ShellEvent] = asyncio.Queue()
        self._state = ShellState(model=model)
        self._session_start = datetime.now()
        self._app: Optional[Application] = None
        self._response_area: Optional[TextArea] = None
        self._tools_area: Optional[TextArea] = None
        self._commands: dict[str, Any] = {
            "exit": self._cmd_exit,
            "quit": self._cmd_exit,
            "export": self._cmd_export,
            "stats": self._cmd_stats,
            "help": self._cmd_help,
        }

    # ── rendering ───────────────────────────────────────────────────────────────

    def _header_left_text(self) -> list[tuple[str, str]]:
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
            ("class:header", f"turn: {len(s.transcript)}"),
            ("class:header.sep", " │ "),
            (status_style, f"{indicator} {label} "),
        ]

    def _header_alert_text(self) -> list[tuple[str, str]]:
        s = self._state
        if s.current_alert is None:
            return [("class:header", "")]
        icon = _ALERT_ICONS.get(s.current_alert.level, "·")
        return [("class:header.alert", f" {icon} {s.current_alert.message} ")]

    def _viewed_turn(self) -> Optional[TurnBlock]:
        if not self._state.transcript:
            return None
        idx = self._state.viewed_turn_index
        if idx is None:
            return self._state.transcript[-1]
        return self._state.transcript[idx]

    def _render_response(self) -> str:
        turn = self._viewed_turn()
        return turn.render_response() if turn else ""

    def _render_tools(self) -> str:
        turn = self._viewed_turn()
        return turn.render_tools() if turn else ""

    def _refresh(self) -> None:
        """Update the response and tools areas and invalidate the app."""
        if self._response_area is not None:
            text = self._render_response()
            self._response_area.text = text
            # Auto-scroll only when following the live latest turn
            if not self._state.transcript_focused and self._state.viewed_turn_index is None:
                self._response_area.buffer.cursor_position = len(text)
                self._response_area.window.vertical_scroll = 999999
        if self._tools_area is not None:
            tools_text = self._render_tools()
            self._tools_area.text = tools_text
            if self._state.viewed_turn_index is None:
                self._tools_area.window.vertical_scroll = 999999
        if self._app is not None:
            self._app.invalidate()

    # ── layout ──────────────────────────────────────────────────────────────────

    def _build_app(self) -> Application:
        state = self._state

        # Response pane — LLM answer, bright and scrollable
        response_area = TextArea(
            text="",
            scrollbar=True,
            focusable=True,
            read_only=True,
            wrap_lines=True,
            style="fg:#dcdcdc bold",
        )
        self._response_area = response_area

        # Tools pane — tool rows, fixed height and scrollable
        tools_area = TextArea(
            text="",
            height=8,
            scrollbar=True,
            focusable=False,
            read_only=True,
            wrap_lines=True,
            style="fg:#7a8494",
        )
        self._tools_area = tools_area

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
        def _approval_hint() -> list[tuple[str, str]]:
            card = state.pending_approval
            if card is None:
                return [("class:hint.approval", "  [Y] approve   [N] deny")]
            # Ensure kwargs is a dict before calling .items()
            kwargs_dict = card.kwargs if isinstance(card.kwargs, dict) else {}
            args_repr = ", ".join(f"{k}={v!r}" for k, v in kwargs_dict.items())[:60]
            call = f"{card.skill_name}/{card.target}({args_repr})"
            return [
                ("class:hint.approval", "  [Y] approve   [N] deny"),
                ("class:hint.approval.sep", "  │  "),
                ("class:hint.approval.call", call),
            ]

        def _transcript_hint() -> list[tuple[str, str]]:
            n = len(state.transcript)
            idx = state.viewed_turn_index
            if idx is None:
                position = f"Turn {n}"
            else:
                position = f"Turn {idx + 1}/{n}"
            return [("class:hint.transcript", f"  ► {position}  Ctrl+P prev  Ctrl+N next  ↑↓ scroll  Ctrl+T exit")]

        hint_control = FormattedTextControl(
            lambda: (
                _transcript_hint()
                if state.transcript_focused
                else (
                    _approval_hint()
                    if approval_active()
                    else (
                        [("class:hint.busy", "  ● working…  Ctrl-C cancel turn")]
                        if state.status != AgentStatus.IDLE
                        else [("class:hint", "  /help /stats /export /exit   Alt+Enter newline   Ctrl+T transcript")]
                    )
                )
            )
        )
        hint_window = Window(height=1, content=hint_control)

        # Header — VSplit: left (model | tokens | turn | status) + right (alert)
        header_left = Window(
            height=1,
            content=FormattedTextControl(self._header_left_text),
            style="class:header",
        )
        header_right = Window(
            height=1,
            content=FormattedTextControl(self._header_alert_text),
            style="class:header",
            dont_extend_width=True,
        )
        header_window = VSplit([header_left, header_right], height=1)

        # Tools separator line
        tools_sep = Window(
            height=1,
            content=FormattedTextControl(lambda: [("class:tools.sep", "  ──── tools ────")]),
        )

        # Footer — session start time
        start_str = self._session_start.strftime("%H:%M:%S")
        footer_window = Window(
            height=1,
            content=FormattedTextControl(lambda: [("class:footer", f"  Session started {start_str}")]),
        )

        # Turn N frame: response area (top) + separator + tools area (bottom)
        def turn_title() -> str:
            n = len(state.transcript)
            if n == 0:
                return "—"
            idx = state.viewed_turn_index
            if idx is None:
                return f"Turn {n}"
            return f"Turn {idx + 1} / {n}"

        turn_frame = Frame(
            body=HSplit([response_area, tools_sep, tools_area]),
            title=turn_title,
        )

        # Five-region root layout
        root = HSplit([
            header_window,
            Frame(body=HSplit([composer_input, hint_window]), title=""),
            turn_frame,
            footer_window,
        ])

        layout = Layout(root, focused_element=composer_input)

        def _set_transcript_focus(enabled: bool) -> None:
            state.transcript_focused = enabled
            if enabled:
                layout.focus(response_area)
            else:
                # Reset history view when exiting — return to live latest turn
                state.viewed_turn_index = None
                layout.focus(composer_input)

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
                _set_transcript_focus(False)
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
                _set_transcript_focus(False)
            elif state.transcript:
                _set_transcript_focus(True)
            self._refresh()

        transcript_focused_filter = Condition(lambda: state.transcript_focused)

        @kb.add("c-p", filter=transcript_focused_filter)
        def _on_ctrl_p(event) -> None:
            n = len(state.transcript)
            if n < 2:
                return
            if state.viewed_turn_index is None:
                state.viewed_turn_index = n - 2
            elif state.viewed_turn_index > 0:
                state.viewed_turn_index -= 1
            self._refresh()

        @kb.add("c-n", filter=transcript_focused_filter)
        def _on_ctrl_n(event) -> None:
            if state.viewed_turn_index is None:
                return
            n = len(state.transcript)
            if state.viewed_turn_index >= n - 1:
                state.viewed_turn_index = None
            else:
                state.viewed_turn_index += 1
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
            # Header bar — IDEA Darcula toolbar
            "header": "bg:#3c3f41 #a9b7c6",
            "header.sep": "bg:#3c3f41 #515151",
            "header.alert": "bg:#3c3f41 #cc7832",
            # Status indicators
            "status.idle": "bg:#3c3f41 #808080",
            "status.streaming": "bg:#3c3f41 #6a8759 bold",
            "status.tool": "bg:#3c3f41 #6897bb bold",
            "status.approval": "bg:#3c3f41 #ffc66d bold",
            "status.error": "bg:#3c3f41 #ff6b68 bold",
            # Frame borders
            "frame.border": "#4e5254",
            # Hint bar
            "hint": "#606366 italic",
            "hint.busy": "#6897bb italic",
            "hint.approval": "#ffc66d bold",
            "hint.approval.sep": "#515151",
            "hint.approval.call": "#cc7832",
            "hint.transcript": "#6897bb bold",
            # Tools separator
            "tools.sep": "#4e5254",
            # Footer
            "footer": "#606366 italic",
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
                        tr.output_preview = (event.output or "")[:200]
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
                if self._state.current_alert and self._state.current_alert.level == "COMPACT":
                    self._state.current_alert = None
                self._refresh()
                break

            elif isinstance(event, TurnCancelled):
                self._state.finish_turn(turn, cancelled=True)
                self._refresh()
                break

            elif isinstance(event, HistoryCompacted):
                self._state.add_alert(
                    "COMPACT",
                    f"context limit — dropped {event.messages_dropped} messages"
                    f" (retry {event.attempt}/{3})",
                )
                self._refresh()

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
