"""
AI Operator TUI Prototype
=========================

Three-region layout:
  1. Header  (1 line)   — model | tokens | live status
  2. Composer (3 lines) — prompt input at the top (command-bar model)
  3. Transcript (rest)  — scrollable; all events rendered inline

Requirements:
    pip install prompt_toolkit

Run:
    python tui.py

Demo keys:
  Enter   — submit prompt
  y / n   — approve / deny a pending skill (when card is visible)
  a       — inject a simulated harness alert
  Ctrl-Q  — quit
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame, TextArea

# =========================================================
# MODELS
# =========================================================


class AlertLevel(str, Enum):
    INFO = "INFO"
    HINT = "HINT"
    WARNING = "WARNING"
    ERROR = "ERROR"


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

ALERT_ICON = {
    AlertLevel.INFO: "ℹ",
    AlertLevel.HINT: "ℹ",
    AlertLevel.WARNING: "⚠",
    AlertLevel.ERROR: "✗",
}


@dataclass
class ToolRow:
    name: str
    args: str
    state: str = "running"  # running | done | failed
    elapsed: float = 0.0
    start_time: float = field(default_factory=time.time)


@dataclass
class ApprovalCard:
    skill_name: str
    action: str
    resolved: Optional[bool] = None  # None = pending, True = approved, False = denied

    def render(self) -> str:
        if self.resolved is not None:
            mark = "✓" if self.resolved else "✗"
            word = "approved" if self.resolved else "denied"
            return f"  {mark} {self.skill_name} {word}"
        lines = [
            "  ┌─ ⚠ Approval required " + "─" * 20 + "┐",
            f"  │  {self.skill_name}  {self.action:<38}│",
            "  │" + " " * 42 + "│",
            "  │  [Y] approve   [N] deny" + " " * 17 + "│",
            "  └" + "─" * 42 + "┘",
        ]
        return "\n".join(lines)


@dataclass
class InlineAlert:
    level: AlertLevel
    message: str

    def render(self) -> str:
        icon = ALERT_ICON[self.level]
        return f"  {icon}  {self.level.value}  {self.message}"


@dataclass
class TurnBlock:
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

        # User message box
        msg = self.user_message
        inner_width = max(len(msg) + 2, 44)
        parts.append("  ╔" + "═" * inner_width + "╗")
        parts.append(f"  ║ You: {msg:<{inner_width - 6}} ║")
        parts.append("  ╚" + "═" * inner_width + "╝")
        parts.append("")

        # Agent response
        for line in self.response_lines:
            parts.append(f"  {line}")

        # Tool rows
        for tool in self.tools:
            if tool.state == "running":
                elapsed_str = "···"
                icon = "···"
            else:
                elapsed_str = f"{tool.elapsed:.1f}s"
                icon = "✓" if tool.state == "done" else "✗"
            parts.append(f"  ▶ {tool.name}  {tool.args}  {elapsed_str} {icon}")

        # Approval card (inline, where the agent paused)
        if self.approval:
            parts.append("")
            parts.append(self.approval.render())

        # Turn footer
        if self.state in ("done", "failed", "cancelled"):
            elapsed = self.elapsed or (time.time() - self.start_time)
            sep = f"  {'─' * 12} {elapsed:.0f}s │ {self.tokens} tokens {'─' * 12}"
            parts.append("")
            parts.append(sep)

        parts.append("")
        return "\n".join(parts)


# =========================================================
# APP STATE
# =========================================================


class AppState:
    def __init__(self) -> None:
        self.model = "GEMMA"
        self.total_tokens = 0
        self.status = AgentStatus.IDLE
        self.active_tool: Optional[str] = None
        self.transcript: list[TurnBlock | InlineAlert] = []
        self.current_turn: Optional[TurnBlock] = None
        self.pending_approval: Optional[ApprovalCard] = None

    def new_turn(self, message: str) -> TurnBlock:
        turn = TurnBlock(user_message=message)
        self.transcript.append(turn)
        self.current_turn = turn
        self.status = AgentStatus.STREAMING
        return turn

    def finish_turn(self, turn: TurnBlock) -> None:
        turn.state = "done"
        turn.elapsed = time.time() - turn.start_time
        turn.tokens = max(1, len(turn.user_message.split()) * 12)
        self.total_tokens += turn.tokens
        self.status = AgentStatus.IDLE
        self.current_turn = None

    def add_alert(self, level: AlertLevel, message: str) -> None:
        self.transcript.append(InlineAlert(level=level, message=message))

    def request_approval(self, skill_name: str, action: str) -> ApprovalCard:
        card = ApprovalCard(skill_name=skill_name, action=action)
        if self.current_turn:
            self.current_turn.approval = card
        self.pending_approval = card
        self.status = AgentStatus.AWAITING_APPROVAL
        return card


state = AppState()

# =========================================================
# HEADER
# =========================================================


def header_text() -> list[tuple[str, str]]:
    label = state.status.value
    if state.status == AgentStatus.TOOL and state.active_tool:
        label = f"TOOL: {state.active_tool}"
    indicator = STATUS_INDICATOR[state.status]
    status_style = STATUS_STYLE[state.status]
    return [
        ("class:header", f" {state.model}"),
        ("class:header.sep", " │ "),
        ("class:header", f"tokens: {state.total_tokens}"),
        ("class:header.sep", " │ "),
        (status_style, f"{indicator} {label} "),
    ]


header = Window(
    height=1,
    content=FormattedTextControl(header_text),
)

# =========================================================
# TRANSCRIPT
# =========================================================


def render_transcript() -> str:
    parts: list[str] = []
    for item in state.transcript:
        parts.append(item.render())
    return "\n".join(parts)


transcript_area = TextArea(
    text="",
    scrollbar=True,
    focusable=False,
    read_only=True,
)

transcript_frame = Frame(
    body=transcript_area,
    title="TRANSCRIPT",
)

# =========================================================
# COMPOSER
# =========================================================


async def handle_prompt(text: str) -> None:
    text = text.strip()
    if not text:
        return

    turn = state.new_turn(text)
    refresh()

    # Simulate streaming response
    response = f"Processing: {text}"
    current = ""
    for ch in response:
        current += ch
        if turn.response_lines:
            turn.response_lines[-1] = current
        else:
            turn.response_lines.append(current)
        refresh()
        await asyncio.sleep(0.02)
    turn.response_lines.append("")

    # Simulate a tool for certain keywords
    if any(x in text.lower() for x in ["test", "run", "exec", "build"]):
        tool = ToolRow(name="shell.execute", args=f'cmd="{text[:24]}"')
        turn.tools.append(tool)
        state.status = AgentStatus.TOOL
        state.active_tool = "shell.execute"
        refresh()
        await asyncio.sleep(1.2)
        tool.state = "done"
        tool.elapsed = 1.2
        state.active_tool = None

    # Simulate approval request for risky actions
    if any(x in text.lower() for x in ["delete", "remove", "deploy"]):
        state.request_approval("shell.execute", f'rm "{text[:28]}"')
        refresh()
        return  # wait for approval resolution via y/n key

    state.finish_turn(turn)
    refresh()


prompt_input = TextArea(
    height=1,
    prompt="> ",
    multiline=False,
)

composer_frame = Frame(
    body=prompt_input,
    title="",
)

# =========================================================
# REFRESH
# =========================================================


def refresh() -> None:
    transcript_area.text = render_transcript()
    app.invalidate()


# =========================================================
# KEYBINDINGS
# =========================================================

kb = KeyBindings()


@kb.add("c-q")
def _(event) -> None:
    event.app.exit()


@kb.add("enter")
def _(event) -> None:
    if event.app.layout.current_window == prompt_input.window:
        text = prompt_input.text
        prompt_input.text = ""
        asyncio.create_task(handle_prompt(text))
        refresh()


@kb.add("y")
def _(event) -> None:
    """Approve a pending skill."""
    card = state.pending_approval
    if card and card.resolved is None:
        card.resolved = True
        state.pending_approval = None
        turn = state.current_turn
        if turn:
            asyncio.create_task(_finish_after_approval(turn, approved=True))
        refresh()


@kb.add("n")
def _(event) -> None:
    """Deny a pending skill."""
    card = state.pending_approval
    if card and card.resolved is None:
        card.resolved = False
        state.pending_approval = None
        turn = state.current_turn
        if turn:
            turn.state = "cancelled"
            turn.elapsed = time.time() - turn.start_time
            state.status = AgentStatus.IDLE
            state.current_turn = None
        refresh()


async def _finish_after_approval(turn: TurnBlock, *, approved: bool) -> None:
    if approved:
        await asyncio.sleep(0.8)
        turn.response_lines.append("Skill executed.")
    state.finish_turn(turn)
    refresh()


@kb.add("a")
def _(event) -> None:
    """Inject a simulated harness alert (demo)."""
    state.add_alert(
        random.choice(list(AlertLevel)),
        "Simulated harness alert",
    )
    refresh()


# =========================================================
# LAYOUT  —  three regions: header / composer / transcript
# =========================================================

root_container = HSplit(
    [
        header,
        composer_frame,
        transcript_frame,
    ]
)

layout = Layout(root_container, focused_element=prompt_input)

# =========================================================
# STYLE
# =========================================================

style = Style.from_dict(
    {
        "header": "bg:#1a1a2e #c0c0d0",
        "header.sep": "bg:#1a1a2e #555566",
        "status.idle": "bg:#1a1a2e #666677",
        "status.streaming": "bg:#1a1a2e #00cc66 bold",
        "status.tool": "bg:#1a1a2e #00aaff bold",
        "status.approval": "bg:#1a1a2e #ffaa00 bold",
        "status.error": "bg:#1a1a2e #ff4444 bold",
        "frame.border": "#444466",
    }
)

# =========================================================
# APP
# =========================================================

app = Application(
    layout=layout,
    key_bindings=kb,
    full_screen=True,
    style=style,
)


async def heartbeat() -> None:
    while True:
        refresh()
        await asyncio.sleep(1)


async def main() -> None:
    asyncio.create_task(heartbeat())
    await app.run_async()


if __name__ == "__main__":
    asyncio.run(main())
