"""Queue-based protocol between the agent shell and AgentManager.

Two typed asyncio.Queue instances form the boundary:
  command_queue: shell -> AgentManager   (ShellCommand items)
  event_queue:   AgentManager -> shell   (ShellEvent items)

The shell sends commands such as StartTurn or CancelTurn and reads
back events such as TokenEmitted or TurnFinished without ever touching
internal AgentManager callbacks or synchronisation primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union
from uuid import uuid4


# ── commands (shell -> manager) ───────────────────────────────────────────────


@dataclass
class StartTurn:
    """Ask the manager to begin a new agent turn with *message*."""

    message: str
    turn_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class ApproveSkill:
    """Allow a pending skill-execution request identified by *request_id*."""

    request_id: str


@dataclass
class DenySkill:
    """Deny a pending skill-execution request identified by *request_id*."""

    request_id: str


@dataclass
class CancelTurn:
    """Ask the manager to cancel the currently active turn."""


@dataclass
class Shutdown:
    """Ask the bridge to stop accepting commands and exit cleanly."""


ShellCommand = Union[StartTurn, ApproveSkill, DenySkill, CancelTurn, Shutdown]


# ── events (manager -> shell) ─────────────────────────────────────────────────


@dataclass
class TurnStarted:
    """The manager has accepted a StartTurn command and begun streaming."""

    turn_id: str


@dataclass
class TokenEmitted:
    """A partial LLM token was received."""

    turn_id: str
    text: str


@dataclass
class ToolStarted:
    """A tool call has been dispatched for execution."""

    turn_id: str
    tool_id: str
    name: str
    args: dict
    description: str = ""


@dataclass
class ToolFinished:
    """A tool call completed (successfully or with an error)."""

    turn_id: str
    tool_id: str
    name: str
    output: str
    is_error: bool
    duration_ms: float | None = None


@dataclass
class ApprovalRequested:
    """An untrusted skill requires user confirmation.

    The shell should prompt the user and then enqueue
    ApproveSkill(request_id) or DenySkill(request_id).
    """

    request_id: str
    turn_id: str
    skill_name: str
    target: str
    kwargs: dict


@dataclass
class TurnFinished:
    """The agent turn completed and produced *content* as its final response."""

    turn_id: str
    content: str


@dataclass
class TurnCancelled:
    """The active turn was cancelled (in response to CancelTurn)."""

    turn_id: str


@dataclass
class ManagerError:
    """An error occurred in the bridge or manager orchestration.

    Distinct from tool failures, which appear as ToolFinished(is_error=True).
    """

    message: str
    turn_id: str | None = None


@dataclass
class StatusChanged:
    """The manager moved between idle and busy states."""

    is_busy: bool


@dataclass
class HistoryCompacted:
    """The agent automatically dropped old turns after a context-window error."""

    turn_id: str
    attempt: int
    messages_dropped: int


ShellEvent = Union[
    TurnStarted,
    TokenEmitted,
    ToolStarted,
    ToolFinished,
    ApprovalRequested,
    HistoryCompacted,
    TurnFinished,
    TurnCancelled,
    ManagerError,
    StatusChanged,
]
