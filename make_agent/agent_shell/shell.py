"""MakeAgentShell — interactive REPL for the make-agent."""

from __future__ import annotations

import asyncio
import signal
from contextlib import suppress
from pathlib import Path
from typing import Any
from uuid import uuid4

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings

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
from make_agent.tool_display import ToolDisplayFormatter


class MakeAgentShell:
    """Async interactive REPL that delegates all LLM interaction to an :class:`AgentManager`."""

    PROMPT = "make-agent> "

    def __init__(
        self,
        agent_manager: AgentManager,
        session_id: str,
        model: str,
        history_path: Path,
    ) -> None:
        self._agent_manager = agent_manager
        self._session_id = session_id
        self._model = model
        self._history_path = history_path
        self._current_tool: str | None = None
        self._token_count: int = 0
        self._command_queue: asyncio.Queue[ShellCommand] = asyncio.Queue()
        self._event_queue: asyncio.Queue[ShellEvent] = asyncio.Queue()
        self._commands: dict[str, Any] = {
            "exit": self._cmd_exit,
            "quit": self._cmd_exit,
            "export": self._cmd_export,
            "stats": self._cmd_stats,
            "help": self._cmd_help,
        }

    # ── prompt-toolkit setup ───────────────────────────────────────────────

    def _build_session(self) -> PromptSession:
        kb = KeyBindings()

        @kb.add("enter")
        def _(event) -> None:
            event.current_buffer.validate_and_handle()

        @kb.add("escape", "enter")
        def _(event) -> None:
            event.current_buffer.insert_text("\n")

        completer = WordCompleter(
            ["/" + name for name in self._commands],
            sentence=True,
        )

        return PromptSession(
            history=FileHistory(str(self._history_path)),
            completer=completer,
            key_bindings=kb,
            multiline=True,
            bottom_toolbar=self._toolbar,
        )

    def _toolbar(self) -> str:
        parts = [f"model: {self._model}"]
        if self._current_tool:
            parts.append(f"tool: {self._current_tool}")
        if self._token_count > 0:
            parts.append(f"tokens: {self._token_count}")
        return " | ".join(parts)

    # ── command handlers ───────────────────────────────────────────────────

    def _cmd_exit(self) -> bool:
        return True

    def _cmd_export(self) -> bool:
        path = self._agent_manager.export_conversation(self._session_id)
        if path:
            print(f"Conversation exported to {path}")
        return False

    def _cmd_stats(self) -> bool:
        stats = self._agent_manager.get_token_stats(self._session_id)
        if not stats:
            print("No token usage stats available (memory not enabled or no LLM calls yet).")
            return False
        print(f"Token usage for session {self._session_id}:")
        print(f"  Model(s):      {', '.join(stats['models'])}")
        print(f"  Input tokens:  {stats['input_tokens']}")
        print(f"  Output tokens: {stats['output_tokens']}")
        print(f"  Total tokens:  {stats['total_tokens']}")

        return False

    def _cmd_help(self) -> bool:
        print("Commands: " + "  ".join(f"/{name}" for name in self._commands))
        print("Any other input is sent to the agent. Press Ctrl-C to cancel a running turn.")
        return False

    def _dispatch_command(self, line: str) -> bool:
        """Dispatch a /command. Returns True if the shell should exit."""
        name, *_ = line.strip().split(None, 1)
        handler = self._commands.get(name)
        if handler is None:
            print(f"Unknown command: /{name}  (type /help for a list)")
            return False
        return handler()

    # ── event consumption ──────────────────────────────────────────────────

    async def _consume_turn_events(self, turn_id: str) -> None:
        """Read events from the bridge until the current turn ends."""
        formatter = ToolDisplayFormatter()
        while True:
            event = await self._event_queue.get()
            if isinstance(event, TokenEmitted):
                print(event.text, end="", flush=True)
            elif isinstance(event, ToolStarted):
                self._current_tool = event.name
                print(f"\n{formatter.format_start(event.name, event.args)}", flush=True)
                if event.description:
                    print(f"  {event.description}\n", flush=True)
            elif isinstance(event, ToolFinished):
                self._current_tool = None
                output_preview = event.output[:200] if event.output else "(no output)"
                print(
                    f"{formatter.format_done(event.name, output_preview, event.is_error, event.duration_ms)}",
                    flush=True,
                )
            elif isinstance(event, TurnFinished):
                print()
                stats = self._agent_manager.get_token_stats(self._session_id)
                if stats:
                    self._token_count = stats["total_tokens"]
                break
            elif isinstance(event, TurnCancelled):
                print("\nCancelled.")
                break
            elif isinstance(event, ManagerError):
                print(f"Error: {event.message}")
                break
            elif isinstance(event, ApprovalRequested):
                args_repr = ", ".join(f"{k}={v!r}" for k, v in event.kwargs.items())
                prompt = f"\nAllow {event.skill_name}/{event.target}({args_repr})? [y/N] "
                loop = asyncio.get_running_loop()
                answer = await loop.run_in_executor(None, input, prompt)
                if answer.strip().lower() in ("y", "yes"):
                    await self._command_queue.put(ApproveSkill(request_id=event.request_id))
                else:
                    await self._command_queue.put(DenySkill(request_id=event.request_id))
            elif isinstance(event, (TurnStarted, StatusChanged)):
                pass  # informational; no rendering needed yet

    # ── agent turn ─────────────────────────────────────────────────────────

    async def _run_turn(self, message: str) -> None:
        """Send one turn to the bridge and consume events until it finishes."""
        turn_id = str(uuid4())
        await self._command_queue.put(StartTurn(message=message, turn_id=turn_id))

        consume_task = asyncio.create_task(self._consume_turn_events(turn_id))
        event_loop = asyncio.get_running_loop()

        def _request_cancel() -> None:
            asyncio.ensure_future(self._command_queue.put(CancelTurn()))

        event_loop.add_signal_handler(signal.SIGINT, _request_cancel)
        try:
            await consume_task
        except Exception as e:
            print(f"Error: {e}")
        finally:
            event_loop.remove_signal_handler(signal.SIGINT)

    # ── main loop ──────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Start the interactive REPL loop."""
        bridge_task = asyncio.create_task(
            self._agent_manager.run_shell_bridge(
                self._session_id, self._command_queue, self._event_queue
            )
        )
        session = self._build_session()
        print(
            "Type your message. Prefix shell commands with /  "
            "(e.g. /exit, /help). Use Meta+Enter (Alt+Enter) for newlines. Press Ctrl-D or Ctrl-C to exit.\n"
        )
        try:
            while True:
                try:
                    line = await session.prompt_async(self.PROMPT)
                except EOFError:
                    print()
                    break
                except KeyboardInterrupt:
                    print()
                    break
                line = line.strip()
                if not line:
                    continue
                if line.startswith("/"):
                    should_exit = self._dispatch_command(line[1:])
                    if should_exit:
                        break
                    continue
                await self._run_turn(line)
        finally:
            await self._command_queue.put(Shutdown())
            with suppress(asyncio.CancelledError, Exception):
                await bridge_task
