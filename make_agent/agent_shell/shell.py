"""MakeAgentShell — interactive REPL for the make-agent."""

from __future__ import annotations

import asyncio
import readline
import signal
from typing import Any

from make_agent.agent_core import (
    AgentManager,
    ConfirmEvent,
    DoneEvent,
    TokenEvent,
    ToolDoneEvent,
    ToolStartEvent,
)
from make_agent.tool_display import ToolDisplayFormatter


class MakeAgentShell:
    """Async interactive REPL that delegates all LLM interaction to an :class:`AgentManager`."""

    prompt = "make-agent> "

    def __init__(self, agent_manager: AgentManager, session_id: str) -> None:
        self._agent_manager = agent_manager
        self._session_id = session_id
        self._commands: dict[str, Any] = {
            "exit": self._cmd_exit,
            "quit": self._cmd_exit,
            "export": self._cmd_export,
            "stats": self._cmd_stats,
            "help": self._cmd_help,
        }

    # ── readline completion ────────────────────────────────────────────────

    def _setup_readline(self) -> None:
        """Configure readline so /cmd completions work."""
        try:
            readline.set_completer_delims(readline.get_completer_delims().replace("/", ""))
            readline.set_completer(self._completer)
            readline.parse_and_bind("tab: complete")
        except Exception:
            pass

    def _completer(self, text: str, state: int) -> str | None:
        if not text.startswith("/"):
            return None
        cmd_text = text[1:]
        matches = ["/" + name for name in self._commands if name.startswith(cmd_text)]
        return matches[state] if state < len(matches) else None

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

    # ── agent turn ─────────────────────────────────────────────────────────

    async def _stream_turn(self, message: str) -> None:
        """Stream one agent turn, printing events as they arrive."""
        loop = asyncio.get_running_loop()
        formatter = ToolDisplayFormatter()
        async for event in self._agent_manager.astream_events(self._session_id, message):
            if isinstance(event, TokenEvent):
                print(event.text, end="", flush=True)
            elif isinstance(event, ToolStartEvent):
                print(f"\n{formatter.format_start(event.name, event.args)}", flush=True)
                if event.description:
                    print(f"  {event.description}\n", flush=True)
            elif isinstance(event, ToolDoneEvent):
                output_preview = event.output[:200] if event.output else "(no output)"
                print(f"{formatter.format_done(event.name, output_preview, event.is_error, event.duration_ms)}", flush=True)
            elif isinstance(event, DoneEvent):
                print()  # trailing newline after streamed content
            elif isinstance(event, ConfirmEvent):
                args_repr = ", ".join(f"{k}={v!r}" for k, v in event.kwargs.items())
                prompt = f"\nAllow {event.skill_name}/{event.target}({args_repr})? [y/N] "
                answer = await loop.run_in_executor(None, input, prompt)
                if answer.strip().lower() in ("y", "yes"):
                    event.allow()
                else:
                    event.deny()

    async def _run_turn(self, message: str) -> None:
        """Run one agent turn with per-turn Ctrl-C cancellation."""
        task = asyncio.create_task(self._stream_turn(message))
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, task.cancel)
        try:
            await task
        except asyncio.CancelledError:
            print("\nCancelled.")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            loop.remove_signal_handler(signal.SIGINT)

    # ── main loop ──────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Start the interactive REPL loop."""
        self._setup_readline()
        loop = asyncio.get_running_loop()
        print("Type your message. Prefix shell commands with /  " "(e.g. /exit, /help). Press Ctrl-D or Ctrl-C twice to exit.\n")
        while True:
            try:
                line = await loop.run_in_executor(None, input, self.prompt)
            except EOFError:
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
