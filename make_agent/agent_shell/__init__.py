import asyncio
import readline
import signal
from typing import Any, Optional

from make_agent.agent_core import (
    _DEFAULT_MAX_RETRIES,
    _DEFAULT_MAX_TOKENS,
    _DEFAULT_MAX_TOOL_OUTPUT,
    _DEFAULT_REASONING_EFFORT,
    _DEFAULT_TOOL_TIMEOUT,
    AgentConfig,
    AgentManager,
    DoneEvent,
    TokenEvent,
    ToolDoneEvent,
    ToolStartEvent,
)
from make_agent.app_dirs import default_skills_dir, project_dir
from make_agent.memory import Memory
from make_agent.tool_handler import ToolHandler


class MakeAgentShell:
    """Async interactive REPL that delegates all LLM interaction to an :class:`Agent`."""

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

        # Per-agent breakdown
        agents = stats.get("agents", {})
        if agents:
            print("\nPer-agent breakdown:")
            for agent_name, agent_stats in sorted(agents.items()):
                print(f"    {agent_name}:")
                print(f"      Input:  {agent_stats['input_tokens']}")
                print(f"      Output: {agent_stats['output_tokens']}")
                print(f"      Total:  {agent_stats['total_tokens']}")

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
        async for event in self._agent_manager.astream_agent(self._session_id, message):
            if isinstance(event, TokenEvent):
                print(event.text, end="", flush=True)
            elif isinstance(event, ToolStartEvent):
                print(f"\nRunning: {event.name}...", flush=True)
            elif isinstance(event, ToolDoneEvent):
                pass  # tool output visible via agent logs; keep terminal clean
            elif isinstance(event, DoneEvent):
                print()  # trailing newline after streamed content

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
        print(
            "Type your message. Prefix shell commands with /  "
            "(e.g. /exit, /help). Press Ctrl-D or Ctrl-C twice to exit.\n"
        )
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


async def run(
    system_prompt: str,
    model: str,
    memory: Memory,
    tool_handler: ToolHandler,
    prompt: Optional[str] = None,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    tool_timeout: int = _DEFAULT_TOOL_TIMEOUT,
    max_tool_output: int = _DEFAULT_MAX_TOOL_OUTPUT,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    skills_dir: str | None = None,
    disabled_builtin_tools: frozenset[str] = frozenset(),
    reasoning_effort: str = _DEFAULT_REASONING_EFFORT,
) -> None:
    """Start the interactive shell (or send a single prompt and return).

    Uses *system_prompt* as the agent's system instruction.  Enters a
    :class:`MakeAgentShell` loop.  Press Ctrl-D or type ``/exit`` to leave.
    When *prompt* is given the shell is bypassed: the prompt is sent to the
    agent and the reply is printed.
    """
    agent_config = AgentConfig(
        system_prompt=system_prompt,
        model=model,
        max_retries=max_retries,
        tool_timeout=tool_timeout,
        max_tool_output=max_tool_output,
        max_tokens=max_tokens,
        skills_dir=skills_dir or default_skills_dir(),
        disabled_builtin_tools=disabled_builtin_tools,
        reasoning_effort=reasoning_effort,
        project_dir=project_dir(),
    )
    agent_manager = AgentManager(memory, tool_handler)
    session_id = agent_manager.create_session(agent_config)
    if system_prompt:
        print("System prompt loaded.")
    else:
        print("No system prompt — using built-in defaults.")

    if prompt:
        print("Sending initial prompt...\n")
        print(await agent_manager.arun_agent(session_id, prompt))
        return

    shell = MakeAgentShell(agent_manager, session_id)
    try:
        await shell.run()
    except KeyboardInterrupt:
        print()
