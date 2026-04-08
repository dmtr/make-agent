import cmd
import readline
from pathlib import Path
from typing import Optional

from make_agent.agent import (
    _DEFAULT_MAX_RETRIES,
    _DEFAULT_MAX_TOKENS,
    _DEFAULT_MAX_TOOL_OUTPUT,
    _DEFAULT_MODEL,
    _DEFAULT_REASONING_EFFORT,
    _DEFAULT_TOOL_TIMEOUT,
    AgentConfig,
    AgentManager,
)
from make_agent.memory import Memory


class MakeAgentShell(cmd.Cmd):
    """Interactive shell that delegates all LLM interaction to an :class:`Agent`."""

    prompt = "make-agent> "
    intro = "Welcome to the Make Agent shell! Type your message and press Enter. Type '/exit' or '/quit' to leave."

    def __init__(self, agent_manager: AgentManager, session_id: str) -> None:
        super().__init__()
        self._agent_manager = agent_manager
        self._session_id = session_id

    def preloop(self) -> None:
        """Configure readline to treat '/' as part of a word so /cmd completions work."""
        try:
            readline.set_completer_delims(readline.get_completer_delims().replace("/", ""))
        except ImportError:
            pass

    def completenames(self, text: str, *ignored) -> list[str]:
        """Complete /command names; bare words have no completions (they go to the LLM)."""
        if text.startswith("/"):
            return ["/" + name for name in super().completenames(text[1:], *ignored)]
        return []

    def parseline(self, line: str):
        """Route /commands to cmd.Cmd dispatch; everything else goes to the LLM.

        The bare string ``'EOF'`` (injected by cmdloop on Ctrl-D) is passed
        through unchanged so that :meth:`do_EOF` is still reachable.
        """
        stripped = line.strip()
        if stripped == "EOF":
            return super().parseline(stripped)
        if stripped.startswith("/"):
            return super().parseline(stripped[1:])
        return "", "", stripped

    def default(self, line: str) -> None:
        """Send *line* to the agent and print the reply."""
        try:
            print(self._agent_manager.notify_agent(self._session_id, line))
        except Exception as e:
            print(f"Error: {e}")

    def emptyline(self) -> None:
        """Do nothing on an empty line (overrides cmd.Cmd's repeat-last-command)."""

    def do_EOF(self, line: str) -> bool:
        """Exit on Ctrl-D."""
        print()
        return True

    def do_exit(self, line: str) -> bool:
        """Exit the shell."""
        return True

    def do_quit(self, line: str) -> bool:
        """Exit the shell."""
        return True

    def do_export(self, line: str) -> None:
        """Export the conversation to a timestamped HTML file in the current directory."""
        path = self._agent_manager.export_conversation(self._session_id)
        if path:
            print(f"Conversation exported to {path}")

    def do_stats(self, line: str) -> None:
        """Print aggregated token usage stats for the current session."""
        stats = self._agent_manager.get_token_stats(self._session_id)
        if not stats:
            print("No token usage stats available (memory not enabled or no LLM calls yet).")
            return
        print(f"Token usage for session {self._session_id}:")
        print(f"  Model(s):      {', '.join(stats['models'])}")
        print(f"  Input tokens:  {stats['input_tokens']}")
        print(f"  Output tokens: {stats['output_tokens']}")
        print(f"  Total tokens:  {stats['total_tokens']}")


def run(
    makefile_path: Path,
    model: str = _DEFAULT_MODEL,
    prompt: Optional[str] = None,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    tool_timeout: int = _DEFAULT_TOOL_TIMEOUT,
    max_tool_output: int = _DEFAULT_MAX_TOOL_OUTPUT,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    agents_dir: str | None = None,
    with_memory: bool = False,
    disabled_builtin_tools: frozenset[str] = frozenset(),
    reasoning_effort: str = _DEFAULT_REASONING_EFFORT,
) -> None:
    """Start the interactive shell.

    Reads the system prompt and tool definitions from *makefile_path*, then
    enters a :class:`MakeAgentShell` loop.  Press Ctrl-D, Ctrl-C, or type
    ``exit`` / ``quit`` to leave.
    """

    agent_config = AgentConfig(
        makefile_path=makefile_path,
        model=model,
        max_retries=max_retries,
        tool_timeout=tool_timeout,
        max_tool_output=max_tool_output,
        max_tokens=max_tokens,
        agents_dir=agents_dir,
        disabled_builtin_tools=disabled_builtin_tools,
        reasoning_effort=reasoning_effort,
    )
    agent_manager = AgentManager()
    session_id = agent_manager.create_session(agent_config, with_memory=with_memory)
    print(f"Loaded {makefile_path}")

    if prompt:
        print("Sending initial prompt...\n")
        print(agent_manager.notify_agent(session_id, prompt))
        return

    print("Type your message. Prefix shell commands with /  (e.g. /exit, /help). Press Ctrl-D or Ctrl-C to exit.\n")
    shell = MakeAgentShell(agent_manager, session_id)
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print()
