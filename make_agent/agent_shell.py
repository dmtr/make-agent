import cmd
import readline
from datetime import datetime
from pathlib import Path
from typing import Optional

from make_agent.agent import (
    _DEFAULT_MAX_RETRIES,
    _DEFAULT_MAX_TOKENS,
    _DEFAULT_MAX_TOOL_OUTPUT,
    _DEFAULT_MODEL,
    _DEFAULT_REASONING_EFFORT,
    _DEFAULT_TOOL_TIMEOUT,
    Agent,
    AgentConfig,
)
from make_agent.commands import export_conversation
from make_agent.memory import Memory


class MakeAgentShell(cmd.Cmd):
    """Interactive shell that delegates all LLM interaction to an :class:`Agent`."""

    prompt = "make-agent> "
    intro = "Welcome to the Make Agent shell! Type your message and press Enter. Type '/exit' or '/quit' to leave."

    def __init__(self, agent: Agent) -> None:
        super().__init__()
        self._agent = agent

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
            print(self._agent(line))
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
        if not self._agent.messages:
            print("Nothing to export yet.")
            return
        path = export_conversation(self._agent.messages, self._agent._model)
        print(f"Exported to {path}")


def run(
    makefile_path: Path,
    model: str = _DEFAULT_MODEL,
    prompt: Optional[str] = None,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    tool_timeout: int = _DEFAULT_TOOL_TIMEOUT,
    max_tool_output: int = _DEFAULT_MAX_TOOL_OUTPUT,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    agents_dir: str | None = None,
    memory: Memory | None = None,
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
        memory=memory,
        disabled_builtin_tools=disabled_builtin_tools,
        reasoning_effort=reasoning_effort,
    )
    agent = Agent(agent_config)
    print(f"Loaded {makefile_path}  |  tools: {agent.tool_names}")

    if prompt:
        print("Sending initial prompt...\n")
        print(agent(prompt))
        return

    print("Type your message. Prefix shell commands with /  (e.g. /exit, /help). Press Ctrl-D or Ctrl-C to exit.\n")
    shell = MakeAgentShell(agent)
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print()
