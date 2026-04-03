import cmd
from pathlib import Path
from typing import Optional

from make_agent.agent import _DEFAULT_MAX_RETRIES, _DEFAULT_MAX_TOKENS, _DEFAULT_MAX_TOOL_OUTPUT, _DEFAULT_MODEL, _DEFAULT_TOOL_TIMEOUT, Agent, AgentConfig
from make_agent.memory import Memory


class MakeAgentShell(cmd.Cmd):
    """Interactive shell that delegates all LLM interaction to an :class:`Agent`."""

    prompt = "make-agent> "
    intro = ""

    def __init__(self, agent: Agent) -> None:
        super().__init__()
        self._agent = agent

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
    )
    agent = Agent(agent_config)
    print(f"Loaded {makefile_path}  |  tools: {agent.tool_names}")

    if prompt:
        print("Sending initial prompt...\n")
        print(agent(prompt))
        return

    print("Type your message. Press Ctrl-D or Ctrl-C to exit.\n")
    shell = MakeAgentShell(agent)
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print()
