"""Interactive REPL agent loop for the make-agent."""

from __future__ import annotations

import cmd
import json
from pathlib import Path

import litellm

from make_agent.parser import parse_file
from make_agent.tools import build_tools, run_tool

_DEFAULT_MODEL = "openai/gpt-4o"


class Agent:
    """LLM agent that maintains conversation history and dispatches tool calls.

    Call the instance with a user message to get the assistant's reply::

        agent = Agent(Path("Makefile"), model="openai/gpt-4o")
        reply = agent("List the files in the current directory.")
    """

    def __init__(self, makefile_path: Path, model: str = _DEFAULT_MODEL) -> None:
        mf = parse_file(makefile_path)
        self._model = model
        self._makefile_path = makefile_path
        self._tools = build_tools(mf)
        self._tool_kwargs: dict = {"tools": self._tools, "tool_choice": "auto"} if self._tools else {}
        self._messages: list[dict] = []
        if mf.system_prompt:
            self._messages.append({"role": "system", "content": mf.system_prompt})

    @property
    def tool_names(self) -> list[str]:
        return [t["function"]["name"] for t in self._tools]

    def __call__(self, user_input: str) -> str:
        """Send *user_input* to the LLM and return the assistant's reply.

        Dispatches tool calls in a loop until the model returns a plain
        text response.
        """
        self._messages.append({"role": "user", "content": user_input})

        while True:
            response = litellm.completion(
                model=self._model,
                messages=self._messages,
                **self._tool_kwargs,
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                self._messages.append(msg.model_dump(exclude_none=True))

                for tc in msg.tool_calls:
                    target = tc.function.name
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                    output = run_tool(target, arguments, self._makefile_path)

                    self._messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": output,
                        }
                    )
            else:
                content = msg.content or ""
                self._messages.append({"role": "assistant", "content": content})
                return content


class MakeAgentShell(cmd.Cmd):
    """Interactive shell that delegates all LLM interaction to an :class:`Agent`."""

    prompt = "make-agent> "
    intro = ""

    def __init__(self, agent: Agent) -> None:
        super().__init__()
        self._agent = agent

    def default(self, line: str) -> None:
        """Send *line* to the agent and print the reply."""
        print(self._agent(line))

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


def run(makefile_path: Path, model: str = _DEFAULT_MODEL) -> None:
    """Start the interactive shell.

    Reads the system prompt and tool definitions from *makefile_path*, then
    enters a :class:`MakeAgentShell` loop.  Press Ctrl-D, Ctrl-C, or type
    ``exit`` / ``quit`` to leave.
    """
    agent = Agent(makefile_path, model)
    print(f"Loaded {makefile_path}  |  tools: {agent.tool_names}")
    print("Type your message. Press Ctrl-D or Ctrl-C to exit.\n")

    shell = MakeAgentShell(agent)
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print()
