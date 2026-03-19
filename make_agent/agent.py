"""Interactive REPL agent loop for the make-agent."""

from __future__ import annotations

import cmd
import json
import logging
from pathlib import Path
from typing import Optional

import litellm

from make_agent.parser import parse_file
from make_agent.tools import build_tools, run_tool

_DEFAULT_MODEL = "openai/gpt-4o"
_log = logging.getLogger(__name__)


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
            _log.debug("[system]\n%s", mf.system_prompt)

    @property
    def tool_names(self) -> list[str]:
        return [t["function"]["name"] for t in self._tools]

    def __call__(self, user_input: str) -> str:
        """Send *user_input* to the LLM and return the assistant's reply.

        Dispatches tool calls in a loop until the model returns a plain
        text response.
        """
        self._messages.append({"role": "user", "content": user_input})
        _log.debug("[user]\n%s", user_input)

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

                    _log.debug("[tool_call] %s args=%s", target, arguments)
                    output = run_tool(target, arguments, self._makefile_path)
                    _log.debug("[tool_result] %s -> %s", target, output)

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
                _log.debug("[assistant]\n%s", content)
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


def run(makefile_path: Path, model: str = _DEFAULT_MODEL, prompt: Optional[str] = None, debug: bool = False) -> None:
    """Start the interactive shell.

    Reads the system prompt and tool definitions from *makefile_path*, then
    enters a :class:`MakeAgentShell` loop.  Press Ctrl-D, Ctrl-C, or type
    ``exit`` / ``quit`` to leave.

    When *debug* is ``True`` all messages are logged to ``make-agent.log`` in
    the current working directory.
    """
    if debug:
        log_file = Path("make-agent.log")
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        _log.addHandler(handler)
        _log.setLevel(logging.DEBUG)
        print(f"Debug logging enabled → {log_file}\n")
    agent = Agent(makefile_path, model)
    print(f"Loaded {makefile_path}  |  tools: {agent.tool_names}")
    print("Type your message. Press Ctrl-D or Ctrl-C to exit.\n")

    if prompt:
        print("Sending initial prompt...\n")
        print(agent(prompt))
        return

    shell = MakeAgentShell(agent)
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print()
