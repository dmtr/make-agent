"""Tests for tool_handler.py — schema builder and make executor."""

from __future__ import annotations

from make_agent.memory import Memory
from make_agent.parser import parse
from make_agent.tool_handler import ToolHandler, build_tools


def test_build_tools_no_tool_rules():
    mf = parse("build:\n\tgcc main.c")
    assert build_tools(mf) == []


def test_build_tools_single_no_params():
    mf = parse("# <tool>\n# Build the project.\n# </tool>\nbuild:")
    tools = build_tools(mf)
    assert len(tools) == 1
    fn = tools[0]["function"]
    assert fn["name"] == "build"
    assert fn["description"] == "Build the project."
    assert fn["parameters"]["properties"] == {}
    assert fn["parameters"]["required"] == []


def test_build_tools_single_with_params():
    text = (
        "# <tool>\n"
        "# Greet someone.\n"
        "# @param NAME string The name\n"
        "# @param GREETING string The greeting\n"
        "# </tool>\n"
        "greet:"
    )
    mf = parse(text)
    tools = build_tools(mf)
    assert len(tools) == 1
    fn = tools[0]["function"]
    assert fn["name"] == "greet"
    assert fn["description"] == "Greet someone."
    assert fn["parameters"]["properties"] == {
        "NAME": {"type": "string", "description": "The name"},
        "GREETING": {"type": "string", "description": "The greeting"},
    }
    assert fn["parameters"]["required"] == ["NAME", "GREETING"]


def test_build_tools_multiple_rules():
    text = (
        "# <tool>\n# Build it.\n# </tool>\nbuild:\n"
        "# <tool>\n# Test it.\n# </tool>\ntest:\n"
        "clean:"
    )
    mf = parse(text)
    tools = build_tools(mf)
    names = [t["function"]["name"] for t in tools]
    assert names == ["build", "test"]


def test_build_tools_type_is_string():
    """The outer type wrapper is always 'function'."""
    mf = parse("# <tool>\n# Desc.\n# </tool>\nbuild:")
    assert build_tools(mf)[0]["type"] == "function"


# ── format_tool_result ────────────────────────────────────────────────────────


def format_tool_result(
    stdout: str,
    stderr: str,
    exit_code: int | None,
    max_output: int = 0,
) -> str:
    """Return the formatted output string for a tool execution.

    This is a thin wrapper around :func:`get_tool_result` that returns
    only the ``output`` portion of the :class:`ToolExecutionResult` tuple.
    """
    return ToolHandler.get_tool_result(stdout, stderr, exit_code, max_output).output


def test_format_tool_result_success():
    result = format_tool_result("hello\n", "", 0)
    assert result == "hello"


def test_format_tool_result_failure():
    result = format_tool_result("", "oops", 1)
    assert "ERROR" in result
    assert "oops" in result


def test_format_tool_result_framework_error():
    result = format_tool_result("", "timeout", None)
    assert "ERROR" in result
    assert "timeout" in result


def test_format_tool_result_no_truncation_when_under_limit():
    stdout = "x" * 100
    result = format_tool_result(stdout, "", 0, max_output=200)
    assert result == "x" * 100
    assert "omitted_chars" not in result


def test_format_tool_result_truncates_when_over_limit():
    stdout = "x" * 1000
    result = format_tool_result(stdout, "", 0, max_output=100)
    assert "omitted_chars" in result


def test_format_tool_result_unlimited_when_max_output_zero():
    stdout = "x" * 50000
    result = format_tool_result(stdout, "", 0, max_output=0)
    assert len(result) == 50000
    assert "omitted_chars" not in result


async def test_tool_handler_supports_async_backend_executors(tmp_path):
    memory = Memory(tmp_path / "memory.db")
    handler = ToolHandler(str(tmp_path), memory, base_dir=tmp_path)

    async def _async_tool(**kwargs):
        assert kwargs == {"name": "ok"}
        return "async result"

    handler._schemas.append(  # noqa: SLF001
        {
            "type": "function",
            "function": {
                "name": "async_tool",
                "description": "Async tool.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        }
    )
    handler._executors["async_tool"] = _async_tool  # noqa: SLF001

    result = await handler.execute("async_tool", {"name": "ok"})

    assert result.output == "async result"
    assert result.is_error is False
