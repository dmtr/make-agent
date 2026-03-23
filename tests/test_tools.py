"""Tests for src/tools.py — schema builder and make executor."""

from __future__ import annotations

import textwrap
from pathlib import Path

from make_agent.parser import parse
from make_agent.tools import build_tools, run_tool


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
    text = "# <tool>\n" "# Greet someone.\n" "# @param NAME string The name\n" "# @param GREETING string The greeting\n" "# </tool>\n" "greet:"
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
    text = "# <tool>\n# Build it.\n# </tool>\nbuild:\n" "# <tool>\n# Test it.\n# </tool>\ntest:\n" "clean:"
    mf = parse(text)
    tools = build_tools(mf)
    names = [t["function"]["name"] for t in tools]
    assert names == ["build", "test"]


def test_build_tools_type_is_string():
    """The outer type wrapper is always 'function'."""
    mf = parse("# <tool>\n# Desc.\n# </tool>\nbuild:")
    assert build_tools(mf)[0]["type"] == "function"


def _write_makefile(tmp_path: Path, content: str) -> Path:
    mf = tmp_path / "Makefile"
    mf.write_text(textwrap.dedent(content))
    return mf


def test_run_tool_stdout_captured(tmp_path):
    mf = _write_makefile(
        tmp_path,
        """\
        .PHONY: hello
        hello:
        \t@echo hello world
    """,
    )
    result = run_tool("hello", {}, mf)
    assert "hello world" in result


def test_run_tool_passes_variables(tmp_path):
    mf = _write_makefile(
        tmp_path,
        """\
        .PHONY: greet
        greet:
        \t@echo $(GREETING) $(NAME)
    """,
    )
    result = run_tool("greet", {"NAME": "Alice", "GREETING": "Hi"}, mf)
    assert "Hi Alice" in result


def test_run_tool_error_on_nonzero_exit(tmp_path):
    mf = _write_makefile(
        tmp_path,
        """\
        .PHONY: fail
        fail:
        \t@exit 1
    """,
    )
    result = run_tool("fail", {}, mf)
    assert result.startswith("Error")


def test_run_tool_error_includes_stdout(tmp_path):
    """Stdout output produced before a failure must not be discarded."""
    mf = _write_makefile(
        tmp_path,
        """\
        .PHONY: partial
        partial:
        \t@echo partial output
        \t@exit 1
    """,
    )
    result = run_tool("partial", {}, mf)
    assert result.startswith("Error")
    assert "partial output" in result


def test_run_tool_error_includes_stderr(tmp_path):
    """Stderr is included in the error output."""
    mf = _write_makefile(
        tmp_path,
        """\
        .PHONY: warn
        warn:
        \t@echo error detail >&2; exit 2
    """,
    )
    result = run_tool("warn", {}, mf)
    assert result.startswith("Error (exit 2)")
    assert "error detail" in result


def test_run_tool_unknown_target(tmp_path):
    mf = _write_makefile(
        tmp_path,
        """\
        .PHONY: build
        build:
        \t@echo ok
    """,
    )
    result = run_tool("nonexistent", {}, mf)
    assert result.startswith("Error")
