"""Tests for src/tools.py — schema builder and make executor."""

from __future__ import annotations

import textwrap
from pathlib import Path

from make_agent.parser import parse
from make_agent.tools import build_tools, get_content_params, run_tool


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


def test_build_tools_content_param_schema():
    """content-typed params appear as JSON Schema string with an extra hint."""
    text = (
        "# <tool>\n# Write a file.\n"
        "# @param FILE string Destination path\n"
        "# @param CONTENT content Text to write\n"
        "# </tool>\nwrite-file:"
    )
    mf = parse(text)
    props = build_tools(mf)[0]["function"]["parameters"]["properties"]
    assert props["FILE"] == {"type": "string", "description": "Destination path"}
    content_prop = props["CONTENT"]
    assert content_prop["type"] == "string"
    assert "CONTENT" in build_tools(mf)[0]["function"]["parameters"]["required"]
    assert "Pass the full text as-is" in content_prop["description"]


def test_get_content_params_returns_mapping():
    text = (
        "# <tool>\n# Write a file.\n"
        "# @param FILE string Destination\n"
        "# @param CONTENT content Text\n"
        "# </tool>\nwrite-file:\n"
        "# <tool>\n# List files.\n"
        "# @param DIR string Directory\n"
        "# </tool>\nlist-files:\n"
    )
    mf = parse(text)
    mapping = get_content_params(mf)
    assert mapping == {"write-file": frozenset({"CONTENT"})}


def test_get_content_params_empty_when_none():
    mf = parse("# <tool>\n# Greet.\n# @param NAME string Name\n# </tool>\ngreet:")
    assert get_content_params(mf) == {}

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


def test_run_tool_dollar_in_argument_not_expanded(tmp_path):
    """$ signs in argument values must reach the recipe literally, not be
    expanded by Make as variable references (e.g. $(Name) must survive)."""
    mf = _write_makefile(
        tmp_path,
        """\
        .PHONY: echo-spec
        echo-spec:
        \t@printf '%s' "$$SPEC"
    """,
    )
    mf.write_text(
        textwrap.dedent(
            """\
            export SPEC
            .PHONY: echo-spec
            echo-spec:
            \t@printf '%s' "$$SPEC"
            """
        )
    )
    result = run_tool("echo-spec", {"SPEC": "recipe: $(Name)"}, mf)
    assert "$(Name)" in result


def test_run_tool_timeout(tmp_path):
    mf = _write_makefile(
        tmp_path,
        """\
        .PHONY: slow
        slow:
        \t@sleep 10
    """,
    )
    result = run_tool("slow", {}, mf, timeout=1)
    assert "timeout" in result.lower()
    assert "slow" in result


# ── content_params — temp-file injection ──────────────────────────────────────

def test_run_tool_content_param_writes_file(tmp_path):
    """A content param is written to a temp file; CONTENT_FILE reaches the recipe."""
    out = tmp_path / "out.txt"
    mf = _write_makefile(
        tmp_path,
        f"""\
        .PHONY: write-file
        write-file:
        \t@cat "$(CONTENT_FILE)" > "{out}"
    """,
    )
    multiline = "line one\nline two\nhas \"quotes\" and $VARS"
    run_tool("write-file", {"CONTENT": multiline}, mf, content_params=frozenset({"CONTENT"}))
    assert out.read_text() == multiline


def test_run_tool_content_param_temp_file_is_cleaned_up(tmp_path):
    """Temporary files for content params are removed after the call."""
    import glob as glob_mod

    before = set(glob_mod.glob("/tmp/make-agent-CONTENT-*"))
    mf = _write_makefile(
        tmp_path,
        """\
        .PHONY: noop
        noop:
        \t@cat "$(CONTENT_FILE)" > /dev/null
    """,
    )
    run_tool("noop", {"CONTENT": "hello"}, mf, content_params=frozenset({"CONTENT"}))
    after = set(glob_mod.glob("/tmp/make-agent-CONTENT-*"))
    assert after == before  # no new temp files remain


def test_run_tool_content_param_handles_quotes_and_newlines(tmp_path):
    """Content with unbalanced quotes and newlines reaches the file intact."""
    out = tmp_path / "script.py"
    mf = _write_makefile(
        tmp_path,
        f"""\
        .PHONY: write-file
        write-file:
        \t@cat "$(CONTENT_FILE)" > "{out}"
    """,
    )
    python_code = '#!/usr/bin/env python3\nprint("hello")\ndata = \'world\'\n'
    run_tool("write-file", {"CONTENT": python_code}, mf, content_params=frozenset({"CONTENT"}))
    assert out.read_text() == python_code


def test_run_tool_non_content_params_unchanged(tmp_path):
    """Non-content params are still passed as NAME=value (no temp file)."""
    mf = _write_makefile(
        tmp_path,
        """\
        .PHONY: greet
        greet:
        \t@echo "$(NAME)"
    """,
    )
    result = run_tool("greet", {"NAME": "Alice"}, mf, content_params=frozenset())
    assert "Alice" in result


def test_run_tool_rejects_invalid_argument_name(tmp_path):
    """Argument names must be make-variable-safe to block option injection."""
    mf = _write_makefile(
        tmp_path,
        """\
        .PHONY: greet
        greet:
        \t@echo ok
    """,
    )
    result = run_tool("greet", {"--file": "x"}, mf)
    assert result.startswith("Error (invalid argument name)")


def test_run_tool_escapes_shell_sensitive_value_chars(tmp_path):
    """Backticks and $(...) payloads must stay literal, not execute."""
    marker_one = tmp_path / "pwned-backtick"
    marker_two = tmp_path / "pwned-subst"
    mf = _write_makefile(
        tmp_path,
        f"""\
        .PHONY: show
        show:
        \t@printf '%s\\n' "$(TASK)"
        \t@test ! -e "{marker_one}"
        \t@test ! -e "{marker_two}"
    """,
    )
    payload = f'hello `touch "{marker_one}"` $(touch "{marker_two}") world'
    result = run_tool("show", {"TASK": payload}, mf)
    assert marker_one.exists() is False
    assert marker_two.exists() is False
    assert payload in result
