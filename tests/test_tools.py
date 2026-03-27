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


# ── params.mk injection ───────────────────────────────────────────────────────

def test_run_tool_dollar_in_value_preserved(tmp_path):
    """$ signs in single-line values must survive Make and shell expansion.

    params.mk stores ``SPEC = result: $$(MAKE_VAR)`` so Make expands
    ``$(SPEC)`` to ``result: $(MAKE_VAR)``.  Single-quoting in the recipe
    prevents the shell from further expanding it.
    """
    mf = _write_makefile(
        tmp_path,
        """\
        MAKE_VAR = EXPANDED
        .PHONY: echo-spec
        echo-spec:
        \t@printf '%s\\n' '$(SPEC)'
    """,
    )
    result = run_tool("echo-spec", {"SPEC": "result: $(MAKE_VAR)"}, mf)
    assert "$(MAKE_VAR)" in result
    assert "EXPANDED" not in result


def test_run_tool_multiline_value(tmp_path):
    """Multiline values are written to a temp file; PARAM_FILE reaches recipe."""
    out = tmp_path / "out.txt"
    mf = _write_makefile(
        tmp_path,
        f"""\
        .PHONY: write-file
        write-file:
        \t@cat "$(CONTENT_FILE)" > "{out}"
    """,
    )
    multiline = "line one\nline two\nhas quotes and $VARS"
    run_tool("write-file", {"CONTENT": multiline}, mf)
    assert out.read_text() == multiline


def test_run_tool_params_mk_and_file_cleaned_up(tmp_path):
    """All temp files (params.mk and PARAM_FILE) are removed after the call."""
    import glob as glob_mod

    before_mk = set(glob_mod.glob("/tmp/make-agent-params-*"))
    before_content = set(glob_mod.glob("/tmp/make-agent-X-*"))
    mf = _write_makefile(
        tmp_path,
        """\
        .PHONY: noop
        noop:
        \t@true
    """,
    )
    run_tool("noop", {"X": "hello"}, mf)
    after_mk = set(glob_mod.glob("/tmp/make-agent-params-*"))
    after_content = set(glob_mod.glob("/tmp/make-agent-X-*"))
    assert after_mk == before_mk
    assert after_content == before_content


def test_run_tool_file_var_always_available(tmp_path):
    """$(PARAM_FILE) is always provided, even for single-line values."""
    out = tmp_path / "out.txt"
    mf = _write_makefile(
        tmp_path,
        f"""\
        .PHONY: save
        save:
        \t@cat "$(MSG_FILE)" > "{out}"
    """,
    )
    run_tool("save", {"MSG": "hello world"}, mf)
    assert "hello world" in out.read_text()


def test_run_tool_no_params_no_temp_files(tmp_path):
    """When there are no arguments, no temp files are created."""
    mf = _write_makefile(
        tmp_path,
        """\
        .PHONY: hello
        hello:
        \t@echo hi
    """,
    )
    result = run_tool("hello", {}, mf)
    assert "hi" in result


def test_run_tool_quotes_in_value(tmp_path):
    """Values with double quotes are passed safely via the PARAM_FILE temp file."""
    out = tmp_path / "out.txt"
    mf = _write_makefile(
        tmp_path,
        f"""\
        .PHONY: show
        show:
        \t@cat "$(MSG_FILE)" > "{out}"
    """,
    )
    run_tool("show", {"MSG": 'say "hello"'}, mf)
    assert 'say "hello"' in out.read_text()

