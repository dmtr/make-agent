"""Tests for src/tools.py — schema builder and make executor."""

from __future__ import annotations

import textwrap
from pathlib import Path

from make_agent.parser import parse
from make_agent.tools import build_tools, get_tool_result, run_tool


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
    assert "ERROR" in result.output


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
    assert "ERROR" in result.output
    assert "partial output" in result.output


def test_run_tool_error_includes_stderr(tmp_path):
    """Stderr is reported separately from stdout."""
    mf = _write_makefile(
        tmp_path,
        """\
        .PHONY: warn
        warn:
        \t@echo error detail >&2; exit 2
    """,
    )
    result = run_tool("warn", {}, mf)
    assert "ERROR" in result.output
    assert "error detail" in result.output


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
    assert "ERROR" in result.output


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
    assert "exceeded" in result.output
    assert "slow" in result.output


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
    assert "not a valid make variable name" in result.output


def test_run_tool_rejects_system_env_var_override(tmp_path):
    """Arguments must not be allowed to shadow existing environment variables."""
    mf = _write_makefile(
        tmp_path,
        """\
        .PHONY: noop
        noop:
        \t@true
    """,
    )
    result = run_tool("noop", {"PATH": "/evil/bin"}, mf)
    assert "shadows the system environment variable" in result.output


# ── params.mk injection ───────────────────────────────────────────────────────


def test_run_tool_param_accessible_via_shell_var(tmp_path):
    """Single-line params are accessible as $$PARAM (shell env var) in recipes."""
    mf = _write_makefile(
        tmp_path,
        """\
        .PHONY: greet
        greet:
        \t@printf '%s' "$$NAME"
    """,
    )
    result = run_tool("greet", {"NAME": "Alice"}, mf)
    assert "Alice" in result


def test_run_tool_multiline_value(tmp_path):
    """Multiline values are available as $$PARAM via the env var mechanism."""
    out = tmp_path / "out.txt"
    mf = _write_makefile(
        tmp_path,
        f"""\
        .PHONY: write-file
        write-file:
        \t@printf '%s' "$$CONTENT" > "{out}"
    """,
    )
    multiline = "line one\nline two\nline three"
    run_tool("write-file", {"CONTENT": multiline}, mf)
    # Verify the file was written correctly
    assert out.read_text() == multiline


def test_run_tool_no_temp_files_created(tmp_path):
    """No temporary files are ever created — all params go via env vars."""
    mf = _write_makefile(
        tmp_path,
        """\
        .PHONY: noop
        noop:
        \t@true
    """,
    )
    before = set(tmp_path.iterdir())
    run_tool("noop", {"X": "hello\nworld"}, mf)
    after = set(tmp_path.iterdir())
    assert after == before


def test_run_tool_endef_in_multiline_value(tmp_path):
    """A multiline value containing a bare 'endef' line is passed correctly."""
    out = tmp_path / "out.txt"
    mf = _write_makefile(
        tmp_path,
        f"""\
        .PHONY: write-file
        write-file:
        \t@printf '%s' "$$CONTENT" > "{out}"
    """,
    )
    value = "before\nendef\nafter"
    run_tool("write-file", {"CONTENT": value}, mf)
    assert "before" in out.read_text()
    assert "after" in out.read_text()


def test_run_tool_quotes_in_value(tmp_path):
    """Values with double quotes are passed correctly via env vars."""
    out = tmp_path / "out.txt"
    mf = _write_makefile(
        tmp_path,
        f"""\
        .PHONY: show
        show:
        \t@printf '%s' "$$MSG" > "{out}"
    """,
    )
    run_tool("show", {"MSG": 'say "hello"'}, mf)
    assert 'say "hello"' in out.read_text()


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
    return get_tool_result(stdout, stderr, exit_code, max_output).output


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


def test_run_tool_truncates_output(tmp_path):
    mf = _write_makefile(
        tmp_path,
        """\
        .PHONY: big
        big:
        \t@python3 -c "print('a' * 500)"
    """,
    )
    result = run_tool("big", {}, mf, max_output=100)
    assert len(result.output) == 100
    assert "omitted_chars" in result.output
