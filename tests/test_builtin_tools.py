"""Tests for make_agent/builtin_tools.py."""

from __future__ import annotations

import textwrap
from unittest.mock import MagicMock, patch

import pytest

from make_agent.builtin_tools import (
    BUILTIN_SCHEMAS,
    _agent_summary,
    _valid_agent_name,
    get_builtin_tools,
    list_agent,
    run_agent,
    validate_agent,
)


# ── _valid_agent_name ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["file-search", "agent1", "my.agent", "A_B"])
def test_valid_agent_name_accepts_valid(name):
    assert _valid_agent_name(name) is True


@pytest.mark.parametrize("name", ["", "-bad", "../escape", "has space", "has/slash"])
def test_valid_agent_name_rejects_invalid(name):
    assert _valid_agent_name(name) is False


# ── _agent_summary ────────────────────────────────────────────────────────────

_AGENT_MK = """\
define SYSTEM_PROMPT
You are a file specialist.
endef

.PHONY: read-file write-file

# <tool>
# Read the contents of a file.
# @param PATH string The file path
# </tool>
read-file:
\t@cat "$(PATH)"

# <tool>
# Write content to a file.
# @param PATH string The destination path
# @param CONTENT string The content to write
# </tool>
write-file:
\t@cat "$(CONTENT_FILE)" > "$(PATH)"
"""


def test_agent_summary_includes_system_prompt(tmp_path):
    mk = tmp_path / "agent.mk"
    mk.write_text(_AGENT_MK)
    summary = _agent_summary(mk)
    assert "You are a file specialist." in summary


def test_agent_summary_includes_tool_names(tmp_path):
    mk = tmp_path / "agent.mk"
    mk.write_text(_AGENT_MK)
    summary = _agent_summary(mk)
    assert "read-file" in summary
    assert "write-file" in summary


def test_agent_summary_includes_tool_descriptions(tmp_path):
    mk = tmp_path / "agent.mk"
    mk.write_text(_AGENT_MK)
    summary = _agent_summary(mk)
    assert "Read the contents of a file." in summary
    assert "Write content to a file." in summary


def test_agent_summary_includes_param_names(tmp_path):
    mk = tmp_path / "agent.mk"
    mk.write_text(_AGENT_MK)
    summary = _agent_summary(mk)
    assert "PATH" in summary
    assert "CONTENT" in summary


def test_agent_summary_no_tools(tmp_path):
    mk = tmp_path / "agent.mk"
    mk.write_text("define SYSTEM_PROMPT\nJust a prompt.\nendef\n")
    summary = _agent_summary(mk)
    assert "Just a prompt." in summary
    assert "tools:" not in summary


def test_agent_summary_parse_error(tmp_path):
    mk = tmp_path / "bad.mk"
    mk.write_text("")
    result = _agent_summary(mk)
    # Should not raise; returns a fallback string
    assert isinstance(result, str)


# ── list_agent ────────────────────────────────────────────────────────────────

def test_list_agent_missing_dir(tmp_path):
    result = list_agent(str(tmp_path / "nonexistent"))
    assert "No agents found" in result


def test_list_agent_empty_dir(tmp_path):
    result = list_agent(str(tmp_path))
    assert "No agents found" in result


def test_list_agent_returns_agents(tmp_path):
    (tmp_path / "search.mk").write_text("define SYSTEM_PROMPT\nYou are a search specialist.\nendef\n")
    (tmp_path / "writer.mk").write_text("define SYSTEM_PROMPT\nYou are a writer.\nendef\n")
    result = list_agent(str(tmp_path))
    assert "search:" in result
    assert "You are a search specialist." in result
    assert "writer:" in result
    assert "You are a writer." in result


def test_list_agent_sorted(tmp_path):
    (tmp_path / "zzz.mk").write_text("define SYSTEM_PROMPT\nZ agent.\nendef\n")
    (tmp_path / "aaa.mk").write_text("define SYSTEM_PROMPT\nA agent.\nendef\n")
    result = list_agent(str(tmp_path))
    assert result.index("aaa:") < result.index("zzz:")


# ── validate_agent ────────────────────────────────────────────────────────────

def test_validate_agent_ok(tmp_path):
    (tmp_path / "ok.mk").write_text(_AGENT_MK)
    result = validate_agent("ok", str(tmp_path))
    assert result.startswith("OK")
    assert "2 tool(s)" in result


def test_validate_agent_not_found(tmp_path):
    result = validate_agent("ghost", str(tmp_path))
    assert "not found" in result


def test_validate_agent_invalid_name(tmp_path):
    result = validate_agent("../evil", str(tmp_path))
    assert result.startswith("Error")


def test_validate_agent_reports_errors(tmp_path):
    # param declared but never used in recipe → validation error
    bad_mk = textwrap.dedent("""\
        define SYSTEM_PROMPT
        Bad agent.
        endef

        .PHONY: do-thing

        # <tool>
        # Do something.
        # @param UNUSED string Not referenced
        # </tool>
        do-thing:
        \t@echo hello
    """)
    (tmp_path / "bad.mk").write_text(bad_mk)
    result = validate_agent("bad", str(tmp_path))
    assert "Validation errors" in result
    assert "UNUSED" in result


# ── run_agent ─────────────────────────────────────────────────────────────────

def test_run_agent_missing_mk_file(tmp_path):
    result = run_agent("ghost", "do something", str(tmp_path), "test-model")
    assert "not found" in result


def test_run_agent_invalid_name(tmp_path):
    result = run_agent("../evil", "do something", str(tmp_path), "test-model")
    assert result.startswith("Error")


def test_run_agent_success(tmp_path):
    (tmp_path / "worker.mk").write_text("define SYSTEM_PROMPT\nWorker.\nendef\n")
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "task done\n"
    mock_result.stderr = ""
    with patch("make_agent.builtin_tools.subprocess.run", return_value=mock_result) as mock_run:
        result = run_agent("worker", "do the task", str(tmp_path), "anthropic/test")
    assert result == "task done\n"
    cmd = mock_run.call_args[0][0]
    assert "-f" in cmd
    assert str(tmp_path / "worker.mk") in cmd
    assert "--prompt" in cmd
    assert "do the task" in cmd
    assert "--model" in cmd
    assert "anthropic/test" in cmd
    assert "--agents-dir" in cmd
    assert "--debug" not in cmd


def test_run_agent_passes_debug_flag(tmp_path):
    (tmp_path / "worker.mk").write_text("define SYSTEM_PROMPT\nWorker.\nendef\n")
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "done"
    mock_result.stderr = ""
    with patch("make_agent.builtin_tools.subprocess.run", return_value=mock_result) as mock_run:
        run_agent("worker", "task", str(tmp_path), "model", debug=True)
    assert "--debug" in mock_run.call_args[0][0]


def test_run_agent_nonzero_exit(tmp_path):
    (tmp_path / "bad.mk").write_text("define SYSTEM_PROMPT\nBad.\nendef\n")
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "something went wrong"
    with patch("make_agent.builtin_tools.subprocess.run", return_value=mock_result):
        result = run_agent("bad", "fail", str(tmp_path), "model")
    assert result.startswith("Error (exit 1)")
    assert "something went wrong" in result


def test_run_agent_os_error(tmp_path):
    (tmp_path / "worker.mk").write_text("define SYSTEM_PROMPT\nWorker.\nendef\n")
    with patch("make_agent.builtin_tools.subprocess.run", side_effect=OSError("not found")):
        result = run_agent("worker", "task", str(tmp_path), "model")
    assert result.startswith("Error")


# ── BUILTIN_SCHEMAS ───────────────────────────────────────────────────────────

def test_builtin_schemas_has_four_entries():
    assert len(BUILTIN_SCHEMAS) == 4


def test_builtin_schemas_names():
    names = {s["function"]["name"] for s in BUILTIN_SCHEMAS}
    assert names == {"list_agent", "validate_agent", "create_agent", "run_agent"}


def test_builtin_schemas_are_function_type():
    for schema in BUILTIN_SCHEMAS:
        assert schema["type"] == "function"


def test_builtin_schemas_required_params():
    by_name = {s["function"]["name"]: s["function"] for s in BUILTIN_SCHEMAS}
    assert by_name["list_agent"]["parameters"]["required"] == []
    assert by_name["validate_agent"]["parameters"]["required"] == ["name"]
    assert set(by_name["run_agent"]["parameters"]["required"]) == {"name", "prompt"}


# ── get_builtin_tools ─────────────────────────────────────────────────────────

def test_get_builtin_tools_returns_all_four():
    tools = get_builtin_tools(".agents", "test-model")
    assert set(tools.keys()) == {"list_agent", "validate_agent", "create_agent", "run_agent"}


def test_get_builtin_tools_list_agent_callable(tmp_path):
    tools = get_builtin_tools(str(tmp_path), "model")
    result = tools["list_agent"]()
    assert "No agents found" in result


def test_get_builtin_tools_validate_agent_callable(tmp_path):
    (tmp_path / "ok.mk").write_text(_AGENT_MK)
    tools = get_builtin_tools(str(tmp_path), "model")
    result = tools["validate_agent"](name="ok")
    assert result.startswith("OK")


def test_get_builtin_tools_run_agent_passes_model(tmp_path):
    (tmp_path / "worker.mk").write_text("define SYSTEM_PROMPT\nW.\nendef\n")
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "done"
    mock_result.stderr = ""
    with patch("make_agent.builtin_tools.subprocess.run", return_value=mock_result) as mock_run:
        tools = get_builtin_tools(str(tmp_path), "my-model")
        tools["run_agent"](name="worker", prompt="go")
    cmd = mock_run.call_args[0][0]
    assert "my-model" in cmd
