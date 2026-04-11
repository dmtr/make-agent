"""Tests for make_agent/builtin_tools.py."""

from __future__ import annotations

import textwrap

import pytest
from make_agent.builtin_tools import (
    BUILTIN_SCHEMAS,
    _agent_description,
    _RunAgent,
    _valid_agent_name,
    create_agent,
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


# ── _agent_description ───────────────────────────────────────────────────────

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
\t@printf '%s' "$$CONTENT" > "$(PATH)"
"""


def test_agent_description_returns_description(tmp_path):
    mk = tmp_path / "agent.mk"
    mk.write_text("define DESCRIPTION\nA file specialist.\nendef\n")
    result = _agent_description(mk)
    assert "A file specialist." in result


def test_agent_description_no_description(tmp_path):
    mk = tmp_path / "agent.mk"
    mk.write_text(_AGENT_MK)
    result = _agent_description(mk)
    assert "(no description)" in result


def test_agent_description_parse_error(tmp_path):
    mk = tmp_path / "bad.mk"
    mk.write_text("")
    result = _agent_description(mk)
    assert isinstance(result, str)


# ── list_agent ────────────────────────────────────────────────────────────────


def test_list_agent_missing_dir(tmp_path):
    result = list_agent(str(tmp_path / "nonexistent"))
    assert "No agents found" in result


def test_list_agent_empty_dir(tmp_path):
    result = list_agent(str(tmp_path))
    assert "No agents found" in result


def test_list_agent_returns_agents(tmp_path):
    (tmp_path / "search.mk").write_text("define DESCRIPTION\nSearches files by pattern.\nendef\n")
    (tmp_path / "writer.mk").write_text("define DESCRIPTION\nWrites and edits files.\nendef\n")
    result = list_agent(str(tmp_path))
    assert "search:" in result
    assert "Searches files by pattern." in result
    assert "writer:" in result
    assert "Writes and edits files." in result


def test_list_agent_sorted(tmp_path):
    (tmp_path / "zzz.mk").write_text("define DESCRIPTION\nZ agent.\nendef\n")
    (tmp_path / "aaa.mk").write_text("define DESCRIPTION\nA agent.\nendef\n")
    result = list_agent(str(tmp_path))
    assert result.index("aaa:") < result.index("zzz:")


def test_list_agent_excludes_current_agent(tmp_path):
    (tmp_path / "me.mk").write_text("define DESCRIPTION\nSelf.\nendef\n")
    (tmp_path / "other.mk").write_text("define DESCRIPTION\nOther.\nendef\n")
    result = list_agent(str(tmp_path), current_agent="me")
    assert "me:" not in result
    assert "other:" in result


def test_list_agent_no_agents_when_only_self(tmp_path):
    (tmp_path / "me.mk").write_text("define DESCRIPTION\nSelf.\nendef\n")
    result = list_agent(str(tmp_path), current_agent="me")
    assert "No agents found" in result


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
    bad_mk = textwrap.dedent(
        """\
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
    """
    )
    (tmp_path / "bad.mk").write_text(bad_mk)
    result = validate_agent("bad", str(tmp_path))
    assert "Validation errors" in result
    assert "UNUSED" in result


def test_validate_agent_no_tools(tmp_path):
    # Makefile has a rule but no # <tool> annotations
    no_tools_mk = textwrap.dedent(
        """\
        define SYSTEM_PROMPT
        I do things.
        endef

        search:
        \t@echo searching
    """
    )
    (tmp_path / "notool.mk").write_text(no_tools_mk)
    result = validate_agent("notool", str(tmp_path))
    assert "Validation errors" in result
    assert "No tools defined" in result



def test_create_agent_writes_file(tmp_path):
    result = create_agent("myagent", _AGENT_MK, "A file management agent.", str(tmp_path))
    assert result.startswith("Created agent 'myagent'")
    written = (tmp_path / "myagent.mk").read_text()
    assert "define DESCRIPTION" in written
    assert "A file management agent." in written


def test_create_agent_reports_tool_count(tmp_path):
    result = create_agent("myagent", _AGENT_MK, "A file management agent.", str(tmp_path))
    assert "2 tool(s)" in result


def test_create_agent_invalid_name(tmp_path):
    result = create_agent("../evil", _AGENT_MK, "Evil agent.", str(tmp_path))
    assert result.startswith("Error")


def test_create_agent_validation_error(tmp_path):
    bad_mk = textwrap.dedent(
        """\
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
    """
    )
    result = create_agent("bad", bad_mk, "Bad agent.", str(tmp_path))
    assert "Validation errors" in result
    assert "UNUSED" in result
    assert not (tmp_path / "bad.mk").exists()


def test_get_builtin_tools_create_agent_callable(tmp_path):
    tools = get_builtin_tools(str(tmp_path))
    result = tools["create_agent"](name="myagent", description="A file agent.", makefile=_AGENT_MK)
    assert result.startswith("Created agent 'myagent'")


# ── BUILTIN_SCHEMAS ───────────────────────────────────────────────────────────


def test_builtin_schemas_has_five_entries():
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
    assert set(by_name["create_agent"]["parameters"]["required"]) == {"name", "description", "makefile"}
    assert set(by_name["run_agent"]["parameters"]["required"]) == {"name", "prompt"}


# ── get_builtin_tools ─────────────────────────────────────────────────────────


def test_get_builtin_tools_returns_all_four():
    tools = get_builtin_tools(".agents")
    assert set(tools.keys()) == {
        "list_agent",
        "validate_agent",
        "create_agent",
        "run_agent",
        "read_file",
        "write_file",
    }


def test_get_builtin_tools_list_agent_callable(tmp_path):
    tools = get_builtin_tools(str(tmp_path))
    result = tools["list_agent"]()
    assert "No agents found" in result


def test_get_builtin_tools_validate_agent_callable(tmp_path):
    (tmp_path / "ok.mk").write_text(_AGENT_MK)
    tools = get_builtin_tools(str(tmp_path))
    result = tools["validate_agent"](name="ok")
    assert result.startswith("OK")


# ── run_agent ─────────────────────────────────────────────────────────────────


def test_run_agent_missing_mk_file(tmp_path):
    result = run_agent("ghost", "do something", str(tmp_path))
    assert "not found" in result


def test_run_agent_invalid_name(tmp_path):
    result = run_agent("../evil", "do something", str(tmp_path))
    assert result.startswith("Error")


def test_run_agent_returns_run_sentinel(tmp_path):
    (tmp_path / "worker.mk").write_text("define SYSTEM_PROMPT\nWorker.\nendef\n")
    result = run_agent("worker", "do the task", str(tmp_path))
    assert isinstance(result, _RunAgent)
    assert result.mk_path == tmp_path / "worker.mk"
    assert result.prompt == "do the task"


def test_get_builtin_tools_run_agent_returns_sentinel(tmp_path):
    (tmp_path / "worker.mk").write_text("define SYSTEM_PROMPT\nW.\nendef\n")
    tools = get_builtin_tools(str(tmp_path))
    result = tools["run_agent"](name="worker", prompt="go")
    assert isinstance(result, _RunAgent)
    assert result.prompt == "go"
