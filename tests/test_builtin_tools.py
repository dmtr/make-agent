"""Tests for make_agent/builtin_tools — skill tools."""

from __future__ import annotations

import textwrap

import pytest
from make_agent.builtin_tools import (
    BUILTIN_SCHEMAS,
    _ExecuteSkill,
    _valid_skill_name,
    create_skill,
    execute_skill,
    get_builtin_tools,
    list_skills,
    read_skill,
    validate_skill,
)

_SKILL_MK = """\
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


# ── _valid_skill_name ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["file-search", "skill1", "my.skill", "A_B"])
def test_valid_skill_name_accepts_valid(name):
    assert _valid_skill_name(name) is True


@pytest.mark.parametrize("name", ["", "-bad", "../escape", "has space", "has/slash"])
def test_valid_skill_name_rejects_invalid(name):
    assert _valid_skill_name(name) is False


# ── list_skills ───────────────────────────────────────────────────────────────


def test_list_skills_missing_dir(tmp_path):
    result = list_skills(str(tmp_path / "nonexistent"))
    assert "No skills found" in result


def test_list_skills_empty_dir(tmp_path):
    result = list_skills(str(tmp_path))
    assert "No skills found" in result


def test_list_skills_returns_skills(tmp_path):
    (tmp_path / "search").mkdir()
    (tmp_path / "search" / "skill.md").write_text('---\ndescription: "Searches files by pattern."\n---\n')
    (tmp_path / "writer").mkdir()
    (tmp_path / "writer" / "skill.md").write_text('---\ndescription: "Writes and edits files."\n---\n')
    result = list_skills(str(tmp_path))
    assert "search:" in result
    assert "Searches files by pattern." in result
    assert "writer:" in result
    assert "Writes and edits files." in result


def test_list_skills_sorted(tmp_path):
    (tmp_path / "zzz").mkdir()
    (tmp_path / "zzz" / "skill.md").write_text('---\ndescription: "Z skill."\n---\n')
    (tmp_path / "aaa").mkdir()
    (tmp_path / "aaa" / "skill.md").write_text('---\ndescription: "A skill."\n---\n')
    result = list_skills(str(tmp_path))
    assert result.index("aaa:") < result.index("zzz:")


def test_list_skills_marks_has_tools(tmp_path):
    (tmp_path / "rich").mkdir()
    (tmp_path / "rich" / "skill.md").write_text('---\ndescription: "Has tools."\n---\n')
    (tmp_path / "rich" / "skill.mk").write_text(_SKILL_MK)
    result = list_skills(str(tmp_path))
    assert "[has tools]" in result


def test_list_skills_no_description_fallback(tmp_path):
    (tmp_path / "bare").mkdir()
    (tmp_path / "bare" / "skill.md").write_text("Just some text\n")
    result = list_skills(str(tmp_path))
    assert "(no description)" in result


# ── read_skill ────────────────────────────────────────────────────────────────


def test_read_skill_not_found(tmp_path):
    result = read_skill("ghost", str(tmp_path))
    assert "not found" in result


def test_read_skill_invalid_name(tmp_path):
    result = read_skill("../evil", str(tmp_path))
    assert result.startswith("Error")


def test_read_skill_missing_skill_md(tmp_path):
    (tmp_path / "broken").mkdir()
    result = read_skill("broken", str(tmp_path))
    assert "missing skill.md" in result


def test_read_skill_md_only(tmp_path):
    (tmp_path / "simple").mkdir()
    (tmp_path / "simple" / "skill.md").write_text("Follow these steps.\n")
    result = read_skill("simple", str(tmp_path))
    assert "Follow these steps." in result
    assert "(no skill.mk)" in result


def test_read_skill_with_mk(tmp_path):
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "skill.md").write_text("Instructions.\n")
    (tmp_path / "full" / "skill.mk").write_text(_SKILL_MK)
    result = read_skill("full", str(tmp_path))
    assert "Instructions." in result
    assert "read-file" in result


# ── execute_skill ─────────────────────────────────────────────────────────────


def test_execute_skill_invalid_name(tmp_path):
    result = execute_skill("../evil", str(tmp_path))
    assert result.startswith("Error")


def test_execute_skill_not_found(tmp_path):
    result = execute_skill("ghost", str(tmp_path))
    assert "not found" in result


def test_execute_skill_no_mk_returns_sentinel(tmp_path):
    (tmp_path / "simple").mkdir()
    (tmp_path / "simple" / "skill.md").write_text("Follow these steps.\n")
    result = execute_skill("simple", str(tmp_path))
    assert isinstance(result, _ExecuteSkill)
    assert "Follow these steps." in result.skill_md
    assert result.tool_schemas == []
    assert result.mk_path is None


def test_execute_skill_with_mk_returns_sentinel(tmp_path):
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "skill.md").write_text("Instructions.\n")
    (tmp_path / "full" / "skill.mk").write_text(_SKILL_MK)
    result = execute_skill("full", str(tmp_path))
    assert isinstance(result, _ExecuteSkill)
    assert result.mk_path is not None
    assert len(result.tool_schemas) == 2
    tool_names = [s["function"]["name"] for s in result.tool_schemas]
    assert "read-file" in tool_names
    assert "write-file" in tool_names


def test_execute_skill_tool_summary(tmp_path):
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "skill.md").write_text("Instructions.\n")
    (tmp_path / "full" / "skill.mk").write_text(_SKILL_MK)
    result = execute_skill("full", str(tmp_path))
    assert isinstance(result, _ExecuteSkill)
    assert "read-file" in result.tool_summary


# ── create_skill ──────────────────────────────────────────────────────────────


def test_create_skill_invalid_name(tmp_path):
    result = create_skill("../evil", "Evil skill.", "instructions", str(tmp_path))
    assert result.startswith("Error")


def test_create_skill_md_only(tmp_path):
    result = create_skill("myskill", "A test skill.", "Do this thing.", str(tmp_path))
    assert result.startswith("Created skill 'myskill'")
    assert "(no tools)" in result
    written = (tmp_path / "myskill" / "skill.md").read_text()
    assert "Do this thing." in written
    assert not (tmp_path / "myskill" / "skill.mk").exists()


def test_create_skill_auto_frontmatter(tmp_path):
    create_skill("myskill", "A test skill.", "Do this thing.", str(tmp_path))
    written = (tmp_path / "myskill" / "skill.md").read_text()
    assert written.startswith("---")
    assert 'description: "A test skill."' in written


def test_create_skill_preserves_existing_frontmatter(tmp_path):
    md = '---\ndescription: "Already there."\n---\n\nDo this.\n'
    create_skill("myskill", "Ignored desc.", md, str(tmp_path))
    written = (tmp_path / "myskill" / "skill.md").read_text()
    assert written.startswith("---")
    assert "Already there." in written


def test_create_skill_with_mk(tmp_path):
    result = create_skill("full", "A full skill.", "Instructions.", str(tmp_path), mk_content=_SKILL_MK)
    assert result.startswith("Created skill 'full'")
    assert "2 tool(s)" in result
    assert (tmp_path / "full" / "skill.mk").exists()


def test_create_skill_validation_error(tmp_path):
    bad_mk = textwrap.dedent(
        """\
        .PHONY: do-thing

        # <tool>
        # Do something.
        # @param UNUSED string Not referenced
        # </tool>
        do-thing:
        \t@echo hello
    """
    )
    result = create_skill("bad", "Bad skill.", "Instructions.", str(tmp_path), mk_content=bad_mk)
    assert "Validation errors" in result
    assert "UNUSED" in result
    assert not (tmp_path / "bad").exists()


# ── validate_skill ────────────────────────────────────────────────────────────


def test_validate_skill_invalid_name(tmp_path):
    result = validate_skill("../evil", str(tmp_path))
    assert result.startswith("Error")


def test_validate_skill_not_found(tmp_path):
    result = validate_skill("ghost", str(tmp_path))
    assert "not found" in result


def test_validate_skill_missing_md(tmp_path):
    (tmp_path / "broken").mkdir()
    result = validate_skill("broken", str(tmp_path))
    assert "missing skill.md" in result


def test_validate_skill_md_only(tmp_path):
    (tmp_path / "simple").mkdir()
    (tmp_path / "simple" / "skill.md").write_text("Instructions.\n")
    result = validate_skill("simple", str(tmp_path))
    assert result.startswith("OK")
    assert "no tools" in result


def test_validate_skill_ok_with_mk(tmp_path):
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "skill.md").write_text("Instructions.\n")
    (tmp_path / "full" / "skill.mk").write_text(_SKILL_MK)
    result = validate_skill("full", str(tmp_path))
    assert result.startswith("OK")
    assert "2 tool(s)" in result


def test_validate_skill_reports_errors(tmp_path):
    bad_mk = textwrap.dedent(
        """\
        .PHONY: do-thing

        # <tool>
        # Do something.
        # @param UNUSED string Not referenced
        # </tool>
        do-thing:
        \t@echo hello
    """
    )
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "skill.md").write_text("Instructions.\n")
    (tmp_path / "bad" / "skill.mk").write_text(bad_mk)
    result = validate_skill("bad", str(tmp_path))
    assert "Validation errors" in result
    assert "UNUSED" in result


def test_validate_skill_no_tools_error(tmp_path):
    no_tools_mk = "search:\n\t@echo searching\n"
    (tmp_path / "notool").mkdir()
    (tmp_path / "notool" / "skill.md").write_text("Instructions.\n")
    (tmp_path / "notool" / "skill.mk").write_text(no_tools_mk)
    result = validate_skill("notool", str(tmp_path))
    assert "Validation errors" in result
    assert "No tools defined" in result


# ── BUILTIN_SCHEMAS ───────────────────────────────────────────────────────────


def test_builtin_schemas_has_five_entries():
    assert len(BUILTIN_SCHEMAS) == 5


def test_builtin_schemas_names():
    names = {s["function"]["name"] for s in BUILTIN_SCHEMAS}
    assert names == {"list_skills", "read_skill", "execute_skill", "create_skill", "validate_skill"}


def test_builtin_schemas_are_function_type():
    for schema in BUILTIN_SCHEMAS:
        assert schema["type"] == "function"


def test_builtin_schemas_required_params():
    by_name = {s["function"]["name"]: s["function"] for s in BUILTIN_SCHEMAS}
    assert by_name["list_skills"]["parameters"]["required"] == []
    assert by_name["read_skill"]["parameters"]["required"] == ["name"]
    assert by_name["execute_skill"]["parameters"]["required"] == ["name"]
    assert set(by_name["create_skill"]["parameters"]["required"]) == {"name", "description", "md_content"}
    assert by_name["validate_skill"]["parameters"]["required"] == ["name"]


# ── get_builtin_tools ─────────────────────────────────────────────────────────


def test_get_builtin_tools_returns_all_five(tmp_path):
    tools = get_builtin_tools(str(tmp_path))
    assert set(tools.keys()) == {"list_skills", "read_skill", "execute_skill", "create_skill", "validate_skill"}


def test_get_builtin_tools_list_skills_callable(tmp_path):
    tools = get_builtin_tools(str(tmp_path))
    result = tools["list_skills"]()
    assert "No skills found" in result


def test_get_builtin_tools_validate_skill_callable(tmp_path):
    (tmp_path / "ok").mkdir()
    (tmp_path / "ok" / "skill.md").write_text("Instructions.\n")
    (tmp_path / "ok" / "skill.mk").write_text(_SKILL_MK)
    tools = get_builtin_tools(str(tmp_path))
    result = tools["validate_skill"](name="ok")
    assert result.startswith("OK")


def test_get_builtin_tools_execute_skill_returns_sentinel(tmp_path):
    (tmp_path / "simple").mkdir()
    (tmp_path / "simple" / "skill.md").write_text("Instructions.\n")
    tools = get_builtin_tools(str(tmp_path))
    result = tools["execute_skill"](name="simple")
    assert isinstance(result, _ExecuteSkill)
