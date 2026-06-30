"""Tests for skill modules, skill backends, and file tools."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from make_agent.builtin_tools import builtin_tool_names
from make_agent.builtin_tools.file_tools import FILE_SCHEMAS, edit_file, write_file
from make_agent.builtin_tools.skill_tools import SKILL_SCHEMAS as MAKEFILE_SKILL_SCHEMAS
from make_agent.builtin_tools.skill_tools import (
    _valid_skill_name,
    create_skill,
    execute_skill,
    list_skills,
    read_skill,
    validate_skill,
)
from make_agent.memory import Memory
from make_agent.tool_handler import ToolHandler

_SKILL_MK = """\
define DESCRIPTION
Read and write files on the filesystem
endef

.PHONY: read-file write-file

read-file:
@cat "$$FILE"

write-file:
@printf '%s' "$$CONTENT" > "$$FILE"
"""


@pytest.mark.parametrize("name", ["file-search", "skill1", "my.skill", "A_B"])
def test_valid_skill_name_accepts_valid(name):
    assert _valid_skill_name(name) is True


@pytest.mark.parametrize("name", ["", "-bad", "../escape", "has space", "has/slash"])
def test_valid_skill_name_rejects_invalid(name):
    assert _valid_skill_name(name) is False


def test_list_skills_missing_dir(tmp_path):
    result = list_skills(str(tmp_path / "nonexistent"))
    assert "No skills found" in result


def test_list_skills_empty_dir(tmp_path):
    result = list_skills(str(tmp_path))
    assert "No skills found" in result


def test_list_skills_returns_skills(tmp_path):
    (tmp_path / "search").mkdir()
    (tmp_path / "search" / "skill.mk").write_text(
        "define DESCRIPTION\nSearches files by pattern.\nendef\n\nsearch:\n\t@echo hi\n"
    )
    (tmp_path / "writer").mkdir()
    (tmp_path / "writer" / "skill.mk").write_text(
        "define DESCRIPTION\nWrites and edits files.\nendef\n\nwrite:\n\t@echo hi\n"
    )
    result = list_skills(str(tmp_path))
    assert "search:" in result
    assert "Searches files by pattern." in result
    assert "writer:" in result
    assert "Writes and edits files." in result


def test_list_skills_sorted(tmp_path):
    (tmp_path / "zzz").mkdir()
    (tmp_path / "zzz" / "skill.mk").write_text("define DESCRIPTION\nZ skill.\nendef\n")
    (tmp_path / "aaa").mkdir()
    (tmp_path / "aaa" / "skill.mk").write_text("define DESCRIPTION\nA skill.\nendef\n")
    result = list_skills(str(tmp_path))
    assert result.index("aaa:") < result.index("zzz:")


def test_list_skills_no_description_fallback(tmp_path):
    (tmp_path / "bare").mkdir()
    (tmp_path / "bare" / "skill.mk").write_text("search:\n\t@echo hi\n")
    result = list_skills(str(tmp_path))
    assert "(no description)" in result


def test_read_skill_not_found(tmp_path):
    result = read_skill("ghost", str(tmp_path))
    assert "not found" in result


def test_read_skill_invalid_name(tmp_path):
    result = read_skill("../evil", str(tmp_path))
    assert result.startswith("Error")


def test_read_skill_missing_skill_mk(tmp_path):
    (tmp_path / "broken").mkdir()
    result = read_skill("broken", str(tmp_path))
    assert "missing skill.mk" in result


def test_read_skill_returns_raw_mk(tmp_path):
    (tmp_path / "simple").mkdir()
    (tmp_path / "simple" / "skill.mk").write_text(_SKILL_MK)
    result = read_skill("simple", str(tmp_path))
    assert "define DESCRIPTION" in result
    assert "read-file" in result


async def test_execute_skill_invalid_name(tmp_path):
    result = await execute_skill("../evil", "make", str(tmp_path))
    assert result.startswith("Error")


async def test_execute_skill_not_found(tmp_path):
    result = await execute_skill("ghost", "make", str(tmp_path))
    assert "not found" in result


async def test_execute_skill_runs_default_target(tmp_path):
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "skill.mk").write_text(_SKILL_MK)
    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(b"hello world\n", b""))
    mock_proc.returncode = 0
    with patch(
        "make_agent.builtin_tools.skill_tools.asyncio.create_subprocess_exec",
        AsyncMock(return_value=mock_proc),
    ) as mock_create:
        result = await execute_skill("full", "make", str(tmp_path))
    call_args = mock_create.call_args.args
    assert "make" in call_args
    assert result == "hello world"


async def test_execute_skill_runs_named_target(tmp_path):
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "skill.mk").write_text(_SKILL_MK)
    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(b"hello world\n", b""))
    mock_proc.returncode = 0
    with patch(
        "make_agent.builtin_tools.skill_tools.asyncio.create_subprocess_exec",
        AsyncMock(return_value=mock_proc),
    ) as mock_create:
        result = await execute_skill("full", "make read-file", str(tmp_path))
    call_args = mock_create.call_args.args
    assert "read-file" in call_args
    assert result == "hello world"


async def test_execute_skill_leading_env_vars(tmp_path):
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "skill.mk").write_text(_SKILL_MK)
    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(b"ok\n", b""))
    mock_proc.returncode = 0
    with patch(
        "make_agent.builtin_tools.skill_tools.asyncio.create_subprocess_exec",
        AsyncMock(return_value=mock_proc),
    ) as mock_create:
        await execute_skill("full", "FILE=/safe/f.txt make read-file", str(tmp_path))
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["env"]["FILE"] == "/safe/f.txt"
    assert "read-file" in mock_create.call_args.args


async def test_execute_skill_failed_target(tmp_path):
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "skill.mk").write_text(_SKILL_MK)
    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(
        return_value=(b"", b"make: *** No rule to make target 'bad-target'")
    )
    mock_proc.returncode = 2
    with patch(
        "make_agent.builtin_tools.skill_tools.asyncio.create_subprocess_exec",
        AsyncMock(return_value=mock_proc),
    ):
        result = await execute_skill("full", "make bad-target", str(tmp_path))
    assert "ERROR" in result


async def test_execute_skill_invalid_env_var_name(tmp_path):
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "skill.mk").write_text(_SKILL_MK)
    result = await execute_skill("full", "bad-key=value make read-file", str(tmp_path))
    assert result.startswith("Error")


async def test_execute_skill_empty_command(tmp_path):
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "skill.mk").write_text(_SKILL_MK)
    result = await execute_skill("full", "", str(tmp_path))
    assert result.startswith("Error")


async def test_execute_skill_rejects_forbidden_make_options(tmp_path):
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "skill.mk").write_text(_SKILL_MK)
    with patch(
        "make_agent.builtin_tools.skill_tools.asyncio.create_subprocess_exec",
        AsyncMock(),
    ) as mock_create:
        result = await execute_skill(
            "full", "make -f /tmp/evil.mk read-file", str(tmp_path)
        )
    assert result.startswith("Error")
    assert "not allowed" in result
    mock_create.assert_not_called()


def test_create_skill_invalid_name(tmp_path):
    result = create_skill("../evil", _SKILL_MK, str(tmp_path))
    assert result.startswith("Error")


def test_create_skill_success(tmp_path):
    result = create_skill("myskill", _SKILL_MK, str(tmp_path))
    assert result.startswith("Created skill 'myskill'")
    written = (tmp_path / "myskill" / "skill.mk").read_text()
    assert "define DESCRIPTION" in written


def test_create_skill_rejects_symlinked_skill_dir(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, tmp_path / "myskill")
    result = create_skill("myskill", _SKILL_MK, str(tmp_path))
    assert result.startswith("Error")
    assert not (outside / "skill.mk").exists()


def test_create_skill_missing_description_block(tmp_path):
    no_desc = ".PHONY: do-thing\n\ndo-thing:\n\t@echo hello\n"
    result = create_skill("nodesc", no_desc, str(tmp_path))
    assert "DESCRIPTION" in result
    assert not (tmp_path / "nodesc").exists()


def test_create_skill_invalid_makefile(tmp_path):
    result = create_skill(
        "bad", "define DESCRIPTION\nOK\nendef\n\n{{not valid make", str(tmp_path)
    )
    assert isinstance(result, str)


def test_validate_skill_invalid_name(tmp_path):
    result = validate_skill("../evil", str(tmp_path))
    assert result.startswith("Error")


def test_validate_skill_not_found(tmp_path):
    result = validate_skill("ghost", str(tmp_path))
    assert "not found" in result


def test_validate_skill_missing_mk(tmp_path):
    (tmp_path / "broken").mkdir()
    result = validate_skill("broken", str(tmp_path))
    assert "missing skill.mk" in result


def test_validate_skill_ok(tmp_path):
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "skill.mk").write_text(_SKILL_MK)
    result = validate_skill("full", str(tmp_path))
    assert result.startswith("OK")


def test_validate_skill_missing_description(tmp_path):
    no_desc = "search:\n\t@echo searching\n"
    (tmp_path / "notool").mkdir()
    (tmp_path / "notool" / "skill.mk").write_text(no_desc)
    result = validate_skill("notool", str(tmp_path))
    assert "DESCRIPTION" in result


def test_makefile_skill_schemas_names():
    names = {schema["function"]["name"] for schema in MAKEFILE_SKILL_SCHEMAS}
    assert names == {
        "list_skills",
        "read_skill",
        "execute_skill",
        "create_skill",
        "validate_skill",
    }


def test_makefile_skill_schemas_required_params():
    by_name = {
        schema["function"]["name"]: schema["function"]
        for schema in MAKEFILE_SKILL_SCHEMAS
    }
    assert by_name["list_skills"]["parameters"]["required"] == []
    assert by_name["read_skill"]["parameters"]["required"] == ["name"]
    assert set(by_name["execute_skill"]["parameters"]["required"]) == {
        "name",
        "command",
    }
    assert set(by_name["create_skill"]["parameters"]["required"]) == {
        "name",
        "mk_content",
    }
    assert by_name["validate_skill"]["parameters"]["required"] == ["name"]


def test_file_schemas_structure():
    assert len(FILE_SCHEMAS) == 2
    names = {schema["function"]["name"] for schema in FILE_SCHEMAS}
    assert names == {"write_file", "edit_file"}
    by_name = {
        schema["function"]["name"]: schema["function"] for schema in FILE_SCHEMAS
    }
    assert set(by_name["write_file"]["parameters"]["required"]) == {"path", "content"}
    assert set(by_name["edit_file"]["parameters"]["required"]) == {
        "path",
        "old_text",
        "new_text",
    }


def test_builtin_tool_names_makefile():
    assert builtin_tool_names("makefile") == {
        "list_skills",
        "read_skill",
        "execute_skill",
        "create_skill",
        "validate_skill",
        "write_file",
        "edit_file",
        "search_user_memory",
        "search_agent_memory",
        "get_recent_messages",
    }


def test_builtin_tool_names_unsupported_mode():
    with pytest.raises(ValueError, match="unsupported skill mode"):
        builtin_tool_names("python")


def test_tool_handler_returns_expected_tools(tmp_path):
    memory = Memory(tmp_path / "memory.db")
    handler = ToolHandler(str(tmp_path), memory, base_dir=tmp_path)
    expected = {
        "list_skills",
        "read_skill",
        "execute_skill",
        "create_skill",
        "validate_skill",
        "write_file",
        "edit_file",
        "search_user_memory",
        "search_agent_memory",
        "get_recent_messages",
    }
    assert handler.tool_names == expected


def test_tool_handler_list_skills_callable(tmp_path):
    memory = Memory(tmp_path / "memory.db")
    handler = ToolHandler(str(tmp_path), memory, base_dir=tmp_path)
    result = handler._executors["list_skills"]()  # noqa: SLF001
    assert "No skills found" in result


def test_tool_handler_validate_skill_callable(tmp_path):
    (tmp_path / "ok").mkdir()
    (tmp_path / "ok" / "skill.mk").write_text(_SKILL_MK)
    memory = Memory(tmp_path / "memory.db")
    handler = ToolHandler(str(tmp_path), memory, base_dir=tmp_path)
    result = handler._executors["validate_skill"](name="ok")  # noqa: SLF001
    assert result.startswith("OK")


async def test_tool_handler_execute_skill_callable(tmp_path):
    (tmp_path / "simple").mkdir()
    (tmp_path / "simple" / "skill.mk").write_text(_SKILL_MK)
    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(b"done\n", b""))
    mock_proc.returncode = 0
    memory = Memory(tmp_path / "memory.db")
    handler = ToolHandler(str(tmp_path), memory, base_dir=tmp_path)
    with patch(
        "make_agent.builtin_tools.skill_tools.asyncio.create_subprocess_exec",
        AsyncMock(return_value=mock_proc),
    ):
        result = await handler._executors["execute_skill"](  # noqa: SLF001
            name="simple", command="make read-file"
        )
    assert result == "done"


def test_write_file_creates_new_file(tmp_path):
    result = write_file("hello.txt", "hello world", tmp_path)
    assert "Successfully wrote" in result
    assert (tmp_path / "hello.txt").read_text() == "hello world"


def test_write_file_overwrites_existing(tmp_path):
    (tmp_path / "f.txt").write_text("old content")
    write_file("f.txt", "new content", tmp_path)
    assert (tmp_path / "f.txt").read_text() == "new content"


def test_write_file_creates_parent_dirs(tmp_path):
    result = write_file("a/b/c.txt", "deep", tmp_path)
    assert "Successfully wrote" in result
    assert (tmp_path / "a" / "b" / "c.txt").read_text() == "deep"


def test_write_file_rejects_traversal(tmp_path):
    result = write_file("../../evil.txt", "x", tmp_path)
    assert result.startswith("Error")


def test_edit_file_replaces_first_occurrence(tmp_path):
    (tmp_path / "code.py").write_text("foo bar foo")
    result = edit_file("code.py", "foo", "baz", tmp_path)
    assert "Successfully replaced" in result
    assert (tmp_path / "code.py").read_text() == "baz bar foo"


def test_edit_file_missing_file(tmp_path):
    result = edit_file("ghost.txt", "x", "y", tmp_path)
    assert result.startswith("Error")
    assert "does not exist" in result


def test_edit_file_text_not_found(tmp_path):
    (tmp_path / "f.txt").write_text("hello world")
    result = edit_file("f.txt", "missing text", "replacement", tmp_path)
    assert result.startswith("Error")
    assert "not found" in result


def test_edit_file_rejects_traversal(tmp_path):
    result = edit_file("../../evil.txt", "x", "y", tmp_path)
    assert result.startswith("Error")


def test_tool_handler_write_file_callable(tmp_path):
    memory = Memory(tmp_path / "memory.db")
    handler = ToolHandler(str(tmp_path), memory, base_dir=tmp_path)
    result = handler._executors["write_file"](path="out.txt", content="hello")  # noqa: SLF001
    assert "Successfully wrote" in result
    assert (tmp_path / "out.txt").read_text() == "hello"


def test_tool_handler_edit_file_callable(tmp_path):
    (tmp_path / "src.py").write_text("x = 1")
    memory = Memory(tmp_path / "memory.db")
    handler = ToolHandler(str(tmp_path), memory, base_dir=tmp_path)
    result = handler._executors["edit_file"](  # noqa: SLF001
        path="src.py", old_text="x = 1", new_text="x = 2"
    )
    assert "Successfully replaced" in result
    assert (tmp_path / "src.py").read_text() == "x = 2"
