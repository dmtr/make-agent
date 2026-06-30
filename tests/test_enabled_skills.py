"""Tests for the --enabled-skills CLI feature."""

from __future__ import annotations

import argparse
import sys
from unittest.mock import patch

import make_agent.main as main_module
import pytest
from make_agent.builtin_tools.skill_tools import list_skills
from make_agent.main import _discover_skill_names
from make_agent.tool_handler import ToolHandler

_SKILL_MK = """\
define DESCRIPTION
A test skill
endef

.PHONY: greet
greet:
@echo hello
"""


class TestParseEnabledSkills:
    def test_none_returns_none(self):
        assert main_module._parse_enabled_skills(None, frozenset()) is None

    def test_comma_separated_list(self):
        all_names = frozenset({"git", "run", "file-list"})
        result = main_module._parse_enabled_skills("git,run", all_names)
        assert result == frozenset({"git", "run"})

    def test_single_name(self):
        result = main_module._parse_enabled_skills("git", frozenset({"git"}))
        assert result == frozenset({"git"})

    def test_unknown_name_exits(self):
        all_names = frozenset({"git", "run"})
        with patch.object(sys, "exit", side_effect=SystemExit) as mock_exit:
            try:
                main_module._parse_enabled_skills("ghost", all_names)
            except SystemExit:
                pass
        mock_exit.assert_called_once()

    def test_whitespace_stripped(self):
        result = main_module._parse_enabled_skills(" git , run ", frozenset({"git", "run"}))
        assert result == frozenset({"git", "run"})


class TestDiscoverSkillNames:
    def test_missing_dir(self, tmp_path):
        assert _discover_skill_names(str(tmp_path / "nonexistent")) == []

    def test_empty_dir(self, tmp_path):
        assert _discover_skill_names(str(tmp_path)) == []

    def test_returns_sorted_names(self, tmp_path):
        (tmp_path / "zzz").mkdir()
        (tmp_path / "zzz" / "skill.mk").write_text(_SKILL_MK)
        (tmp_path / "aaa").mkdir()
        (tmp_path / "aaa" / "skill.mk").write_text(_SKILL_MK)
        assert _discover_skill_names(str(tmp_path)) == ["aaa", "zzz"]


class TestListSkillsFiltering:
    def test_no_filter_shows_all(self, tmp_path):
        for name in ("git", "run", "file-list"):
            (tmp_path / name).mkdir()
            (tmp_path / name / "skill.mk").write_text(_SKILL_MK)
        result = list_skills(str(tmp_path))
        assert "git:" in result
        assert "run:" in result
        assert "file-list:" in result

    def test_subset_filters_correctly(self, tmp_path):
        for name in ("git", "run", "file-list"):
            (tmp_path / name).mkdir()
            (tmp_path / name / "skill.mk").write_text(_SKILL_MK)
        result = list_skills(str(tmp_path), enabled_skills=frozenset({"git", "file-list"}))
        assert "git:" in result
        assert "file-list:" in result
        assert "run:" not in result

    def test_empty_result_when_none_match(self, tmp_path):
        (tmp_path / "git").mkdir()
        (tmp_path / "git" / "skill.mk").write_text(_SKILL_MK)
        result = list_skills(str(tmp_path), enabled_skills=frozenset({"ghost"}))
        assert "No skills found" in result


class TestBackendIntegration:
    def test_list_skills_filtered(self, tmp_path):
        from make_agent.memory import Memory

        for name in ("git", "run"):
            (tmp_path / name).mkdir()
            (tmp_path / name / "skill.mk").write_text(_SKILL_MK)
        memory = Memory(tmp_path / "memory.db")
        handler = ToolHandler(str(tmp_path), memory, enabled_skills=frozenset({"git"}))
        result = handler._executors["list_skills"]()  # noqa: SLF001
        assert "git:" in result
        assert "run:" not in result

    def test_default_no_filter(self, tmp_path):
        """When enabled_skills is None (default), all skills work."""
        from make_agent.memory import Memory

        (tmp_path / "git").mkdir()
        (tmp_path / "git" / "skill.mk").write_text(_SKILL_MK)
        memory = Memory(tmp_path / "memory.db")
        handler = ToolHandler(str(tmp_path), memory)
        result = handler._executors["list_skills"]()  # noqa: SLF001
        assert "git:" in result


class TestCmdRunWithEnabledSkills:
    @pytest.fixture
    def skills_dir(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True)
        (skills_dir / "git").mkdir()
        (skills_dir / "git" / "skill.mk").write_text(_SKILL_MK)
        (skills_dir / "run").mkdir()
        (skills_dir / "run" / "skill.mk").write_text(_SKILL_MK)
        return skills_dir

    def test_enabled_skills_passed_to_backend(self, skills_dir, tmp_path):

        args = _run_args(
            prompt="do it",
            skills_dir=str(skills_dir),
            enabled_skills="git",
        )
        captured: dict = {}

        async def _fake_run(**kwargs):
            captured.update(kwargs)

        with (
            patch.object(main_module, "run", _fake_run),
            patch.object(main_module, "ensure_mode_system_prompt"),
            patch.object(main_module, "mode_dir", return_value=tmp_path / "makefile-mode"),
            patch.object(
                main_module,
                "mode_memory_path",
                return_value=tmp_path / "makefile-memory.db",
            ),
        ):
            main_module._cmd_run(args)

        handler = captured["tool_handler"]
        result = handler._executors["list_skills"]()  # noqa: SLF001
        assert "git:" in result
        assert "run:" not in result

    def test_no_enabled_skills_flag_passes_none(self, skills_dir, tmp_path):
        args = _run_args(prompt="do it", skills_dir=str(skills_dir), enabled_skills=None)
        captured: dict = {}

        async def _fake_run(**kwargs):
            captured.update(kwargs)

        with (
            patch.object(main_module, "run", _fake_run),
            patch.object(main_module, "ensure_mode_system_prompt"),
            patch.object(main_module, "mode_dir", return_value=tmp_path / "makefile-mode"),
            patch.object(
                main_module,
                "mode_memory_path",
                return_value=tmp_path / "makefile-memory.db",
            ),
        ):
            main_module._cmd_run(args)

        handler = captured["tool_handler"]
        result = handler._executors["list_skills"]()  # noqa: SLF001
        assert "git:" in result


def _run_args(**kwargs) -> argparse.Namespace:
    defaults = dict(
        model="model-x",
        prompt=None,
        prompt_file=None,
        system=None,
        system_file=None,
        max_retries=5,
        tool_timeout=600,
        max_tool_output=20000,
        max_tokens=4096,
        skills_dir=None,
        disable_builtin_tools=None,
        reasoning_effort="auto",
        skill_mode="makefile",
        prompt_cache=False,
        enabled_skills=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)
