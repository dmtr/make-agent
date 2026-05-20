"""Tests for the make-agent CLI."""

from __future__ import annotations

import argparse
import subprocess
import sys
from unittest.mock import patch

import pytest

import make_agent.main as main_module
from make_agent.builtin_tools import builtin_tool_names
from make_agent.skill_backend import MakefileSkillBackend, PythonSkillBackend


def _run(*args: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "make_agent.main", *args],
        capture_output=True,
        text=True,
        **kwargs,
    )


def _write(tmp_path, name: str, content: str):
    p = tmp_path / name
    p.write_text(content)
    return p


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
        skill_mode="python",
        compact_threshold=None,
        compact_threshold_ratio=0.7,
        compact_min_threshold=24_000,
        compact_max_threshold=120_000,
        compact_context_window=0,
        prompt_cache=False,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestRunPromptInput:
    def test_prompt_file_content_is_passed_to_run(self, tmp_path):
        prompt_file = _write(tmp_path, "prompt.txt", "hello from file")
        args = _run_args(prompt_file=str(prompt_file))
        captured: dict = {}

        async def _fake_run(**kwargs):
            captured.update(kwargs)

        with (
            patch.object(main_module, "run", _fake_run),
            patch.object(main_module, "ensure_mode_system_prompt"),
            patch.object(
                main_module, "mode_dir", return_value=tmp_path / "python-mode"
            ),
            patch.object(
                main_module,
                "mode_memory_path",
                return_value=tmp_path / "python-memory.db",
            ),
            patch.object(
                main_module,
                "default_skills_dir",
                return_value=tmp_path / "python-skills",
            ),
        ):
            main_module._cmd_run(args)

        assert captured["prompt"] == "hello from file"
        assert captured["system_prompt"] == ""
        assert isinstance(captured["tool_handler"]._backend, PythonSkillBackend)  # noqa: SLF001
        assert captured["tool_handler"].tool_names == builtin_tool_names("python")
        assert captured["memory"]._db_path == tmp_path / "python-memory.db"  # noqa: SLF001

    def test_makefile_mode_builds_makefile_backend(self, tmp_path):
        args = _run_args(prompt="do something", skill_mode="makefile")
        captured: dict = {}

        async def _fake_run(**kwargs):
            captured.update(kwargs)

        with (
            patch.object(main_module, "run", _fake_run),
            patch.object(main_module, "ensure_mode_system_prompt"),
            patch.object(
                main_module, "mode_dir", return_value=tmp_path / "makefile-mode"
            ),
            patch.object(
                main_module,
                "mode_memory_path",
                return_value=tmp_path / "makefile-memory.db",
            ),
            patch.object(
                main_module,
                "default_skills_dir",
                return_value=tmp_path / "makefile-skills",
            ),
        ):
            main_module._cmd_run(args)

        assert captured["system_prompt"] == ""
        assert captured["prompt"] == "do something"
        assert isinstance(captured["tool_handler"]._backend, MakefileSkillBackend)  # noqa: SLF001
        assert captured["tool_handler"].tool_names == builtin_tool_names("makefile")
        assert captured["memory"]._db_path == tmp_path / "makefile-memory.db"  # noqa: SLF001

    def test_custom_skills_dir_gets_mode_subfolder(self, tmp_path):
        custom_dir = tmp_path / "custom"
        args = _run_args(
            prompt="do something", skill_mode="python", skills_dir=str(custom_dir)
        )
        captured: dict = {}

        async def _fake_run(**kwargs):
            captured.update(kwargs)

        with (
            patch.object(main_module, "run", _fake_run),
            patch.object(main_module, "ensure_mode_system_prompt"),
            patch.object(
                main_module, "mode_dir", return_value=tmp_path / "python-mode"
            ),
            patch.object(
                main_module,
                "mode_memory_path",
                return_value=tmp_path / "python-memory.db",
            ),
        ):
            main_module._cmd_run(args)

        backend = captured["tool_handler"]._backend  # noqa: SLF001
        assert isinstance(backend, PythonSkillBackend)
        assert backend._skills_dir == str(custom_dir / "python")  # noqa: SLF001

    def test_custom_skills_dir_makefile_mode_gets_mode_subfolder(self, tmp_path):
        custom_dir = tmp_path / "custom"
        args = _run_args(
            prompt="do something", skill_mode="makefile", skills_dir=str(custom_dir)
        )
        captured: dict = {}

        async def _fake_run(**kwargs):
            captured.update(kwargs)

        with (
            patch.object(main_module, "run", _fake_run),
            patch.object(main_module, "ensure_mode_system_prompt"),
            patch.object(
                main_module, "mode_dir", return_value=tmp_path / "makefile-mode"
            ),
            patch.object(
                main_module,
                "mode_memory_path",
                return_value=tmp_path / "makefile-memory.db",
            ),
        ):
            main_module._cmd_run(args)

        backend = captured["tool_handler"]._backend  # noqa: SLF001
        assert isinstance(backend, MakefileSkillBackend)
        assert backend._skills_dir == str(custom_dir / "makefile")  # noqa: SLF001

    def test_compact_adaptive_args_are_passed_to_run(self, tmp_path):
        args = _run_args(
            prompt="continue",
            compact_threshold=None,
            compact_context_window=200_000,
            compact_threshold_ratio=0.65,
            compact_min_threshold=20_000,
            compact_max_threshold=100_000,
        )
        captured: dict = {}

        async def _fake_run(**kwargs):
            captured.update(kwargs)

        with (
            patch.object(main_module, "run", _fake_run),
            patch.object(main_module, "ensure_mode_system_prompt"),
            patch.object(
                main_module, "mode_dir", return_value=tmp_path / "python-mode"
            ),
            patch.object(
                main_module,
                "mode_memory_path",
                return_value=tmp_path / "python-memory.db",
            ),
            patch.object(
                main_module,
                "default_skills_dir",
                return_value=tmp_path / "python-skills",
            ),
        ):
            main_module._cmd_run(args)

        assert captured["compact_threshold"] is None
        assert captured["compact_context_window"] == 200_000
        assert captured["compact_threshold_ratio"] == 0.65
        assert captured["compact_min_threshold"] == 20_000
        assert captured["compact_max_threshold"] == 100_000

    def test_prompt_cache_flag_is_passed_to_run(self, tmp_path):
        args = _run_args(prompt="continue", prompt_cache=True)
        captured: dict = {}

        async def _fake_run(**kwargs):
            captured.update(kwargs)

        with (
            patch.object(main_module, "run", _fake_run),
            patch.object(main_module, "ensure_mode_system_prompt"),
            patch.object(main_module, "mode_dir", return_value=tmp_path / "python-mode"),
            patch.object(main_module, "mode_memory_path", return_value=tmp_path / "python-memory.db"),
            patch.object(main_module, "default_skills_dir", return_value=tmp_path / "python-skills"),
        ):
            main_module._cmd_run(args)

        assert captured["use_prompt_cache"] is True

    def test_prompt_and_prompt_file_are_mutually_exclusive(self, tmp_path):
        prompt_file = _write(tmp_path, "prompt.txt", "hello")
        result = _run(
            "run",
            "--system",
            "You are a helper.",
            "--prompt",
            "inline",
            "--prompt-file",
            str(prompt_file),
        )
        assert result.returncode != 0
        assert "not allowed with argument" in result.stderr

    def test_system_and_system_file_are_mutually_exclusive(self, tmp_path):
        system_file = _write(tmp_path, "SYSTEM.md", "You are a helper.")
        result = _run(
            "run",
            "--system",
            "inline prompt",
            "--system-file",
            str(system_file),
        )
        assert result.returncode != 0
        assert "not allowed with argument" in result.stderr


class TestResolveSystemPrompt:
    def test_system_string_takes_priority(self, tmp_path):
        args = _run_args(system="You are a helper.", system_file=None)
        result = main_module._resolve_system_prompt(args)
        assert result == "You are a helper."

    def test_system_file_is_read(self, tmp_path):
        prompt_file = tmp_path / "prompt.md"
        prompt_file.write_text("From file.")
        args = _run_args(system=None, system_file=str(prompt_file))
        result = main_module._resolve_system_prompt(args)
        assert result == "From file."

    def test_cwd_system_md_is_discovered(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "SYSTEM.md").write_text("From cwd.")
        args = _run_args(system=None, system_file=None)
        result = main_module._resolve_system_prompt(args)
        assert result == "From cwd."

    def test_mode_system_md_is_discovered(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        mode_path = tmp_path / "python"
        mode_path.mkdir()
        (mode_path / "SYSTEM.md").write_text("From mode dir.")
        args = _run_args(system=None, system_file=None, skill_mode="python")
        with patch.object(main_module, "mode_dir", return_value=mode_path):
            result = main_module._resolve_system_prompt(args)
        assert result == "From mode dir."

    def test_returns_empty_string_when_nothing_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        args = _run_args(system=None, system_file=None)
        with patch.object(
            main_module, "mode_dir", return_value=tmp_path / "nonexistent"
        ):
            result = main_module._resolve_system_prompt(args)
        assert result == ""

    def test_system_string_overrides_file(self, tmp_path):
        prompt_file = tmp_path / "prompt.md"
        prompt_file.write_text("From file.")
        args = _run_args(system="Inline prompt.", system_file=str(prompt_file))
        result = main_module._resolve_system_prompt(args)
        assert result == "Inline prompt."


class TestParseDisabledTools:
    def test_all_returns_mode_specific_names(self):
        assert main_module._parse_disabled_tools("all", "python") == builtin_tool_names(
            "python"
        )
        assert main_module._parse_disabled_tools(
            "all", "makefile"
        ) == builtin_tool_names("makefile")

    def test_unknown_name_exits_for_mode(self):
        with patch.object(sys, "exit", side_effect=SystemExit) as mock_exit:
            with patch.object(
                main_module,
                "builtin_tool_names",
                return_value=frozenset({"list_skills"}),
            ):
                try:
                    main_module._parse_disabled_tools("write_file", "python")
                except SystemExit:
                    pass
        mock_exit.assert_called_once()


class TestValidateCompactArgs:
    def test_invalid_compact_threshold_ratio_exits(self):
        args = _run_args(compact_threshold_ratio=0)
        with patch.object(sys, "exit", side_effect=SystemExit) as mock_exit:
            with pytest.raises(SystemExit):
                main_module._validate_compact_args(args)
        mock_exit.assert_called_once()

    def test_invalid_compact_bounds_exit(self):
        args = _run_args(compact_min_threshold=30_000, compact_max_threshold=20_000)
        with patch.object(sys, "exit", side_effect=SystemExit) as mock_exit:
            with pytest.raises(SystemExit):
                main_module._validate_compact_args(args)
        mock_exit.assert_called_once()
