"""Tests for the make-agent CLI."""

from __future__ import annotations

import argparse
import subprocess
import sys
from unittest.mock import patch

import make_agent.main as main_module


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


class TestRunPromptInput:
    def test_prompt_file_content_is_passed_to_run(self, tmp_path):
        prompt_file = _write(tmp_path, "prompt.txt", "hello from file")
        args = argparse.Namespace(
            system=None,
            system_file=None,
            model="model-x",
            prompt=None,
            prompt_file=str(prompt_file),
            loglevel="INFO",
            max_retries=5,
            tool_timeout=600,
            max_tool_output=20000,
            max_tokens=4096,
            skills_dir=None,
            disable_builtin_tools=None,
            reasoning_effort="auto",
        )
        captured: dict = {}

        async def _fake_run(**kwargs):
            captured.update(kwargs)

        original = main_module.run
        main_module.run = _fake_run
        try:
            main_module._cmd_run(args)
        finally:
            main_module.run = original

        assert captured["prompt"] == "hello from file"
        assert captured["system_prompt"] == ""

    def test_system_prompt_string_is_passed_to_run(self, tmp_path):
        args = argparse.Namespace(
            system="You are a helper.",
            system_file=None,
            model="model-x",
            prompt="do something",
            prompt_file=None,
            loglevel="INFO",
            max_retries=5,
            tool_timeout=600,
            max_tool_output=20000,
            max_tokens=4096,
            skills_dir=None,
            disable_builtin_tools=None,
            reasoning_effort="auto",
        )
        captured: dict = {}

        async def _fake_run(**kwargs):
            captured.update(kwargs)

        original = main_module.run
        main_module.run = _fake_run
        try:
            main_module._cmd_run(args)
        finally:
            main_module.run = original

        assert captured["system_prompt"] == "You are a helper."
        assert captured["prompt"] == "do something"

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


# ── _resolve_system_prompt ────────────────────────────────────────────────────


def _make_args(**kwargs) -> argparse.Namespace:
    defaults = dict(
        model=None,
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
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestResolveSystemPrompt:
    def test_system_string_takes_priority(self, tmp_path):
        args = _make_args(system="You are a helper.", system_file=None)
        result = main_module._resolve_system_prompt(args)
        assert result == "You are a helper."

    def test_system_file_is_read(self, tmp_path):
        f = tmp_path / "prompt.md"
        f.write_text("From file.")
        args = _make_args(system=None, system_file=str(f))
        result = main_module._resolve_system_prompt(args)
        assert result == "From file."

    def test_cwd_system_md_is_discovered(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "SYSTEM.md").write_text("From cwd.")
        args = _make_args(system=None, system_file=None)
        result = main_module._resolve_system_prompt(args)
        assert result == "From cwd."

    def test_returns_empty_string_when_nothing_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        args = _make_args(system=None, system_file=None)
        with patch("make_agent.main.project_dir", return_value=tmp_path / "nonexistent"):
            result = main_module._resolve_system_prompt(args)
        assert result == ""

    def test_system_string_overrides_file(self, tmp_path):
        f = tmp_path / "prompt.md"
        f.write_text("From file.")
        args = _make_args(system="Inline prompt.", system_file=str(f))
        result = main_module._resolve_system_prompt(args)
        assert result == "Inline prompt."
