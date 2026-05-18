"""Tests for the make-agent CLI."""

from __future__ import annotations

import argparse
import subprocess
import sys

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
            reasoning_effort=None,
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
            reasoning_effort=None,
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
