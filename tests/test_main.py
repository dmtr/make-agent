"""Tests for the make-agent CLI — validate subcommand."""

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


class TestValidateSubcommand:
    def test_valid_makefile_exits_zero(self, tmp_path):
        mf = _write(tmp_path, "Makefile", (
            "# <tool>\n# Greet.\n# @param NAME string A name\n# </tool>\n"
            "greet:\n\t@echo $(NAME)\n"
        ))
        result = _run("validate", "-f", str(mf))
        assert result.returncode == 0
        assert "OK" in result.stdout

    def test_valid_makefile_reports_tool_count(self, tmp_path):
        mf = _write(tmp_path, "Makefile", (
            "# <tool>\n# T1.\n# @param A string a\n# </tool>\nfoo:\n\t@echo $(A)\n"
            "# <tool>\n# T2.\n# @param B string b\n# </tool>\nbar:\n\t@echo $(B)\n"
        ))
        result = _run("validate", "-f", str(mf))
        assert result.returncode == 0

    def test_broken_recipe_exits_nonzero(self, tmp_path):
        mf = _write(tmp_path, "Makefile", (
            "# <tool>\n# Install.\n# @param FILE string A file\n# </tool>\n"
            "install:\n\t@pip install -r\n"
        ))
        result = _run("validate", "-f", str(mf))
        assert result.returncode != 0

    def test_broken_recipe_reports_param_name(self, tmp_path):
        mf = _write(tmp_path, "Makefile", (
            "# <tool>\n# Install.\n# @param FILE string A file\n# </tool>\n"
            "install:\n\t@pip install -r\n"
        ))
        result = _run("validate", "-f", str(mf))
        assert "FILE" in result.stderr

    def test_multiple_errors_all_reported(self, tmp_path):
        mf = _write(tmp_path, "Makefile", (
            "# <tool>\n# T1.\n# @param A string a\n# </tool>\nfoo:\n\t@echo\n"
            "# <tool>\n# T2.\n# @param B string b\n# </tool>\nbar:\n\t@echo\n"
        ))
        result = _run("validate", "-f", str(mf))
        assert result.returncode != 0
        assert "A" in result.stderr
        assert "B" in result.stderr

    def test_no_tools_exits_zero(self, tmp_path):
        mf = _write(tmp_path, "Makefile", "build:\n\t@gcc main.c\n")
        result = _run("validate", "-f", str(mf))
        assert result.returncode == 0

    def test_missing_file_exits_nonzero(self, tmp_path):
        result = _run("validate", "-f", str(tmp_path / "nonexistent.mk"))
        assert result.returncode != 0

    def test_default_file_is_makefile(self, tmp_path):
        _write(tmp_path, "Makefile", "build:\n\t@gcc main.c\n")
        result = _run("validate", cwd=str(tmp_path))
        assert result.returncode == 0


class TestRunPromptInput:
    def test_prompt_file_content_is_passed_to_run(self, tmp_path):
        mf = _write(tmp_path, "Makefile", "noop:\n\t@echo ok\n")
        prompt_file = _write(tmp_path, "prompt.txt", "hello from file")
        args = argparse.Namespace(
            file=str(mf),
            model="model-x",
            prompt=None,
            prompt_file=str(prompt_file),
            debug=False,
            max_retries=5,
            tool_timeout=600,
            max_tool_output=20000,
            agents_dir=".agents",
        )
        captured: dict = {}

        def _fake_run(**kwargs):
            captured.update(kwargs)

        original = main_module.run
        main_module.run = _fake_run
        try:
            main_module._cmd_run(args)
        finally:
            main_module.run = original

        assert captured["prompt"] == "hello from file"
        assert str(captured["makefile_path"]).endswith("Makefile")

    def test_prompt_and_prompt_file_are_mutually_exclusive(self, tmp_path):
        mf = _write(tmp_path, "Makefile", "noop:\n\t@echo ok\n")
        prompt_file = _write(tmp_path, "prompt.txt", "hello")
        result = _run(
            "run",
            "-f",
            str(mf),
            "--prompt",
            "inline",
            "--prompt-file",
            str(prompt_file),
        )
        assert result.returncode != 0
        assert "not allowed with argument" in result.stderr
