"""Tests for make_agent.settings — load, save, and main.py integration."""

from __future__ import annotations

import argparse
from unittest.mock import patch

import pytest
import yaml

import make_agent.settings as settings_module
import make_agent.main as main_module
from make_agent.settings import load_settings, save_settings


# ── Helpers ───────────────────────────────────────────────────────────────────


def _patch_settings_file(tmp_path):
    """Return a context manager that redirects settings_file() to tmp_path/settings.yaml."""
    return patch.object(settings_module, "settings_file", return_value=tmp_path / "settings.yaml")


# ── load_settings ─────────────────────────────────────────────────────────────


class TestLoadSettings:
    def test_returns_none_when_file_missing(self, tmp_path):
        with _patch_settings_file(tmp_path):
            assert load_settings() is None

    def test_returns_model(self, tmp_path):
        (tmp_path / "settings.yaml").write_text("model: mymodel\n")
        with _patch_settings_file(tmp_path):
            result = load_settings()
        assert result == {"model": "mymodel"}

    def test_partial_settings_only_model(self, tmp_path):
        (tmp_path / "settings.yaml").write_text("model: mymodel\n")
        with _patch_settings_file(tmp_path):
            result = load_settings()
        assert result == {"model": "mymodel"}

    def test_makefile_key_is_ignored(self, tmp_path):
        (tmp_path / "settings.yaml").write_text("makefile: special.mk\n")
        with _patch_settings_file(tmp_path):
            result = load_settings()
        assert result == {}

    def test_unknown_keys_are_ignored(self, tmp_path):
        (tmp_path / "settings.yaml").write_text("model: m\nunknown: x\n")
        with _patch_settings_file(tmp_path):
            result = load_settings()
        assert "unknown" not in result

    def test_empty_file_returns_empty_dict(self, tmp_path):
        (tmp_path / "settings.yaml").write_text("")
        with _patch_settings_file(tmp_path):
            result = load_settings()
        assert result == {}


# ── save_settings ─────────────────────────────────────────────────────────────


class TestSaveSettings:
    def test_writes_yaml_file(self, tmp_path):
        with _patch_settings_file(tmp_path):
            save_settings({"model": "m"})
        data = yaml.safe_load((tmp_path / "settings.yaml").read_text())
        assert data == {"model": "m"}

    def test_overwrites_existing_file(self, tmp_path):
        (tmp_path / "settings.yaml").write_text("model: old\n")
        with _patch_settings_file(tmp_path):
            save_settings({"model": "new"})
        data = yaml.safe_load((tmp_path / "settings.yaml").read_text())
        assert data["model"] == "new"


# ── _resolve_run_args (main.py integration) ───────────────────────────────────


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
        with_memory=False,
        disable_builtin_tools=None,
        reasoning_effort=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestResolveRunArgs:
    def test_cli_model_overrides_settings(self):
        with patch("make_agent.main.load_settings", return_value={"model": "settings-model"}):
            args = _make_args(model="cli-model")
            result = main_module._resolve_run_args(args)
        assert result.model == "cli-model"

    def test_settings_model_used_when_no_cli_model(self):
        with patch("make_agent.main.load_settings", return_value={"model": "settings-model"}):
            args = _make_args(model=None)
            result = main_module._resolve_run_args(args)
        assert result.model == "settings-model"

    def test_model_is_none_when_no_settings_and_no_cli(self):
        with patch("make_agent.main.load_settings", return_value={}):
            args = _make_args(model=None)
            result = main_module._resolve_run_args(args)
        assert result.model is None

    def test_reasoning_effort_default_auto_when_not_in_settings(self):
        with patch("make_agent.main.load_settings", return_value={"model": "m"}):
            args = _make_args(model="m")
            result = main_module._resolve_run_args(args)
        assert result.reasoning_effort == "auto"

    def test_reasoning_effort_from_settings(self):
        with patch("make_agent.main.load_settings", return_value={"model": "m", "reasoning_effort": "low"}):
            args = _make_args(model="m")
            result = main_module._resolve_run_args(args)
        assert result.reasoning_effort == "low"

    def test_cli_reasoning_effort_overrides_settings(self):
        with patch("make_agent.main.load_settings", return_value={"model": "m", "reasoning_effort": "low"}):
            args = _make_args(model="m", reasoning_effort="high")
            result = main_module._resolve_run_args(args)
        assert result.reasoning_effort == "high"

    def test_invalid_reasoning_effort_in_settings_raises(self):
        with patch("make_agent.main.load_settings", return_value={"model": "m", "reasoning_effort": "extreme"}):
            args = _make_args(model="m")
            with pytest.raises(ValueError, match="Invalid reasoning_effort"):
                main_module._resolve_run_args(args)


# ── _resolve_system_prompt ────────────────────────────────────────────────────


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
        # Note: argparse enforces mutual exclusivity at parse time; here we just
        # verify priority in the function itself.
        result = main_module._resolve_system_prompt(args)
        assert result == "Inline prompt."
