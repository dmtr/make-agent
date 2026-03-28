"""Tests for make_agent.settings — load, save, and main.py integration."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

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

    def test_returns_model_and_makefile(self, tmp_path):
        (tmp_path / "settings.yaml").write_text("model: mymodel\nmakefile: MyMakefile\n")
        with _patch_settings_file(tmp_path):
            result = load_settings()
        assert result == {"model": "mymodel", "makefile": "MyMakefile"}

    def test_partial_settings_only_model(self, tmp_path):
        (tmp_path / "settings.yaml").write_text("model: mymodel\n")
        with _patch_settings_file(tmp_path):
            result = load_settings()
        assert result == {"model": "mymodel"}

    def test_partial_settings_only_makefile(self, tmp_path):
        (tmp_path / "settings.yaml").write_text("makefile: special.mk\n")
        with _patch_settings_file(tmp_path):
            result = load_settings()
        assert result == {"makefile": "special.mk"}

    def test_unknown_keys_are_ignored(self, tmp_path):
        (tmp_path / "settings.yaml").write_text("model: m\nmakefile: f\nunknown: x\n")
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
            save_settings({"model": "m", "makefile": "Makefile"})
        data = yaml.safe_load((tmp_path / "settings.yaml").read_text())
        assert data == {"model": "m", "makefile": "Makefile"}

    def test_overwrites_existing_file(self, tmp_path):
        (tmp_path / "settings.yaml").write_text("model: old\n")
        with _patch_settings_file(tmp_path):
            save_settings({"model": "new", "makefile": "M"})
        data = yaml.safe_load((tmp_path / "settings.yaml").read_text())
        assert data["model"] == "new"


# ── _resolve_run_args (main.py integration) ───────────────────────────────────


def _make_args(**kwargs) -> argparse.Namespace:
    defaults = dict(
        file=None,
        model=None,
        prompt=None,
        prompt_file=None,
        debug=False,
        max_retries=5,
        tool_timeout=600,
        max_tool_output=20000,
        agents_dir=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestResolveRunArgs:
    def test_cli_file_overrides_settings(self, tmp_path):
        with patch("make_agent.main.load_settings", return_value={"makefile": "settings.mk", "model": "sm"}):
            args = _make_args(file="cli.mk", model=None)
            result = main_module._resolve_run_args(args)
        assert result.file == "cli.mk"

    def test_cli_model_overrides_settings(self, tmp_path):
        with patch("make_agent.main.load_settings", return_value={"makefile": "s.mk", "model": "settings-model"}):
            args = _make_args(file="f.mk", model="cli-model")
            result = main_module._resolve_run_args(args)
        assert result.model == "cli-model"

    def test_settings_model_used_when_no_cli_model(self):
        with patch("make_agent.main.load_settings", return_value={"makefile": "s.mk", "model": "settings-model"}):
            args = _make_args(file="f.mk", model=None)
            result = main_module._resolve_run_args(args)
        assert result.model == "settings-model"

    def test_settings_makefile_used_when_no_cli_file(self):
        with patch("make_agent.main.load_settings", return_value={"makefile": "settings.mk", "model": "m"}), \
             patch("make_agent.main._find_makefile", return_value="settings.mk"):
            args = _make_args(file=None, model="m")
            result = main_module._resolve_run_args(args)
        assert result.file == "settings.mk"

    def test_settings_makefile_found_in_agents_dir(self):
        with patch("make_agent.main.load_settings", return_value={"makefile": "myfile.mk"}), \
             patch("make_agent.main._find_makefile", return_value="/home/.make-agent/proj/agents/myfile.mk"):
            args = _make_args(file=None, model="m")
            result = main_module._resolve_run_args(args)
        assert result.file == "/home/.make-agent/proj/agents/myfile.mk"

    def test_settings_makefile_kept_as_is_when_not_found(self):
        with patch("make_agent.main.load_settings", return_value={"makefile": "missing.mk"}), \
             patch("make_agent.main._find_makefile", return_value=None):
            args = _make_args(file=None, model="m")
            result = main_module._resolve_run_args(args)
        assert result.file == "missing.mk"

    def test_code_default_model_when_no_settings_and_no_cli(self):
        with patch("make_agent.main.load_settings", return_value={}):
            args = _make_args(file="f.mk", model=None)
            result = main_module._resolve_run_args(args)
        assert result.model == main_module._DEFAULT_MODEL

    def test_code_default_makefile_when_no_settings_and_no_cli(self):
        with patch("make_agent.main.load_settings", return_value={}), \
             patch("make_agent.main._find_makefile", return_value=None):
            args = _make_args(file=None, model="m")
            result = main_module._resolve_run_args(args)
        assert result.file == main_module._DEFAULT_MAKEFILE

    def test_wizard_triggered_when_no_file_no_settings_and_no_makefile_found(self):
        wizard_result = {"makefile": "wizard.mk", "model": "wizard-model"}
        with patch("make_agent.main.load_settings", return_value=None), \
             patch("make_agent.main._find_makefile", return_value=None), \
             patch("make_agent.main.run_setup_wizard", return_value=wizard_result) as mock_wizard:
            args = _make_args(file=None, model=None)
            result = main_module._resolve_run_args(args)
        mock_wizard.assert_called_once()
        assert result.file == "wizard.mk"
        assert result.model == "wizard-model"

    def test_wizard_not_triggered_when_makefile_found_in_search_path(self):
        with patch("make_agent.main.load_settings", return_value=None), \
             patch("make_agent.main._find_makefile", return_value="found.mk"), \
             patch("make_agent.main.run_setup_wizard") as mock_wizard:
            args = _make_args(file=None, model=None)
            result = main_module._resolve_run_args(args)
        mock_wizard.assert_not_called()
        assert result.file == "found.mk"

    def test_found_makefile_used_when_no_settings(self):
        with patch("make_agent.main.load_settings", return_value=None), \
             patch("make_agent.main._find_makefile", return_value="/home/proj/agents/Makefile"):
            args = _make_args(file=None, model="m")
            result = main_module._resolve_run_args(args)
        assert result.file == "/home/proj/agents/Makefile"

    def test_wizard_triggered_when_no_file_and_no_settings(self):
        wizard_result = {"makefile": "wizard.mk", "model": "wizard-model"}
        with patch("make_agent.main.load_settings", return_value=None), \
             patch("make_agent.main._find_makefile", return_value=None), \
             patch("make_agent.main.run_setup_wizard", return_value=wizard_result) as mock_wizard:
            args = _make_args(file=None, model=None)
            result = main_module._resolve_run_args(args)
        mock_wizard.assert_called_once()
        assert result.file == "wizard.mk"
        assert result.model == "wizard-model"

    def test_wizard_not_triggered_when_file_provided(self):
        with patch("make_agent.main.load_settings", return_value=None), \
             patch("make_agent.main.run_setup_wizard") as mock_wizard:
            args = _make_args(file="explicit.mk", model=None)
            main_module._resolve_run_args(args)
        mock_wizard.assert_not_called()

    def test_wizard_not_triggered_when_settings_exist(self):
        with patch("make_agent.main.load_settings", return_value={"makefile": "s.mk", "model": "m"}), \
             patch("make_agent.main.run_setup_wizard") as mock_wizard:
            args = _make_args(file=None, model=None)
            main_module._resolve_run_args(args)
        mock_wizard.assert_not_called()


# ── _find_makefile ────────────────────────────────────────────────────────────


class TestFindMakefile:
    def test_returns_cwd_makefile_when_it_exists(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "Makefile").write_text("all:\n\t@echo ok\n")
        with patch("make_agent.main.default_agents_dir", return_value=str(tmp_path / "agents")):
            result = main_module._find_makefile()
        assert result == "Makefile"

    def test_returns_agents_makefile_when_only_there(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        agents = tmp_path / "agents"
        agents.mkdir()
        (agents / "Makefile").write_text("all:\n\t@echo ok\n")
        with patch("make_agent.main.default_agents_dir", return_value=str(agents)):
            result = main_module._find_makefile()
        assert result == str(agents / "Makefile")

    def test_returns_none_when_not_found_anywhere(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch("make_agent.main.default_agents_dir", return_value=str(tmp_path / "agents")):
            result = main_module._find_makefile()
        assert result is None

    def test_cwd_takes_priority_over_agents_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "Makefile").write_text("cwd:\n\t@echo cwd\n")
        agents = tmp_path / "agents"
        agents.mkdir()
        (agents / "Makefile").write_text("agents:\n\t@echo agents\n")
        with patch("make_agent.main.default_agents_dir", return_value=str(agents)):
            result = main_module._find_makefile()
        assert result == "Makefile"

    def test_custom_name_found_in_agents_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        agents = tmp_path / "agents"
        agents.mkdir()
        (agents / "myfile.mk").write_text("all:\n\t@echo ok\n")
        with patch("make_agent.main.default_agents_dir", return_value=str(agents)):
            result = main_module._find_makefile("myfile.mk")
        assert result == str(agents / "myfile.mk")

    def test_custom_name_found_in_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "myfile.mk").write_text("all:\n\t@echo ok\n")
        with patch("make_agent.main.default_agents_dir", return_value=str(tmp_path / "agents")):
            result = main_module._find_makefile("myfile.mk")
        assert result == "myfile.mk"
