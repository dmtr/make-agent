"""Tests for make_agent.app_dirs — project slug and path helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import make_agent.app_dirs as app_dirs


class TestProjectSlug:
    def test_strips_leading_slash(self):
        assert app_dirs.project_slug("/Users/alice/myapp") == "Users_alice_myapp"

    def test_replaces_all_slashes_with_underscores(self):
        assert app_dirs.project_slug("/a/b/c/d") == "a_b_c_d"

    def test_single_segment(self):
        assert app_dirs.project_slug("/myproject") == "myproject"

    def test_uses_cwd_when_no_argument(self, monkeypatch):
        monkeypatch.chdir("/")
        # os.getcwd() returns "/" — slug should be empty string (stripped)
        slug = app_dirs.project_slug()
        assert isinstance(slug, str)

    def test_cwd_derived_slug(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        slug = app_dirs.project_slug()
        expected = str(tmp_path).lstrip("/").replace("/", "_")
        assert slug == expected


class TestProjectDir:
    def test_returns_path_under_app_home(self, tmp_path):
        with patch.object(app_dirs, "_APP_HOME", tmp_path):
            result = app_dirs.project_dir("/usr/local/myapp")
        assert result == tmp_path / "usr_local_myapp"

    def test_creates_directory(self, tmp_path):
        with patch.object(app_dirs, "_APP_HOME", tmp_path):
            result = app_dirs.project_dir("/a/b")
        assert result.is_dir()

    def test_idempotent_when_dir_already_exists(self, tmp_path):
        with patch.object(app_dirs, "_APP_HOME", tmp_path):
            app_dirs.project_dir("/a/b")
            result = app_dirs.project_dir("/a/b")  # second call should not raise
        assert result.is_dir()


class TestDefaultAgentsDir:
    def test_returns_agents_subdirectory(self, tmp_path):
        with patch.object(app_dirs, "_APP_HOME", tmp_path):
            result = app_dirs.default_agents_dir("/proj/foo")
        assert result == str(tmp_path / "proj_foo" / "agents")

    def test_creates_agents_directory(self, tmp_path):
        with patch.object(app_dirs, "_APP_HOME", tmp_path):
            result = app_dirs.default_agents_dir("/proj/foo")
        assert Path(result).is_dir()

    def test_returns_string(self, tmp_path):
        with patch.object(app_dirs, "_APP_HOME", tmp_path):
            result = app_dirs.default_agents_dir("/proj/foo")
        assert isinstance(result, str)


class TestLogFile:
    def test_returns_log_file_path(self, tmp_path):
        with patch.object(app_dirs, "_APP_HOME", tmp_path):
            result = app_dirs.log_file("/proj/bar")
        assert result == str(tmp_path / "proj_bar" / "logs" / "make-agent.log")

    def test_creates_logs_directory(self, tmp_path):
        with patch.object(app_dirs, "_APP_HOME", tmp_path):
            result = app_dirs.log_file("/proj/bar")
        assert Path(result).parent.is_dir()

    def test_returns_string(self, tmp_path):
        with patch.object(app_dirs, "_APP_HOME", tmp_path):
            result = app_dirs.log_file("/proj/bar")
        assert isinstance(result, str)

    def test_log_filename_is_make_agent_log(self, tmp_path):
        with patch.object(app_dirs, "_APP_HOME", tmp_path):
            result = app_dirs.log_file("/some/path")
        assert Path(result).name == "make-agent.log"
