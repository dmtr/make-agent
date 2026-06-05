"""Centralised path helpers for ~/.make-agent project directories.

All app-related files live under a hidden directory in the user's home folder::

    ~/.make-agent/<project-slug>/agents/            # default agents directory
    ~/.make-agent/<project-slug>/logs/              # log files
    ~/.make-agent/<project-slug>/makefile/          # makefile-mode data
    ~/.make-agent/<project-slug>/python/            # python-mode data

Each mode directory contains its own ``skills/``, ``SYSTEM.md``, and ``memory.db``.
The *project slug* is derived from the absolute working directory by stripping
the leading ``/`` and replacing every remaining ``/`` with ``_``.

Example:  ``/Users/alice/proj/myapp``  →  ``Users_alice_proj_myapp``
"""

from __future__ import annotations

import os
from pathlib import Path

_TEMPLATES_DIR = Path(__file__).parent / "templates"

_APP_HOME = Path.home() / ".make-agent"


def project_slug(cwd: str | None = None) -> str:
    """Return the project slug for *cwd* (defaults to ``os.getcwd()``)."""
    path = cwd or os.getcwd()
    return path.lstrip("/").replace("/", "_")


def project_dir(cwd: str | None = None) -> Path:
    """Return ``~/.make-agent/<slug>/``, creating it if necessary."""
    directory = _APP_HOME / project_slug(cwd)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def mode_dir(mode: str, cwd: str | None = None) -> Path:
    """Return ``~/.make-agent/<slug>/<mode>/``, creating it if necessary."""
    directory = project_dir(cwd) / mode
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def default_agents_dir(cwd: str | None = None) -> str:
    """Return ``~/.make-agent/<slug>/agents/`` as a string, creating it if necessary."""
    directory = project_dir(cwd) / "agents"
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory)


def default_skills_dir(mode: str, cwd: str | None = None) -> Path:
    """Return ``~/.make-agent/<slug>/<mode>/skills/``, creating it if necessary."""
    directory = mode_dir(mode, cwd) / "skills"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def mode_memory_path(mode: str, cwd: str | None = None) -> Path:
    """Return ``~/.make-agent/<slug>/<mode>/memory.db``."""
    return mode_dir(mode, cwd) / "memory.db"


def ensure_mode_system_prompt(mode: str, cwd: str | None = None) -> None:
    """Copy the bundled SYSTEM.md template into the mode dir if it does not exist yet."""
    dest = mode_dir(mode, cwd) / "SYSTEM.md"
    if dest.exists():
        return
    src = _TEMPLATES_DIR / mode / "SYSTEM.md"
    if src.exists():
        dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def log_file(cwd: str | None = None) -> str:
    """Return ``~/.make-agent/<slug>/logs/make-agent.log`` as a string, creating the logs dir if necessary."""
    logs_dir = project_dir(cwd) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return str(logs_dir / "make-agent.log")
