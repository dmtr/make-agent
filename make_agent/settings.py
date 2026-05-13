"""Per-project settings stored in ``~/.make-agent/<project>/settings.yaml``.

Supported fields::

    model: anthropic/claude-haiku-4-5-20251001

Fields present in the file act as defaults; CLI flags always take precedence.
Missing fields are simply ignored — settings are always partial.
"""

from __future__ import annotations

from typing import Any

import yaml

from make_agent.app_dirs import settings_file


def load_settings(cwd: str | None = None) -> dict[str, Any] | None:
    """Load settings from ``~/.make-agent/<project>/settings.yaml``.

    Returns the parsed dict, or ``None`` if the file does not exist.
    Unknown keys are silently ignored.
    """
    path = settings_file(cwd)
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid settings file {path}: expected a YAML mapping, "
            f"got {type(data).__name__}. Please check your settings.yaml."
        )
    return {k: v for k, v in data.items() if k in ("model", "memory", "reasoning_effort")}


def save_settings(data: dict[str, Any], cwd: str | None = None) -> None:
    """Write *data* to ``~/.make-agent/<project>/settings.yaml``."""
    path = settings_file(cwd)
    path.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True), encoding="utf-8")
