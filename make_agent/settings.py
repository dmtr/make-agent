"""Per-project settings stored in ``~/.make-agent/<project>/settings.yaml``.

Supported fields::

    model: anthropic/claude-haiku-4-5-20251001
    makefile: ./Makefile

Fields present in the file act as defaults; CLI flags always take precedence.
Missing fields are simply ignored — settings are always partial.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from make_agent.app_dirs import settings_file

_DEFAULT_MODEL = "anthropic/claude-haiku-4-5-20251001"
_DEFAULT_MAKEFILE = "Makefile"


def load_settings(cwd: str | None = None) -> dict[str, Any] | None:
    """Load settings from ``~/.make-agent/<project>/settings.yaml``.

    Returns the parsed dict, or ``None`` if the file does not exist.
    Unknown keys are silently ignored.
    """
    path = settings_file(cwd)
    if not path.exists():
        return None
    text = path.read_text()
    data = yaml.safe_load(text) or {}
    return {k: v for k, v in data.items() if k in ("model", "makefile", "memory")}


def save_settings(data: dict[str, Any], cwd: str | None = None) -> None:
    """Write *data* to ``~/.make-agent/<project>/settings.yaml``."""
    path = settings_file(cwd)
    path.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True))


def run_setup_wizard() -> dict[str, Any]:
    """Interactively ask the user for project settings, save and return them.

    Prompts for:
    - Makefile path (default: ``Makefile``)
    - Model string  (default: ``anthropic/claude-haiku-4-5-20251001``)

    The collected values are saved to ``settings.yaml`` and returned.
    """
    print("\nNo settings.yaml found for this project.")
    print("Let's create one. Press Enter to accept the default shown in brackets.\n")

    raw_makefile = input(f"  Makefile path [{_DEFAULT_MAKEFILE}]: ").strip()
    makefile = raw_makefile or _DEFAULT_MAKEFILE

    raw_model = input(f"  Model [{_DEFAULT_MODEL}]: ").strip()
    model = raw_model or _DEFAULT_MODEL

    settings: dict[str, Any] = {"makefile": makefile, "model": model}
    save_settings(settings)

    path = settings_file()
    print(f"\nSaved settings to {path}\n")
    return settings
