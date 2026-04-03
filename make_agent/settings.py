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

from make_agent.app_dirs import default_agents_dir, settings_file

_DEFAULT_MODEL = "anthropic/claude-haiku-4-5-20251001"
_DEFAULT_MAKEFILE = "Makefile"
_ORCHESTRA_TEMPLATE = Path(__file__).parent / "templates" / "orchestra.mk"


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
    return {k: v for k, v in data.items() if k in ("model", "makefile", "memory")}


def save_settings(data: dict[str, Any], cwd: str | None = None) -> None:
    """Write *data* to ``~/.make-agent/<project>/settings.yaml``."""
    path = settings_file(cwd)
    path.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True), encoding="utf-8")


def run_setup_wizard() -> dict[str, Any]:
    """Interactively ask the user for project settings, save and return them.

    If the agents directory is empty, copies the bundled ``orchestra.mk``
    template there and uses it as the makefile (no prompt for the path).
    Otherwise prompts for a Makefile path.

    Always prompts for the model string.
    """
    print("\nNo settings.yaml found for this project.")
    print("Let's create one. Press Enter to accept the default shown in brackets.\n")

    agents_dir = Path(default_agents_dir())
    existing_agents = list(agents_dir.glob("*.mk"))

    if not existing_agents:
        dest = agents_dir / "orchestra.mk"
        dest.write_bytes(_ORCHESTRA_TEMPLATE.read_bytes())
        makefile = str(dest)
        print(f"  Created {dest}")
    else:
        print("  Available agents:")
        for i, agent in enumerate(existing_agents, 1):
            print(f"    {i}) {agent.name}")
        while True:
            raw = input(f"  Choose an agent [1-{len(existing_agents)}]: ").strip()
            if raw.isdigit() and 1 <= int(raw) <= len(existing_agents):
                makefile = str(existing_agents[int(raw) - 1])
                break
            print(f"  Please enter a number between 1 and {len(existing_agents)}.")

    raw_model = input(f"  Model [{_DEFAULT_MODEL}]: ").strip()
    model = raw_model or _DEFAULT_MODEL

    settings: dict[str, Any] = {"makefile": makefile, "model": model}
    save_settings(settings)

    path = settings_file()
    print(f"\nSaved settings to {path}\n")
    return settings
