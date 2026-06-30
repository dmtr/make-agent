"""Skill management tools: list_skills, read_skill, execute_skill, create_skill, validate_skill."""

from __future__ import annotations

import asyncio
import os
import re
import shlex
from pathlib import Path
from typing import Any

from make_agent.parser import parse, parse_file

_VALID_SKILL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_VALID_MAKE_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FORBIDDEN_MAKE_OPTIONS = frozenset(
    {
        "-f",
        "--file",
        "--makefile",
        "-C",
        "--directory",
        "-I",
        "--include-dir",
        "--eval",
    }
)


def _is_valid_make_var_name(name: str) -> bool:
    return bool(_VALID_MAKE_VAR_NAME_RE.fullmatch(name))


def _valid_skill_name(name: str) -> bool:
    return bool(_VALID_SKILL_NAME_RE.fullmatch(name))


def _find_forbidden_make_option(tokens: list[str]) -> str | None:
    """Return the first blocked make option found in *tokens*, if any."""
    for token in tokens:
        if token in _FORBIDDEN_MAKE_OPTIONS:
            return token
        if token.startswith("--file=") or token.startswith("--makefile="):
            return token
        if token.startswith("--directory=") or token.startswith("--include-dir="):
            return token
        if token.startswith("--eval="):
            return token
        if token.startswith("-f") and token != "-f":
            return token
        if token.startswith("-C") and token != "-C":
            return token
        if token.startswith("-I") and token != "-I":
            return token
    return None


def _resolve_safe_skill_path(
    skills_dir: str,
    name: str,
    filename: str,
    *,
    create_dirs: bool = False,
) -> tuple[Path, Path] | str:
    """Return ``(skill_dir, file_path)`` while enforcing symlink and containment rules."""
    skills_root = Path(skills_dir)
    if skills_root.exists():
        if skills_root.is_symlink():
            return f"Error: refusing to use symlinked skills directory: {skills_root}"
        if not skills_root.is_dir():
            return f"Error: skills directory is not a directory: {skills_root}"
    if create_dirs:
        try:
            skills_root.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return f"Error: could not create skills directory: {e}"
        if skills_root.is_symlink():
            return f"Error: refusing to use symlinked skills directory: {skills_root}"

    skill_dir = skills_root / name
    if skill_dir.exists():
        if skill_dir.is_symlink():
            return f"Error: refusing to use symlinked skill directory: {skill_dir}"
        if not skill_dir.is_dir():
            return f"Error: skill path is not a directory: {skill_dir}"
    if create_dirs:
        try:
            skill_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return f"Error: could not create skill directory: {e}"
        if skill_dir.is_symlink():
            return f"Error: refusing to use symlinked skill directory: {skill_dir}"

    file_path = skill_dir / filename
    if file_path.exists() and file_path.is_symlink():
        return f"Error: refusing to use symlinked path: {file_path}"

    try:
        resolved_root = skills_root.resolve(strict=False)
        resolved_skill = skill_dir.resolve(strict=False)
    except OSError as e:
        return f"Error: could not resolve skill path: {e}"
    try:
        resolved_skill.relative_to(resolved_root)
    except ValueError:
        return f"Error: skill path escapes the skills directory: {skill_dir}"

    return skill_dir, file_path


def _skill_description(mk_path: Path) -> str:
    """Return the skill's description from the define DESCRIPTION block, or a fallback."""
    try:
        mf = parse_file(mk_path)
    except Exception:
        return "  (could not read)"
    if mf.description:
        first_line = mf.description.strip().splitlines()[0]
        return f"  {first_line}"
    return "  (no description)"


def list_skills(skills_dir: str, enabled_skills: frozenset[str] | None = None) -> str:
    """List all available skills with their names and descriptions.

    If *enabled_skills* is provided and is not ``{"*"}`` or empty, only those
    skill names are shown.  When the set contains ``"*"`` or is None, all
    discovered skills are listed (default: show everything).
    """
    path = Path(skills_dir)
    if not path.exists():
        return "No skills found (directory does not exist)"

    all_skill_dirs = sorted(
        p for p in path.iterdir() if p.is_dir() and (p / "skill.mk").exists()
    )
    if not all_skill_dirs:
        return "No skills found"

    if enabled_skills is not None:
        skill_dirs = [sd for sd in all_skill_dirs if sd.name in enabled_skills]
    else:
        skill_dirs = all_skill_dirs

    if not skill_dirs:
        return "No skills found"

    entries = []
    for sd in skill_dirs:
        desc = _skill_description(sd / "skill.mk")
        entries.append(f"{sd.name}:{desc}")
    return "\n\n".join(entries)


def read_skill(name: str, skills_dir: str) -> str:
    """Read a skill's full definition by returning the raw skill.mk content."""
    if not _valid_skill_name(name):
        return f"Error: invalid skill name {name!r}. Use letters, numbers, hyphens, underscores, and dots only."
    safe_paths = _resolve_safe_skill_path(skills_dir, name, "skill.mk")
    if isinstance(safe_paths, str):
        return safe_paths
    skill_dir, mk = safe_paths
    if not skill_dir.exists():
        return f"Skill '{name}' not found in {skills_dir}"
    if not mk.exists():
        return f"Skill '{name}' is missing skill.mk"
    try:
        return mk.read_text(encoding="utf-8")
    except OSError as e:
        return f"Error: could not read skill.mk: {e}"


async def execute_skill(
    name: str,
    command: str,
    skills_dir: str,
    timeout: int = 600,
) -> str:
    """Run a make command against a skill's skill.mk.

    *command* is a shell-style string such as ``make``, ``make target``, or
    ``VAR=val make target``.  Leading ``KEY=VAL`` tokens (before ``make``) are
    injected as environment variables; tokens after ``make`` are passed as make
    arguments (targets and/or make-style variable assignments).
    """

    if not _valid_skill_name(name):
        return f"Error: invalid skill name {name!r}. Use letters, numbers, hyphens, underscores, and dots only."
    safe_paths = _resolve_safe_skill_path(skills_dir, name, "skill.mk")
    if isinstance(safe_paths, str):
        return safe_paths
    skill_dir, mk = safe_paths
    if not skill_dir.exists() or not mk.exists():
        return f"Skill '{name}' not found in {skills_dir}"

    try:
        tokens = shlex.split(command)
    except ValueError as e:
        return f"Error: could not parse command {command!r}: {e}"
    if not tokens:
        return "Error: command is empty"

    env_vars: dict[str, str] = {}
    idx = 0
    while (
        idx < len(tokens) and "=" in tokens[idx] and not tokens[idx].startswith("make")
    ):
        token = tokens[idx]
        k, _, v = token.partition("=")
        if not _is_valid_make_var_name(k):
            return f"Error: {k!r} is not a valid make variable name"
        if k in os.environ:
            return (
                f"Error: parameter {k!r} shadows the system environment variable {k!r}"
            )
        env_vars[k] = v
        idx += 1

    if idx < len(tokens) and tokens[idx] == "make":
        idx += 1

    make_args = tokens[idx:]
    if forbidden := _find_forbidden_make_option(make_args):
        return (
            f"Error: make option {forbidden!r} is not allowed in execute_skill. "
            "Blocked options: -f/--file/--makefile, -C/--directory, "
            "-I/--include-dir, --eval."
        )

    env = {**os.environ, **env_vars}
    cmd = ["make", "--no-print-directory", "-f", str(mk), *make_args]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
        )
    except OSError as e:
        return f"Error: failed to run make: {e}"

    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return f"Error: execute_skill '{name}' exceeded {timeout}s timeout"
    except asyncio.CancelledError:
        proc.kill()
        await proc.wait()
        raise

    stdout = stdout_b.decode(errors="replace")
    stderr = stderr_b.decode(errors="replace")
    from make_agent.tool_handler.runner import get_tool_result  # avoids circular import

    return get_tool_result(stdout, stderr, proc.returncode).output


def _write_no_symlink(path: Path, content: str) -> None:
    """Write *content* to *path*, refusing symlink destinations."""
    if path.is_symlink():
        raise ValueError(f"refusing to overwrite symlink: {path}")
    path.write_text(content, encoding="utf-8")


def create_skill(
    name: str,
    mk_content: str,
    skills_dir: str,
) -> str:
    """Create or overwrite a skill directory with a single skill.mk file.

    *mk_content* must contain a ``define DESCRIPTION … endef`` block.
    """
    if not _valid_skill_name(name):
        return f"Error: invalid skill name {name!r}. Use letters, numbers, hyphens, underscores, and dots only."

    try:
        parsed_mk = parse(mk_content)
    except Exception as e:
        return f"Error: could not parse skill.mk: {e}"

    if not parsed_mk.description:
        return "Error: skill.mk must contain a 'define DESCRIPTION … endef' block"

    safe_paths = _resolve_safe_skill_path(
        skills_dir, name, "skill.mk", create_dirs=True
    )
    if isinstance(safe_paths, str):
        return safe_paths
    skill_dir, mk = safe_paths

    try:
        _write_no_symlink(mk, mk_content)
    except (OSError, ValueError) as e:
        return f"Error: could not write skill.mk: {e}"

    return f"Created skill '{name}' at {skill_dir}"


def validate_skill(name: str, skills_dir: str) -> str:
    """Validate a skill: checks skill.mk exists, parses cleanly, and has a DESCRIPTION block."""
    if not _valid_skill_name(name):
        return f"Error: invalid skill name {name!r}. Use letters, numbers, hyphens, underscores, and dots only."
    safe_paths = _resolve_safe_skill_path(skills_dir, name, "skill.mk")
    if isinstance(safe_paths, str):
        return safe_paths
    skill_dir, mk = safe_paths
    if not skill_dir.exists():
        return f"Skill '{name}' not found in {skills_dir}"
    if not mk.exists():
        return f"Skill '{name}' is missing skill.mk"
    try:
        mf = parse_file(mk)
    except OSError as e:
        return f"Error: could not read {mk}: {e}"
    if not mf.description:
        return "Validation error: skill.mk must contain a 'define DESCRIPTION … endef' block"
    return f"OK — {skill_dir}"


SKILL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_skills",
            "description": "List all available skills with their names and descriptions.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_skill",
            "description": "Read a skill's full definition (raw skill.mk content).",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The skill name (directory name).",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_skill",
            "description": (
                "Run a make command against a skill's skill.mk. "
                "Call read_skill first to learn what targets and variables are available. "
                "Pass a shell-style command such as 'make', 'make target', or 'VAR=val make target'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The skill name (directory name).",
                    },
                    "command": {
                        "type": "string",
                        "description": (
                            "The make command to run, e.g. 'make', 'make target', or 'VAR=val make target'. "
                            "Leading KEY=VAL tokens are passed as environment variables."
                        ),
                    },
                },
                "required": ["name", "command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_skill",
            "description": (
                "Create a new skill or overwrite an existing one. "
                "The skill.mk must contain a 'define DESCRIPTION … endef' block. "
                "It must NOT contain a define SYSTEM_PROMPT block."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name (letters, numbers, hyphens, underscores, dots).",
                    },
                    "mk_content": {
                        "type": "string",
                        "description": (
                            "Full content of skill.mk. Must include a 'define DESCRIPTION … endef' block. "
                            "Do NOT include a define SYSTEM_PROMPT block."
                        ),
                    },
                },
                "required": ["name", "mk_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_skill",
            "description": "Validate a skill: checks skill.mk exists, parses cleanly, and has a DESCRIPTION block.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The skill name (directory name).",
                    },
                },
                "required": ["name"],
            },
        },
    },
]
