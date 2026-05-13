"""Skill management tools: list_skills, read_skill, execute_skill, create_skill, validate_skill."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, NamedTuple

import yaml

from make_agent.parser import parse, parse_file, validate
from make_agent.tools import build_tools

_VALID_SKILL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _valid_skill_name(name: str) -> bool:
    return bool(_VALID_SKILL_NAME_RE.fullmatch(name))


def _skill_description(md_path: Path) -> str:
    """Return the skill's one-line description from skill.md frontmatter, or a fallback."""
    try:
        content = md_path.read_text(encoding="utf-8")
    except OSError:
        return "  (could not read)"
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            try:
                fm = yaml.safe_load(content[3:end].strip())
                if isinstance(fm, dict) and "description" in fm:
                    return f"  {fm['description']}"
            except Exception:
                pass
    return "  (no description)"


class _ExecuteSkill(NamedTuple):
    """Sentinel returned by execute_skill to trigger dynamic tool injection."""

    skill_md: str
    tool_schemas: list[dict]
    mk_path: Path | None
    tool_summary: str


def list_skills(skills_dir: str) -> str:
    """List all available skills with their names and descriptions."""
    path = Path(skills_dir)
    if not path.exists():
        return "No skills found (directory does not exist)"
    skill_dirs = sorted(p for p in path.iterdir() if p.is_dir() and (p / "skill.md").exists())
    if not skill_dirs:
        return "No skills found"
    entries = []
    for sd in skill_dirs:
        desc = _skill_description(sd / "skill.md")
        has_mk = (sd / "skill.mk").exists()
        entry = f"{sd.name}:{desc}"
        if has_mk:
            entry += "  [has tools]"
        entries.append(entry)
    return "\n\n".join(entries)


def read_skill(name: str, skills_dir: str) -> str:
    """Read a skill's full definition (skill.md and skill.mk if present)."""
    if not _valid_skill_name(name):
        return f"Error: invalid skill name {name!r}. Use letters, numbers, hyphens, underscores, and dots only."
    skill_dir = Path(skills_dir) / name
    md_path = skill_dir / "skill.md"
    if not skill_dir.exists():
        return f"Skill '{name}' not found in {skills_dir}"
    if not md_path.exists():
        return f"Skill '{name}' is missing skill.md"
    try:
        md_content = md_path.read_text(encoding="utf-8")
    except OSError as e:
        return f"Error: could not read skill.md: {e}"
    mk_path = skill_dir / "skill.mk"
    if not mk_path.exists():
        return f"skill.md:\n{md_content}\n\n(no skill.mk)"
    try:
        mk_content = mk_path.read_text(encoding="utf-8")
    except OSError as e:
        return f"skill.md:\n{md_content}\n\nError reading skill.mk: {e}"
    return f"skill.md:\n{md_content}\n\nskill.mk:\n{mk_content}"


def execute_skill(name: str, skills_dir: str) -> _ExecuteSkill | str:
    """Load a skill and return its instructions; injects skill.mk tools into the active tool set."""
    if not _valid_skill_name(name):
        return f"Error: invalid skill name {name!r}. Use letters, numbers, hyphens, underscores, and dots only."
    skill_dir = Path(skills_dir) / name
    md_path = skill_dir / "skill.md"
    if not skill_dir.exists() or not md_path.exists():
        return f"Skill '{name}' not found in {skills_dir}"
    try:
        skill_md = md_path.read_text(encoding="utf-8")
    except OSError as e:
        return f"Error: could not read skill.md: {e}"
    mk_path = skill_dir / "skill.mk"
    if not mk_path.exists():
        return _ExecuteSkill(
            skill_md=skill_md,
            tool_schemas=[],
            mk_path=None,
            tool_summary="No additional tools available for this skill.",
        )
    try:
        mf = parse_file(mk_path)
    except Exception as e:
        return f"Error: could not parse skill.mk for '{name}': {e}"
    tool_schemas = build_tools(mf)
    if tool_schemas:
        tool_names = [s["function"]["name"] for s in tool_schemas]
        tool_summary = f"Newly available tools: {', '.join(tool_names)}"
    else:
        tool_summary = "skill.mk has no annotated tools."
    return _ExecuteSkill(
        skill_md=skill_md,
        tool_schemas=tool_schemas,
        mk_path=mk_path,
        tool_summary=tool_summary,
    )


def _write_no_symlink(path: Path, content: str) -> None:
    """Write *content* to *path*, refusing symlink destinations."""
    if path.is_symlink():
        raise ValueError(f"refusing to overwrite symlink: {path}")
    path.write_text(content, encoding="utf-8")


def create_skill(
    name: str,
    description: str,
    md_content: str,
    skills_dir: str,
    mk_content: str | None = None,
) -> str:
    """Create or overwrite a skill directory with skill.md and optional skill.mk."""
    if not _valid_skill_name(name):
        return f"Error: invalid skill name {name!r}. Use letters, numbers, hyphens, underscores, and dots only."

    parsed_mk = None
    if mk_content:
        try:
            parsed_mk = parse(mk_content)
        except Exception as e:
            return f"Error: could not parse skill.mk: {e}"
        errors = validate(parsed_mk)
        if errors:
            return "Validation errors in skill.mk:\n" + "\n".join(f"  - {e}" for e in errors)

    skill_dir = Path(skills_dir) / name
    md_path = skill_dir / "skill.md"
    mk_path = skill_dir / "skill.mk"

    if md_path.is_symlink():
        return f"Error: refusing to overwrite symlink: {md_path}"
    if mk_content and mk_path.is_symlink():
        return f"Error: refusing to overwrite symlink: {mk_path}"

    skill_dir.mkdir(parents=True, exist_ok=True)

    if not md_content.strip().startswith("---"):
        md_with_fm = f'---\ndescription: "{description}"\n---\n\n{md_content}'
    else:
        md_with_fm = md_content

    try:
        _write_no_symlink(md_path, md_with_fm)
    except (OSError, ValueError) as e:
        return f"Error: could not write skill.md: {e}"

    if mk_content and parsed_mk is not None:
        try:
            _write_no_symlink(mk_path, mk_content)
        except (OSError, ValueError) as e:
            return f"Error: could not write skill.mk: {e}"
        tool_count = sum(1 for r in parsed_mk.rules if r.description is not None)
        return f"Created skill '{name}' at {skill_dir} ({tool_count} tool(s))"

    return f"Created skill '{name}' at {skill_dir} (no tools)"


def validate_skill(name: str, skills_dir: str) -> str:
    """Validate a skill: checks skill.md exists and validates skill.mk if present."""
    if not _valid_skill_name(name):
        return f"Error: invalid skill name {name!r}. Use letters, numbers, hyphens, underscores, and dots only."
    skill_dir = Path(skills_dir) / name
    md_path = skill_dir / "skill.md"
    if not skill_dir.exists():
        return f"Skill '{name}' not found in {skills_dir}"
    if not md_path.exists():
        return f"Skill '{name}' is missing skill.md"
    mk_path = skill_dir / "skill.mk"
    if not mk_path.exists():
        return f"OK — {skill_dir} (skill.md only, no tools)"
    try:
        mf = parse_file(mk_path)
    except OSError as e:
        return f"Error: could not read {mk_path}: {e}"
    errors = validate(mf)
    tool_count = sum(1 for r in mf.rules if r.params or r.description)
    if not tool_count:
        errors = ["No tools defined: at least one rule must have a # <tool> annotation block."] + errors
    if errors:
        return "Validation errors:\n" + "\n".join(f"  - {e}" for e in errors)
    return f"OK — {skill_dir} ({tool_count} tool(s) valid)"


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
            "description": "Read a skill's full definition (skill.md and skill.mk if present).",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The skill name (directory name)."},
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
                "Execute a skill: returns skill.md instructions and injects skill.mk tools "
                "into the active tool set for the rest of the conversation. "
                "Read the returned instructions carefully and follow them."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The skill name (directory name)."},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_skill",
            "description": (
                "Create a new skill or overwrite an existing one. "
                "A skill consists of skill.md (instructions) and optionally skill.mk (tools). "
                "skill.mk must NOT contain a define SYSTEM_PROMPT block — tools only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name (letters, numbers, hyphens, underscores, dots).",
                    },
                    "description": {
                        "type": "string",
                        "description": "One-line description shown in list_skills.",
                    },
                    "md_content": {
                        "type": "string",
                        "description": "Full content of skill.md (instructions for the agent).",
                    },
                    "mk_content": {
                        "type": "string",
                        "description": (
                            "Optional: content of skill.mk with # <tool> annotated targets. "
                            "Do NOT include a define SYSTEM_PROMPT block."
                        ),
                    },
                },
                "required": ["name", "description", "md_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_skill",
            "description": "Validate a skill: checks skill.md exists and validates skill.mk syntax if present.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The skill name (directory name)."},
                },
                "required": ["name"],
            },
        },
    },
]
