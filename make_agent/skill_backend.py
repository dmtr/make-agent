"""SkillBackend protocol and two implementations: MakefileSkillBackend and PythonSkillBackend."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any, Protocol

import yaml

from make_agent.builtin_tools.file_tools import FILE_SCHEMAS, edit_file, write_file
from make_agent.builtin_tools.skill_tools import SKILL_SCHEMAS as MAKEFILE_SKILL_SCHEMAS
from make_agent.builtin_tools.skill_tools import _resolve_safe_skill_path
from make_agent.builtin_tools.skill_tools import _valid_skill_name
from make_agent.builtin_tools.skill_tools import create_skill as create_makefile_skill
from make_agent.builtin_tools.skill_tools import execute_skill as execute_makefile_skill
from make_agent.builtin_tools.skill_tools import list_skills as list_makefile_skills
from make_agent.builtin_tools.skill_tools import read_skill as read_makefile_skill
from make_agent.builtin_tools.skill_tools import (
    validate_skill as validate_makefile_skill,
)
from make_agent.skill_registry import SkillRegistry

logger = logging.getLogger(__name__)
_PYTHON_SKILL_RESULT_PREFIX = "__MAKE_AGENT_PY_SKILL_RESULT__"
_PYTHON_SKILL_WORKER_CODE = f"""\
import importlib.util
import json
import traceback
import sys

PREFIX = {_PYTHON_SKILL_RESULT_PREFIX!r}
path = sys.argv[1]
target = sys.argv[2]
raw_kwargs = sys.stdin.read()
if not raw_kwargs:
    raw_kwargs = "{{}}"

try:
    kwargs = json.loads(raw_kwargs)
    module_name = "_make_agent_exec_skill"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot create module spec for {{path}}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    fn = None
    targets = getattr(module, "_TARGETS", {{}})
    tool_meta = targets.get(target) if isinstance(targets, dict) else None
    if tool_meta is not None and hasattr(tool_meta, "fn"):
        fn = tool_meta.fn
    elif hasattr(module, target):
        fn = getattr(module, target)
    if fn is None or not callable(fn):
        raise RuntimeError(f"Target '{{target}}' not found in skill")
    result = fn(**kwargs)
    payload = {{"ok": True, "result": None if result is None else str(result)}}
except Exception as e:
    traceback.print_exc()
    payload = {{"ok": False, "error": str(e)}}

print(PREFIX + json.dumps(payload), flush=True)
"""


class SkillBackend(Protocol):
    @property
    def schemas(self) -> list[dict]: ...

    @property
    def executors(self) -> dict[str, Any]: ...

    async def setup(self, model: str) -> None: ...

    def get_skill_trusted(self, name: str) -> bool | None: ...


def _terminate_subprocess_tree(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    if hasattr(os, "killpg"):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            return
        except ProcessLookupError:
            return
    proc.kill()


def _extract_worker_payload(raw: str) -> dict[str, Any] | None:
    for line in reversed(raw.splitlines()):
        if line.startswith(_PYTHON_SKILL_RESULT_PREFIX):
            payload = line[len(_PYTHON_SKILL_RESULT_PREFIX) :]
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                return None
            if isinstance(parsed, dict):
                return parsed
            return None
    return None


async def _run_python_skill_in_subprocess(
    py_path: Path,
    target: str,
    kwargs: dict[str, Any],
    timeout: int,
) -> str:
    try:
        kwargs_json = json.dumps(kwargs)
    except TypeError as e:
        return f"Error: kwargs for execute_skill are not JSON-serializable: {e}"

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            _PYTHON_SKILL_WORKER_CODE,
            str(py_path),
            target,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as e:
        return f"Error: failed to run python skill subprocess: {e}"

    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(input=kwargs_json.encode()),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        _terminate_subprocess_tree(proc)
        await proc.wait()
        return f"Error: execute_skill '{py_path.parent.name}/{target}' exceeded {timeout}s timeout"
    except asyncio.CancelledError:
        _terminate_subprocess_tree(proc)
        await proc.wait()
        raise

    stdout = stdout_b.decode(errors="replace")
    stderr = stderr_b.decode(errors="replace")
    payload = _extract_worker_payload(stdout) or _extract_worker_payload(stderr)
    if payload is None:
        logger.error(
            "execute_skill subprocess returned malformed output for %s/%s (stdout=%r stderr=%r)",
            py_path.parent.name,
            target,
            stdout,
            stderr,
        )
        return "Error: execute_skill returned malformed subprocess output"

    if payload.get("ok") is True:
        result = payload.get("result")
        return (
            str(result)
            if result is not None
            else "OK. Execution succeeded with no output."
        )
    return f"Error: {payload.get('error', 'unknown error')}"


class MakefileSkillBackend:
    def __init__(
        self,
        skills_dir: str,
        tool_timeout: int = 600,
        base_dir: Path | None = None,
    ) -> None:
        self._skills_dir = skills_dir
        self._tool_timeout = tool_timeout
        self._base_dir = base_dir if base_dir is not None else Path.cwd()
        self._schemas = MAKEFILE_SKILL_SCHEMAS + FILE_SCHEMAS
        self._executors: dict[str, Any] = {
            "list_skills": lambda **_kw: list_makefile_skills(self._skills_dir),
            "read_skill": lambda name, **_kw: read_makefile_skill(
                name, self._skills_dir
            ),
            "execute_skill": lambda name, command, **_kw: execute_makefile_skill(
                name,
                command,
                self._skills_dir,
                self._tool_timeout,
            ),
            "create_skill": lambda name, mk_content, **_kw: create_makefile_skill(
                name,
                mk_content,
                self._skills_dir,
            ),
            "validate_skill": lambda name, **_kw: validate_makefile_skill(
                name,
                self._skills_dir,
            ),
            "write_file": lambda path, content, **_kw: write_file(
                path,
                content,
                self._base_dir,
            ),
            "edit_file": lambda path, old_text, new_text, **_kw: edit_file(
                path,
                old_text,
                new_text,
                self._base_dir,
            ),
        }
        self.tool_names = frozenset(self._executors)

    @property
    def schemas(self) -> list[dict]:
        return self._schemas

    @property
    def executors(self) -> dict[str, Any]:
        return self._executors

    async def setup(self, model: str) -> None:
        del model

    def get_skill_trusted(self, name: str) -> bool | None:
        return None


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


def list_python_skills(skills_dir: str) -> str:
    """List all available skills with their names and descriptions."""
    path = Path(skills_dir)
    if not path.exists():
        return "No skills found (directory does not exist)"
    skill_dirs = sorted(
        p for p in path.iterdir() if p.is_dir() and (p / "skill.md").exists()
    )
    if not skill_dirs:
        return "No skills found"
    entries = []
    for skill_dir in skill_dirs:
        desc = _skill_description(skill_dir / "skill.md")
        entry = f"{skill_dir.name}:{desc}"
        if (skill_dir / "skill.py").exists():
            entry += "  [has tools]"
        entries.append(entry)
    return "\n\n".join(entries)


def read_python_skill(name: str, skills_dir: str) -> str:
    """Read a skill's instructions (skill.md only)."""
    if not _valid_skill_name(name):
        return f"Error: invalid skill name {name!r}. Use letters, numbers, hyphens, underscores, and dots only."
    skill_dir = Path(skills_dir) / name
    md_path = skill_dir / "skill.md"
    if not skill_dir.exists():
        return f"Skill '{name}' not found in {skills_dir}"
    if not md_path.exists():
        return f"Skill '{name}' is missing skill.md"
    try:
        return md_path.read_text(encoding="utf-8")
    except OSError as e:
        return f"Error: could not read skill.md: {e}"


def _format_param_help(meta: Any) -> str:
    """Format a one-line param description for error messages."""
    default_str = "" if meta.required else f"  [default: {meta.default!r}]"
    req_str = "  [required]" if meta.required else ""
    desc_str = f"  — {meta.description}" if meta.description else ""
    return f"  {meta.name}: {meta.json_type}{req_str}{default_str}{desc_str}"


async def execute_python_skill(
    name: str,
    target: str,
    skills_dir: str,
    registry: SkillRegistry,
    kwargs: dict[str, Any] | None = None,
    timeout: int = 600,
) -> str:
    """Run *target* in skill *name* using keyword arguments."""
    if not _valid_skill_name(name):
        return f"Error: invalid skill name {name!r}. Use letters, numbers, hyphens, underscores, and dots only."
    safe_paths = _resolve_safe_skill_path(skills_dir, name, "skill.py")
    if isinstance(safe_paths, str):
        return safe_paths
    skill_dir, py_path = safe_paths
    if not skill_dir.exists() or not (skill_dir / "skill.md").exists():
        return f"Skill '{name}' not found in {skills_dir}"
    if not py_path.exists():
        return f"Skill '{name}' has no skill.py"

    entry = await registry.get_entry(name)
    if entry is None:
        entry = await registry.load_or_add(name, skills_dir)
    if entry is None or not entry.valid:
        reason = entry.reject_reason if entry else "unknown"
        return f"Error: skill '{name}' is not valid: {reason}"

    tool_meta = entry.tools.get(target)
    if tool_meta is None:
        available = ", ".join(entry.tools) or "(none)"
        return f"Error: skill '{name}' has no target '{target}'. Available: {available}"

    provided_kwargs = kwargs or {}
    missing = [
        param
        for param in tool_meta.params
        if param.required and param.name not in provided_kwargs
    ]
    if missing:
        lines = [f"Error: missing required argument(s) for '{name}/{target}':"]
        for param in missing:
            lines.append(
                f"  {param.name}: {param.json_type}  — {param.description}"
                if param.description
                else f"  {param.name}: {param.json_type}"
            )
        lines.append(f"\n'{target}' expects:")
        for param in tool_meta.params:
            lines.append(_format_param_help(param))
        return "\n".join(lines)

    final_kwargs: dict[str, Any] = {}
    for param in tool_meta.params:
        if param.name in provided_kwargs:
            final_kwargs[param.name] = provided_kwargs[param.name]
        elif not param.required and param.default is not None:
            final_kwargs[param.name] = param.default

    return await _run_python_skill_in_subprocess(
        py_path,
        target,
        final_kwargs,
        timeout,
    )


def _write_no_symlink(path: Path, content: str) -> None:
    """Write *content* to *path*, refusing symlink destinations."""
    if path.is_symlink():
        raise ValueError(f"refusing to overwrite symlink: {path}")
    path.write_text(content, encoding="utf-8")


async def create_python_skill(
    name: str,
    description: str,
    md_content: str,
    skills_dir: str,
    registry: SkillRegistry,
    py_content: str | None = None,
) -> str:
    """Create or overwrite a skill directory with skill.md and optional skill.py."""
    if not _valid_skill_name(name):
        return f"Error: invalid skill name {name!r}. Use letters, numbers, hyphens, underscores, and dots only."

    md_safe = _resolve_safe_skill_path(
        skills_dir, name, "skill.md", create_dirs=True
    )
    if isinstance(md_safe, str):
        return md_safe
    skill_dir, md_path = md_safe
    py_path = skill_dir / "skill.py"
    if py_content:
        py_safe = _resolve_safe_skill_path(
            skills_dir, name, "skill.py", create_dirs=True
        )
        if isinstance(py_safe, str):
            return py_safe
        _, py_path = py_safe

    if not md_content.strip().startswith("---"):
        md_with_frontmatter = f'---\ndescription: "{description}"\n---\n\n{md_content}'
    else:
        md_with_frontmatter = md_content

    try:
        _write_no_symlink(md_path, md_with_frontmatter)
    except (OSError, ValueError) as e:
        return f"Error: could not write skill.md: {e}"

    if py_content:
        try:
            _write_no_symlink(py_path, py_content)
        except (OSError, ValueError) as e:
            return f"Error: could not write skill.py: {e}"
        entry = await registry.load_or_add(name, skills_dir)
        if entry is None or not entry.valid:
            reason = entry.reject_reason if entry else "unknown"
            py_path.unlink(missing_ok=True)
            return f"Error: skill.py rejected: {reason}"
        return f"Created skill '{name}' at {skill_dir} ({len(entry.tools)} tool(s))"

    return f"Created skill '{name}' at {skill_dir} (no tools)"


async def validate_python_skill(
    name: str, skills_dir: str, registry: SkillRegistry
) -> str:
    """Validate a skill: checks skill.md exists and runs AST trust check on skill.py."""
    if not _valid_skill_name(name):
        return f"Error: invalid skill name {name!r}. Use letters, numbers, hyphens, underscores, and dots only."
    md_safe = _resolve_safe_skill_path(skills_dir, name, "skill.md")
    if isinstance(md_safe, str):
        return md_safe
    skill_dir, md_path = md_safe
    if not skill_dir.exists():
        return f"Skill '{name}' not found in {skills_dir}"
    if not md_path.exists():
        return f"Skill '{name}' is missing skill.md"
    py_path = skill_dir / "skill.py"
    if not py_path.exists():
        return "OK — trusted (skill.md only, no tools)"
    entry = await registry.load_or_add(name, skills_dir)
    if entry is None or not entry.valid:
        reason = entry.reject_reason if entry else "unknown"
        return f"INVALID: {reason}"
    trust_label = "trusted" if entry.trusted else "untrusted"
    return f"OK — {trust_label} ({len(entry.tools)} tool(s))"


PYTHON_SKILL_SCHEMAS: list[dict[str, Any]] = [
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
            "description": "Read a skill's instructions from skill.md. Call this first to learn what targets and parameters are available before calling execute_skill.",
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
                "Run a target in a skill's skill.py. "
                "Only usable when the skill has a skill.py file with @target functions. "
                "Call read_skill first to learn what targets and parameters are available."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The skill name (directory name).",
                    },
                    "target": {
                        "type": "string",
                        "description": "The target function name to run.",
                    },
                    "kwargs": {
                        "type": "object",
                        "description": "Keyword arguments to pass to the target function.",
                        "additionalProperties": True,
                    },
                },
                "required": ["name", "target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_skill",
            "description": (
                "Create a new skill or overwrite an existing one. "
                "A skill consists of skill.md (instructions) and optionally skill.py (tools). "
                "skill.py uses @target decorated functions. After creation, skill.py is automatically validated."
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
                    "py_content": {
                        "type": "string",
                        "description": (
                            "Optional: content of skill.py with @target decorated functions. "
                            "Import target with: from make_agent import target"
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
            "description": "Validate a skill: checks skill.md exists and runs LLM security check on skill.py if present.",
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


class PythonSkillBackend:
    def __init__(self, skills_dir: str, tool_timeout: int = 600) -> None:
        self._skills_dir = skills_dir
        self._tool_timeout = tool_timeout
        self._registry: SkillRegistry | None = None
        self._schemas = PYTHON_SKILL_SCHEMAS
        self._executors: dict[str, Any] = {
            "list_skills": lambda **_kw: list_python_skills(self._skills_dir),
            "read_skill": lambda name, **_kw: read_python_skill(name, self._skills_dir),
            "execute_skill": lambda name, target, kwargs=None, **_kw: (
                execute_python_skill(
                    name,
                    target,
                    self._skills_dir,
                    self._require_registry(),
                    kwargs,
                    self._tool_timeout,
                )
            ),
            "create_skill": lambda name, description, md_content, py_content=None, **_kw: (
                create_python_skill(
                    name,
                    description,
                    md_content,
                    self._skills_dir,
                    self._require_registry(),
                    py_content,
                )
            ),
            "validate_skill": lambda name, **_kw: validate_python_skill(
                name,
                self._skills_dir,
                self._require_registry(),
            ),
        }
        self.tool_names = frozenset(self._executors)

    @property
    def schemas(self) -> list[dict]:
        return self._schemas

    @property
    def executors(self) -> dict[str, Any]:
        return self._executors

    async def setup(self, model: str) -> None:
        self._registry = SkillRegistry()
        await self._registry.load_skills_dir(self._skills_dir)

    def _require_registry(self) -> SkillRegistry:
        if self._registry is None:
            raise RuntimeError(
                "PythonSkillBackend.setup() must be called before executing skill tools"
            )
        return self._registry

    def get_skill_trusted(self, name: str) -> bool | None:
        if self._registry is None:
            return None
        entry = self._registry.get_cached_entry(name)
        if entry is None:
            return None
        return entry.trusted if entry.valid else None


__all__ = [
    "MakefileSkillBackend",
    "PythonSkillBackend",
    "PYTHON_SKILL_SCHEMAS",
    "SkillBackend",
]
