"""In-memory skill registry with hash-based AST-based trust validation.

At agent startup :meth:`SkillRegistry.load_skills_dir` scans for ``skill.py``
files, validates each with a syntax check and an AST trust check, then
imports validated modules and collects their ``_TARGETS`` dicts.

Before executing a target :meth:`SkillRegistry.get_entry` re-hashes the file
and re-validates automatically if the content has changed.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import logging
import py_compile
import sys
from dataclasses import dataclass, field
from pathlib import Path

from make_agent.skill import ToolMeta

logger = logging.getLogger(__name__)

_DANGEROUS_MODULES = frozenset(
    {
        "subprocess",
        "socket",
        "requests",
        "httpx",
        "aiohttp",
        "urllib",
        "http",
        "ftplib",
        "smtplib",
        "imaplib",
        "poplib",
        "telnetlib",
        "xmlrpc",
    }
)

_DANGEROUS_BUILTINS = frozenset({"exec", "eval", "__import__"})

_DANGEROUS_OS_ATTRS = frozenset({"system", "popen", "execv", "execve", "execvp", "execvpe", "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe"})


@dataclass
class SkillEntry:
    name: str
    path: Path  # path to skill.py
    hash: str  # SHA-256 hex digest
    valid: bool
    reject_reason: str | None
    trusted: bool = False
    tools: dict[str, ToolMeta] = field(default_factory=dict)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ast_trust_check(code: str) -> tuple[bool, list[str]]:
    """Inspect *code* with AST analysis and return ``(is_trusted, detected_patterns)``.

    A skill is trusted when its source contains no dangerous imports or calls.
    Detected patterns are logged at DEBUG level only.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, ["unparseable"]

    detected: list[str] = []

    for node in ast.walk(tree):
        # import subprocess / import socket / import requests …
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _DANGEROUS_MODULES:
                    detected.append(root)

        # from subprocess import … / from urllib.request import …
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in _DANGEROUS_MODULES:
                    detected.append(root)

        # exec(...) / eval(...) / __import__(...)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _DANGEROUS_BUILTINS:
                detected.append(func.id)
            # os.system(...) / os.popen(...) etc.
            elif (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "os"
                and func.attr in _DANGEROUS_OS_ATTRS
            ):
                detected.append(f"os.{func.attr}")
            # asyncio.create_subprocess_exec / asyncio.create_subprocess_shell
            elif (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "asyncio"
                and func.attr.startswith("create_subprocess_")
            ):
                detected.append(f"asyncio.{func.attr}")

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for p in detected:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    return len(unique) == 0, unique


def _syntax_check(path: Path) -> str | None:
    """Return an error string if *path* has a syntax error, else ``None``."""
    try:
        py_compile.compile(str(path), doraise=True)
        return None
    except py_compile.PyCompileError as e:
        return str(e)


def _import_skill(path: Path) -> dict[str, ToolMeta]:
    """Import *path* as a fresh module and return its ``_TARGETS`` dict."""
    module_name = f"_make_agent_skill_{path.parent.name}_{abs(hash(str(path)))}"
    sys.modules.pop(module_name, None)  # force re-import on edit
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return dict(getattr(module, "_TARGETS", {}))


class SkillRegistry:
    """In-memory registry of validated, imported skill.py modules."""

    def __init__(self) -> None:
        self._entries: dict[str, SkillEntry] = {}

    async def load_skills_dir(self, skills_dir: str) -> None:
        """Scan *skills_dir* and validate + import every ``skill.py`` found."""
        path = Path(skills_dir)
        if not path.exists():
            return
        for skill_dir in sorted(p for p in path.iterdir() if p.is_dir()):
            py_path = skill_dir / "skill.py"
            md_path = skill_dir / "skill.md"
            if py_path.exists() and md_path.exists():
                entry = await self._build_entry(skill_dir.name, py_path)
                self._entries[skill_dir.name] = entry
                if not entry.valid:
                    logger.warning(
                        "Skill %r rejected at startup: %s",
                        skill_dir.name,
                        entry.reject_reason,
                    )

    async def _build_entry(self, name: str, py_path: Path) -> SkillEntry:
        file_hash = _sha256(py_path)

        syntax_error = _syntax_check(py_path)
        if syntax_error:
            return SkillEntry(
                name=name,
                path=py_path,
                hash=file_hash,
                valid=False,
                trusted=False,
                reject_reason=f"Syntax error: {syntax_error}",
            )

        code = py_path.read_text(encoding="utf-8")
        is_trusted, patterns = _ast_trust_check(code)
        if patterns:
            logger.debug("Skill %r detected dangerous patterns: %s", name, patterns)

        try:
            tools = _import_skill(py_path)
        except Exception as e:
            return SkillEntry(
                name=name,
                path=py_path,
                hash=file_hash,
                valid=False,
                trusted=False,
                reject_reason=f"Import error: {e}",
            )

        return SkillEntry(
            name=name,
            path=py_path,
            hash=file_hash,
            valid=True,
            trusted=is_trusted,
            reject_reason=None,
            tools=tools,
        )

    async def get_entry(self, name: str) -> SkillEntry | None:
        """Return the entry for *name*, re-validating if ``skill.py`` changed."""
        entry = self._entries.get(name)
        if entry is None:
            return None
        try:
            current_hash = _sha256(entry.path)
        except OSError:
            return entry
        if current_hash != entry.hash:
            logger.info("Skill %r changed on disk, re-validating", name)
            new_entry = await self._build_entry(name, entry.path)
            self._entries[name] = new_entry
            return new_entry
        return entry

    async def load_or_add(self, name: str, skills_dir: str) -> SkillEntry | None:
        """Validate and import a single skill. Used after ``create_skill`` writes a new file."""
        py_path = Path(skills_dir) / name / "skill.py"
        if not py_path.exists():
            return None
        entry = await self._build_entry(name, py_path)
        self._entries[name] = entry
        return entry

    def get_cached_entry(self, name: str) -> SkillEntry | None:
        return self._entries.get(name)
