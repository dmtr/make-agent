"""In-memory skill registry with hash-based LLM security validation.

At agent startup :meth:`SkillRegistry.load_skills_dir` scans for ``skill.py``
files, validates each with a syntax check and an LLM security audit, then
imports validated modules and collects their ``_TARGETS`` dicts.

Before executing a target :meth:`SkillRegistry.get_entry` re-hashes the file
and re-validates automatically if the content has changed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import py_compile
import sys
from dataclasses import dataclass, field
from pathlib import Path

import any_llm

from make_agent.skill import ToolMeta

logger = logging.getLogger(__name__)

_SECURITY_PROMPT = """\
You are a security auditor. Review the following Python code for malicious or \
dangerous patterns.

Check for: exec(), eval(), __import__(), subprocess calls, os.system(), \
shell injection, network exfiltration, file system destruction, attempts to \
access secrets or credentials, and any other clearly malicious behaviour.

Respond with exactly one of:
SAFE
or
UNSAFE: <brief reason>

Do not include any other text.

Code to review:
```python
{code}
```"""


@dataclass
class SkillEntry:
    name: str
    path: Path  # path to skill.py
    hash: str  # SHA-256 hex digest
    valid: bool
    reject_reason: str | None
    tools: dict[str, ToolMeta] = field(default_factory=dict)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def _llm_security_check(code: str, model: str) -> tuple[bool, str | None]:
    """Call the LLM to audit *code*. Returns ``(is_safe, reject_reason)``."""
    logger.debug("Running LLM security check on code:\n%s", code)
    prompt = _SECURITY_PROMPT.format(code=code)
    try:
        stream = await any_llm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            stream=True,
        )
        parts: list[str] = []
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                parts.append(chunk.choices[0].delta.content)
        text = "".join(parts).strip()
    except Exception as e:
        logger.error("LLM security check failed: %s", e)
        return False, f"LLM check error: {e}"

    upper = text.upper()
    logger.debug("LLM security check result: %r", text)
    if upper.startswith("SAFE"):
        return True, None
    reason = text[len("UNSAFE:") :].strip() if upper.startswith("UNSAFE:") else text
    return False, reason or "rejected by security check"


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

    def __init__(self, model: str) -> None:
        self._model = model
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
                reject_reason=f"Syntax error: {syntax_error}",
            )

        code = py_path.read_text(encoding="utf-8")
        is_safe, reason = await _llm_security_check(code, self._model)
        if not is_safe:
            return SkillEntry(
                name=name,
                path=py_path,
                hash=file_hash,
                valid=False,
                reject_reason=reason,
            )

        try:
            tools = _import_skill(py_path)
        except Exception as e:
            return SkillEntry(
                name=name,
                path=py_path,
                hash=file_hash,
                valid=False,
                reject_reason=f"Import error: {e}",
            )

        return SkillEntry(
            name=name,
            path=py_path,
            hash=file_hash,
            valid=True,
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
