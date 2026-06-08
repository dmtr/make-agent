"""SkillBackend protocol and MakefileSkillBackend implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from make_agent.builtin_tools.file_tools import FILE_SCHEMAS, edit_file, write_file
from make_agent.builtin_tools.skill_tools import SKILL_SCHEMAS as MAKEFILE_SKILL_SCHEMAS
from make_agent.builtin_tools.skill_tools import create_skill as create_makefile_skill
from make_agent.builtin_tools.skill_tools import execute_skill as execute_makefile_skill
from make_agent.builtin_tools.skill_tools import list_skills as list_makefile_skills
from make_agent.builtin_tools.skill_tools import read_skill as read_makefile_skill
from make_agent.builtin_tools.skill_tools import validate_skill as validate_makefile_skill


class SkillBackend(Protocol):
    @property
    def schemas(self) -> list[dict]: ...

    @property
    def executors(self) -> dict[str, Any]: ...

    def get_skill_trusted(self, name: str) -> bool | None: ...


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
            "read_skill": lambda name, **_kw: read_makefile_skill(name, self._skills_dir),
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

    def get_skill_trusted(self, name: str) -> bool | None:
        return None


__all__ = [
    "MakefileSkillBackend",
    "SkillBackend",
]
