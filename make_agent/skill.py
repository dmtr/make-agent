"""@target decorator for skill.py tool functions.

Each function decorated with ``@target`` becomes a callable tool that the agent
can invoke via ``execute_skill``.  The decorator introspects the function
signature to build :class:`ToolMeta` and stores the result in the module's
``_TARGETS`` dict so the :class:`~make_agent.skill_registry.SkillRegistry` can
collect it after import.

Example skill.py::

    from make_agent import target

    @target
    def read_file(file: str, encoding: str = "utf-8") -> str:
        \"\"\"Read and return the contents of a file.

        :param file: Path to the file to read
        :param encoding: File encoding
        \"\"\"
        with open(file, encoding=encoding) as f:
            return f.read()
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Callable

_PARAM_DOC_RE = re.compile(r":param\s+(\w+):\s*(.*)")

_ANNOTATION_TO_JSON_TYPE: dict = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
}


@dataclass
class ParamMeta:
    name: str
    json_type: str  # JSON Schema primitive: string, integer, number, boolean
    required: bool
    default: Any
    description: str


@dataclass
class ToolMeta:
    name: str
    description: str
    fn: Callable = field(repr=False, compare=False)
    params: list[ParamMeta] = field(default_factory=list)


def _parse_param_docs(docstring: str | None) -> dict[str, str]:
    """Extract ``':param name: description'`` lines from a docstring."""
    if not docstring:
        return {}
    result: dict[str, str] = {}
    for line in docstring.splitlines():
        m = _PARAM_DOC_RE.search(line.strip())
        if m:
            result[m.group(1)] = m.group(2).strip()
    return result


def _first_line(docstring: str | None) -> str:
    """Return the first non-empty line of a docstring."""
    if not docstring:
        return ""
    for line in docstring.strip().splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def target(fn: Callable) -> Callable:
    """Register *fn* as a skill tool target.

    Stores :class:`ToolMeta` in ``module._TARGETS[fn.__name__]`` so the
    :class:`~make_agent.skill_registry.SkillRegistry` can collect it after
    importing the skill module.  Returns *fn* unmodified.
    """
    sig = inspect.signature(fn)
    param_docs = _parse_param_docs(fn.__doc__)
    description = _first_line(fn.__doc__)

    params: list[ParamMeta] = []
    for pname, p in sig.parameters.items():
        annotation = p.annotation
        json_type = (
            _ANNOTATION_TO_JSON_TYPE.get(annotation, "string")
            if annotation is not inspect.Parameter.empty
            else "string"
        )
        required = p.default is inspect.Parameter.empty
        default = None if required else p.default
        params.append(
            ParamMeta(
                name=pname,
                json_type=json_type,
                required=required,
                default=default,
                description=param_docs.get(pname, ""),
            )
        )

    meta = ToolMeta(name=fn.__name__, description=description, params=params, fn=fn)

    module = inspect.getmodule(fn)
    if module is not None:
        if not hasattr(module, "_TARGETS"):
            module._TARGETS: dict[str, ToolMeta] = {}
        module._TARGETS[fn.__name__] = meta

    return fn


__all__ = ["target", "ToolMeta", "ParamMeta"]
