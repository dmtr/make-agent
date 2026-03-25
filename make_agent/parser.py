"""Makefile parser.

Parses a subset of GNU Make syntax into structured data:
  - Variables  (=, :=, ?=, +=)
  - Rules      (target: prerequisites + tab-indented recipes)
  - .PHONY declarations
  - Special comment blocks:
      # <system> … # </system>  — system prompt for the agent
      # <tool>   … # </tool>    — tool description + @param declarations

Inside a ``# <tool>`` block:
  - Lines starting with ``# @param NAME type description`` declare tool
    parameters that are passed to make as ``NAME=value``.
  - All other lines form the human-readable tool description sent to the LLM.

Supported param types
---------------------
``string``, ``number``, ``integer``, ``boolean``
    Standard JSON Schema primitives.  Passed to ``make`` as ``NAME=value``.

``content``
    Arbitrary multi-line text (file contents, scripts, …).  The framework
    writes the value to a temporary file and passes ``NAME_FILE=/tmp/…`` to
    ``make`` instead.  Recipes must therefore reference ``$(NAME_FILE)``::

        # @param BODY content  File text to write
        write-file:
            @cat "$(BODY_FILE)" > "$(FILE)"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

_FLAVOR_MAP = {
    "=": "recursive",
    ":=": "simple",
    "::=": "simple",
    "?=": "conditional",
    "+=": "append",
}

# Matches: NAME = value  /  NAME := value  /  NAME ?= value  /  NAME += value
_VARIABLE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(::?=|\?=|\+=|=)\s*(.*)")

# Matches: target(s): prerequisites
# [^\t#]    — first char is not a tab or comment marker
# [^:=]*?   — non-greedy, no colons or equals in the target portion
# :(?![:=]) — a single colon NOT followed by : or =
_TARGET_RE = re.compile(r"^([^\t#][^:=]*?)\s*:(?![:=])(.*)")

# Matches: .PHONY: target …
_PHONY_RE = re.compile(r"^\.PHONY\s*:(.*)")

# Matches: $(VAR), ${VAR}, or $X variable references
_VAR_REF_RE = re.compile(r"\$(?:\(([^)]+)\)|\{([^}]+)\}|([A-Za-z_]))")

# Matches: @param NAME type description  (inside a <tool> block)
_PARAM_RE = re.compile(r"^@param\s+(\w+)\s+(\w+)\s+(.+)")


@dataclass
class Variable:
    name: str
    value: str
    flavor: str


@dataclass
class Param:
    name: str
    type: str          # JSON Schema primitive: string, number, integer, boolean
    description: str


@dataclass
class Rule:
    target: str
    prerequisites: list[str] = field(default_factory=list)
    recipes: list[str] = field(default_factory=list)
    is_phony: bool = False
    description: str | None = None
    params: list[Param] = field(default_factory=list)


@dataclass
class Makefile:
    system_prompt: str | None = None
    variables: dict[str, Variable] = field(default_factory=dict)
    rules: list[Rule] = field(default_factory=list)
    default_target: str | None = None  # first non-special target


def _expand_vars(s: str, variables: dict[str, Variable]) -> str:
    """Expand $(VAR), ${VAR}, and $X references using already-seen variables."""

    def replace(m: re.Match) -> str:  # type: ignore[type-arg]
        name = m.group(1) or m.group(2) or m.group(3)
        return variables[name].value if name in variables else m.group(0)

    return _VAR_REF_RE.sub(replace, s)


def _strip_comment(s: str) -> str:
    """Strip an inline # comment (not preceded by a backslash)."""
    i = 0
    while i < len(s):
        if s[i] == "#" and (i == 0 or s[i - 1] != "\\"):
            return s[:i].rstrip()
        i += 1
    return s.rstrip()


class _State(Enum):
    NORMAL = auto()
    RECIPE = auto()
    SYSTEM_BLOCK = auto()
    TOOL_BLOCK = auto()


def parse(text: str) -> Makefile:
    """Parse Makefile text and return a structured :class:`Makefile` object."""
    result = Makefile()
    state = _State.NORMAL
    current_recipes: list[str] | None = None
    pending_description: str | None = None
    pending_params: list[Param] = []
    system_lines: list[str] = []
    tool_lines: list[str] = []
    tool_params: list[Param] = []
    phony_targets: set[str] = set()

    # Join line continuations (backslash-newline → space)
    logical_lines: list[str] = []
    buf = ""
    for raw in text.splitlines():
        if raw.endswith("\\"):
            buf += raw[:-1] + " "
        else:
            logical_lines.append(buf + raw)
            buf = ""
    if buf:
        logical_lines.append(buf)

    for line in logical_lines:
        # Recipe lines must start with a tab
        if line.startswith("\t"):
            if state == _State.RECIPE and current_recipes is not None:
                current_recipes.append(line[1:])  # strip the leading tab
            continue

        # A non-tab line ends recipe collection.
        # Exception: plain comment lines (not special block tags) are ignored by
        # GNU Make inside a rule body and should not end recipe collection.
        if state == _State.RECIPE:
            stripped_peek = line.strip()
            if stripped_peek.startswith("#") and stripped_peek not in ("# <tool>", "# <system>", "# </tool>", "# </system>"):
                continue
            state = _State.NORMAL
            current_recipes = None

        stripped = line.strip()

        # ── Inside a special comment block ──────────────────────────────────
        if state == _State.SYSTEM_BLOCK:
            if stripped == "# </system>":
                result.system_prompt = "\n".join(system_lines).strip() or None
                state = _State.NORMAL
            elif stripped.startswith("# "):
                system_lines.append(stripped[2:])
            elif stripped == "#":
                system_lines.append("")
            continue

        if state == _State.TOOL_BLOCK:
            if stripped == "# </tool>":
                pending_description = "\n".join(tool_lines).strip() or None
                pending_params = tool_params
                tool_lines = []
                tool_params = []
                state = _State.NORMAL
            elif stripped.startswith("# "):
                content = stripped[2:]
                param_m = _PARAM_RE.match(content)
                if param_m:
                    tool_params.append(Param(
                        name=param_m.group(1),
                        type=param_m.group(2),
                        description=param_m.group(3).strip(),
                    ))
                else:
                    tool_lines.append(content)
            elif stripped == "#":
                tool_lines.append("")
            continue

        # ── Block-opening tags ───────────────────────────────────────────────
        if stripped == "# <system>":
            system_lines = []
            state = _State.SYSTEM_BLOCK
            continue

        if stripped == "# <tool>":
            tool_lines = []
            tool_params = []
            state = _State.TOOL_BLOCK
            continue

        # ── Skip regular comments and blank lines ────────────────────────────
        if not stripped or stripped.startswith("#"):
            continue

        # ── .PHONY declaration ───────────────────────────────────────────────
        phony_m = _PHONY_RE.match(stripped)
        if phony_m:
            new_phonies = phony_m.group(1).split()
            phony_targets.update(new_phonies)
            for rule in result.rules:
                if rule.target in phony_targets:
                    rule.is_phony = True
            continue

        # ── Variable assignment ──────────────────────────────────────────────
        var_m = _VARIABLE_RE.match(line.rstrip())
        if var_m:
            name = var_m.group(1)
            op = var_m.group(2).strip()
            raw_val = _strip_comment(var_m.group(3))
            expanded = _expand_vars(raw_val, result.variables)
            flavor = _FLAVOR_MAP.get(op, "recursive")
            if op == "+=" and name in result.variables:
                result.variables[name].value = (result.variables[name].value + " " + expanded).strip()
            else:
                result.variables[name] = Variable(name=name, value=expanded, flavor=flavor)
            continue

        # ── Target rule ──────────────────────────────────────────────────────
        target_m = _TARGET_RE.match(line.rstrip())
        if target_m:
            targets_str = target_m.group(1).strip()
            prereqs_str = _strip_comment(target_m.group(2) or "")
            targets = [_expand_vars(t, result.variables) for t in targets_str.split()]
            prereqs = [_expand_vars(p, result.variables) for p in prereqs_str.split()]
            shared_recipes: list[str] = []
            for i, target in enumerate(targets):
                rule = Rule(
                    target=target,
                    prerequisites=prereqs,
                    recipes=shared_recipes,
                    is_phony=target in phony_targets,
                    description=pending_description if i == 0 else None,
                    params=pending_params if i == 0 else [],
                )
                result.rules.append(rule)
                if result.default_target is None and not target.startswith("."):
                    result.default_target = target
            pending_description = None
            pending_params = []
            state = _State.RECIPE
            current_recipes = shared_recipes

    return result


def parse_file(path: str | Path) -> Makefile:
    """Parse a Makefile from disk."""
    return parse(Path(path).read_text())


# Matches $(NAME), ${NAME}, or $$NAME (Make double-$ shell-variable form) in recipe text
_RECIPE_VAR_RE = re.compile(r"\$\(([^)]+)\)|\$\{([^}]+)\}|\$\$(\w+)")


def validate(makefile: Makefile) -> list[str]:
    """Check that every ``@param NAME`` is referenced in the rule's recipe body.

    For standard params, any of ``$(NAME)``, ``${NAME}``, or ``$$NAME`` counts.
    For ``content``-typed params, ``$(NAME_FILE)`` / ``${NAME_FILE}`` /
    ``$$NAME_FILE`` is also accepted (because the framework injects the temp
    file path under that variable name instead of the raw value).

    Returns a list of human-readable error strings (empty list means valid).
    """
    errors: list[str] = []
    for rule in makefile.rules:
        if not rule.params:
            continue
        recipe_text = "\n".join(rule.recipes)
        used_vars = {m.group(1) or m.group(2) or m.group(3) for m in _RECIPE_VAR_RE.finditer(recipe_text)}
        for param in rule.params:
            file_var = f"{param.name}_FILE"
            is_content = param.type == "content"
            referenced = param.name in used_vars or (is_content and file_var in used_vars)
            if not referenced:
                hint = f"$({param.name}), ${{{param.name}}}, $${param.name}"
                if is_content:
                    hint += f", or $({file_var}) (recommended for content params)"
                errors.append(
                    f"Tool '{rule.target}': @param {param.name} declared but never "
                    f"referenced in recipe.\n"
                    f"  Expected {hint} in the recipe body."
                )
    return errors


def validate_or_raise(makefile: Makefile) -> None:
    """Like :func:`validate` but raises :exc:`ValueError` if any errors are found."""
    errors = validate(makefile)
    if errors:
        raise ValueError("\n".join(errors))
