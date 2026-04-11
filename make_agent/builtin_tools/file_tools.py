"""Built-in file editing tools: read_file, patch_file.

These tools give every agent sandboxed file access (all paths must resolve
within the working directory).

``read_file`` returns the raw file content for a line range.
``patch_file`` applies a standard unified diff (``diff -u`` / ``git diff``
format) to a file, creating it if it does not yet exist.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _resolve_and_validate(file_path: str) -> Path:
    """Resolve *file_path* against cwd and verify it stays within the subtree.

    Raises ``ValueError`` on traversal attempts.
    """
    cwd = Path(os.getcwd()).resolve()
    resolved = (cwd / file_path).resolve()
    if not str(resolved).startswith(str(cwd) + os.sep) and resolved != cwd:
        raise ValueError(f"path escapes working directory: {file_path}")
    return resolved


def _read_lines(path: Path) -> list[str]:
    """Read *path* as UTF-8 and return line strings (without newlines).

    Raises ``ValueError`` for binary files (detected by null-byte check).
    """
    raw = path.read_bytes()
    if b"\x00" in raw:
        raise ValueError(f"binary file, cannot edit: {path.name}")
    text = raw.decode("utf-8", errors="replace")
    if text.endswith("\n"):
        text = text[:-1]
    if text == "":
        return []
    return text.split("\n")


def _write_lines(path: Path, lines: list[str]) -> None:
    """Write *lines* to *path* with a trailing newline."""
    path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")


def _apply_unified_diff(lines: list[str], diff: str) -> list[str] | str:
    """Apply *diff* (standard unified format) to *lines*.

    Returns the updated line list, or an ``"error: …"`` string on failure.

    Hunk header semantics follow ``diff -u``:
    - ``@@ -old_start[,old_count] +new_start[,new_count] @@``
    - Omitted count means 1; ``-0,0`` / ``+0,0`` means 0 lines (pure insert/delete).
    - For a pure insertion (old_count == 0) the splice point is *after*
      line ``old_start`` (i.e. ``result[old_start:old_start]``).
    """
    if not diff.strip():
        return "error: DIFF is empty"

    raw = diff.splitlines()
    i = 0
    while i < len(raw) and not _HUNK_HEADER_RE.match(raw[i]):
        i += 1

    hunks: list[tuple[int, int, list[str], list[str]]] = []

    while i < len(raw):
        m = _HUNK_HEADER_RE.match(raw[i])
        if not m:
            i += 1
            continue

        old_start = int(m.group(1))
        old_count = int(m.group(2)) if m.group(2) is not None else 1
        new_count = int(m.group(4)) if m.group(4) is not None else 1
        i += 1

        old_body: list[str] = []
        new_body: list[str] = []

        while i < len(raw):
            body = raw[i]
            if _HUNK_HEADER_RE.match(body):
                break
            if body.startswith("\\ "):
                i += 1
                continue
            if body.startswith(" "):
                old_body.append(body[1:])
                new_body.append(body[1:])
            elif body.startswith("-"):
                old_body.append(body[1:])
            elif body.startswith("+"):
                new_body.append(body[1:])
            else:
                return f"error: unexpected diff line: {body!r}"
            i += 1

        if len(old_body) != old_count:
            return (
                f"error: hunk old-count mismatch at line {old_start}: "
                f"header says {old_count}, body has {len(old_body)}"
            )
        if len(new_body) != new_count:
            return (
                f"error: hunk new-count mismatch at line {old_start}: "
                f"header says {new_count}, body has {len(new_body)}"
            )

        hunks.append((old_start, old_count, old_body, new_body))

    if not hunks:
        return "error: no hunks found in diff"

    result = list(lines)
    for old_start, old_count, old_body, new_body in reversed(hunks):
        start_idx = old_start if old_count == 0 else old_start - 1
        actual = result[start_idx : start_idx + old_count]
        if actual != old_body:
            return (
                f"error: patch context mismatch at line {old_start}: "
                f"expected {old_body!r}, found {actual!r}"
            )
        result[start_idx : start_idx + old_count] = new_body

    return result


def read_file(FILE_PATH: str, START_LINE: int, END_LINE: int) -> str:
    """Read a range of lines from a file and return them as plain text.

    Returns the raw file content for lines START_LINE..END_LINE (1-indexed,
    inclusive) with no added prefixes.  On error returns ``"error: …"``.
    """
    try:
        path = _resolve_and_validate(FILE_PATH)
    except ValueError as e:
        return f"error: {e}"

    if not path.exists():
        return f"error: file not found: {FILE_PATH}"
    if not path.is_file():
        return f"error: not a file: {FILE_PATH}"

    try:
        lines = _read_lines(path)
    except ValueError as e:
        return f"error: {e}"

    total = len(lines)
    start = int(START_LINE)
    end = int(END_LINE)

    if start < 1:
        return f"error: START_LINE must be >= 1, got {start}"
    if end < start:
        return f"error: END_LINE ({end}) must be >= START_LINE ({start})"
    if start > total:
        return f"error: START_LINE ({start}) exceeds file length (total_lines: {total})"
    if end > total:
        return f"error: END_LINE ({end}) exceeds file length (total_lines: {total})"

    return "\n".join(lines[start - 1 : end])


def patch_file(FILE_PATH: str, DIFF: str) -> str:
    """Apply a standard unified diff to a file.

    DIFF must be in ``diff -u`` / ``git diff`` format.  A missing file may be
    created from an all-added patch (``@@ -0,0 +1,N @@``).  Parent directories
    are created automatically.  Returns ``"Changes accepted."`` on success.
    """
    try:
        path = _resolve_and_validate(FILE_PATH)
    except ValueError as e:
        return f"error: {e}"

    if path.is_dir():
        return f"error: path is a directory: {FILE_PATH}"

    lines: list[str] = []
    if path.exists():
        if not path.is_file():
            return f"error: not a file: {FILE_PATH}"
        try:
            lines = _read_lines(path)
        except ValueError as e:
            return f"error: {e}"

    result = _apply_unified_diff(lines, DIFF)
    if isinstance(result, str):
        return result

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_lines(path, result)
    except OSError as e:
        return f"error: {e}"

    return "Changes accepted."


FILE_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a range of lines from a file. "
                "Returns the raw file content for lines START_LINE..END_LINE "
                "(1-indexed, inclusive) with no added prefixes. "
                "Both START_LINE and END_LINE are required."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "FILE_PATH": {
                        "type": "string",
                        "description": "File path relative to the working directory.",
                    },
                    "START_LINE": {
                        "type": "integer",
                        "description": "First line to read (1-indexed, inclusive).",
                    },
                    "END_LINE": {
                        "type": "integer",
                        "description": "Last line to read (1-indexed, inclusive).",
                    },
                },
                "required": ["FILE_PATH", "START_LINE", "END_LINE"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "patch_file",
            "description": (
                "Apply a standard unified diff (diff -u / git diff format) to a file. "
                "A missing file can be created from an all-added patch. "
                "Returns 'Changes accepted.' on success."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "FILE_PATH": {
                        "type": "string",
                        "description": "File path relative to the working directory.",
                    },
                    "DIFF": {
                        "type": "string",
                        "description": (
                            "Unified diff in diff -u / git diff format. "
                            "Example: '@@ -2 +2 @@\\n-old line\\n+new line'."
                        ),
                    },
                },
                "required": ["FILE_PATH", "DIFF"],
            },
        },
    },
]

FILE_TOOL_NAMES: frozenset[str] = frozenset({"read_file", "patch_file"})


def get_file_tools(disabled: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Return a name → callable mapping for file editing tools."""
    tools: dict[str, Any] = {
        "read_file": lambda FILE_PATH, START_LINE, END_LINE, **_kw: read_file(FILE_PATH, START_LINE, END_LINE),
        "patch_file": lambda FILE_PATH, DIFF, **_kw: patch_file(FILE_PATH, DIFF),
    }
    return {name: fn for name, fn in tools.items() if name not in disabled}
