"""Built-in file editing tools: read_file, write_file.

These tools give every agent structured, line-level file access with
path sandboxing (all paths must resolve within the working directory).

Both tools use the same numbered-line format:

    1. First line of the file
    2. Second line of the file
    ...

``read_file`` returns this format. ``write_file`` accepts it and performs
a partial update, replacing only the line numbers present in CONTENT.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


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
    """Read *path* as UTF-8 and return a list of line strings (without newlines).

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
    """Write *lines* back to *path* with a trailing newline."""
    path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")


_NUMBERED_LINE_RE = re.compile(r"^(\d+)\. (.*)$")


def _format_numbered_lines(lines: list[str], start: int, end: int) -> str:
    """Return lines[start-1..end-1] (1-indexed, inclusive) in numbered format."""
    result = []
    for i in range(max(start, 1), min(end, len(lines)) + 1):
        result.append(f"{i}. {lines[i - 1]}")
    return "\n".join(result)


def _parse_numbered_lines(content: str) -> list[tuple[int, str]] | str:
    """Parse numbered-line format into a list of (line_num, text) pairs.

    Returns an error string if any line does not match the expected format.
    """
    result = []
    for raw in content.splitlines():
        m = _NUMBERED_LINE_RE.match(raw)
        if not m:
            return f"error: invalid line format: {raw!r} (expected 'N. content')"
        result.append((int(m.group(1)), m.group(2)))
    return result


def read_file(FILE_PATH: str, START_LINE: int, END_LINE: int) -> str:
    """Read lines from a file, returning them in numbered format.

    Each output line looks like ``N. content``.  On error a plain-text
    ``error: …`` message is returned instead.
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

    return _format_numbered_lines(lines, start, end)


def write_file(FILE_PATH: str, CONTENT: str) -> str:
    """Update a file using numbered-line content.

    CONTENT must use the same ``N. text`` format returned by ``read_file``.
    Only the line numbers present in CONTENT are modified; all other lines
    are preserved.  Pass an empty string to create or truncate to an empty
    file.  The file (and any parent directories) are created if needed.

    Returns ``"Changes accepted."`` on success, or a plain-text
    ``error: …`` message on failure.
    """
    try:
        path = _resolve_and_validate(FILE_PATH)
    except ValueError as e:
        return f"error: {e}"

    if path.is_dir():
        return f"error: path is a directory: {FILE_PATH}"

    if not CONTENT.strip():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        except OSError as e:
            return f"error: {e}"
        return "Changes accepted."

    parsed = _parse_numbered_lines(CONTENT)
    if isinstance(parsed, str):
        return parsed

    lines: list[str] = []
    if path.exists() and path.is_file():
        try:
            lines = _read_lines(path)
        except ValueError as e:
            return f"error: {e}"

    for line_num, text in parsed:
        if line_num < 1:
            return f"error: line number must be >= 1, got {line_num}"
        while len(lines) < line_num:
            lines.append("")
        lines[line_num - 1] = text

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_lines(path, lines)
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
                "Returns a multiline string where every line is prefixed with its "
                "1-indexed line number: '1. first line', '2. second line', etc. "
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
            "name": "write_file",
            "description": (
                "Write lines to a file using the same numbered format as read_file "
                "('1. content', '2. content', …). Only the line numbers present in "
                "CONTENT are updated; all other lines are left unchanged. "
                "Pass an empty string to create or truncate to an empty file. "
                "Parent directories are created automatically. "
                "Returns 'Changes accepted.' on success."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "FILE_PATH": {
                        "type": "string",
                        "description": "File path relative to the working directory.",
                    },
                    "CONTENT": {
                        "type": "string",
                        "description": (
                            "Lines to write in numbered format: '1. first line\\n2. second line\\n…'. "
                            "Pass an empty string to write an empty file."
                        ),
                    },
                },
                "required": ["FILE_PATH", "CONTENT"],
            },
        },
    },
]

FILE_TOOL_NAMES: frozenset[str] = frozenset({"read_file", "write_file"})


def get_file_tools(disabled: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Return a name → callable mapping for file editing tools."""
    tools: dict[str, Any] = {
        "read_file": lambda FILE_PATH, START_LINE, END_LINE, **_kw: read_file(FILE_PATH, START_LINE, END_LINE),
        "write_file": lambda FILE_PATH, CONTENT, **_kw: write_file(FILE_PATH, CONTENT),
    }
    return {name: fn for name, fn in tools.items() if name not in disabled}
