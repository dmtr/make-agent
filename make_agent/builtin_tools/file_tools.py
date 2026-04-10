"""Built-in file editing tools: read_file, write_file, replace_lines, insert_lines.

These tools give every agent structured, line-level file access with
path sandboxing (all paths must resolve within the working directory).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _resolve_and_validate(file_path: str) -> Path:
    """Resolve *file_path* against cwd and verify it stays within the subtree.

    Raises ``ValueError`` on traversal attempts or missing files.
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


def _format_result(lines: list[str], start: int, end: int) -> str:
    """Format lines[start-1 .. end-1] (1-indexed, inclusive) as JSON array."""
    result = []
    for i in range(max(start, 1), min(end, len(lines)) + 1):
        result.append({str(i): lines[i - 1]})
    return json.dumps(result)


def read_file(FILE_PATH: str, START_LINE: int, END_LINE: int) -> str:
    """Read lines from a file, returning a JSON array of {line_number: content}."""
    try:
        path = _resolve_and_validate(FILE_PATH)
    except ValueError as e:
        return json.dumps({"error": str(e)})

    if not path.exists():
        return json.dumps({"error": f"file not found: {FILE_PATH}"})
    if not path.is_file():
        return json.dumps({"error": f"not a file: {FILE_PATH}"})

    try:
        lines = _read_lines(path)
    except ValueError as e:
        return json.dumps({"error": str(e)})

    total = len(lines)

    start = int(START_LINE)
    end = int(END_LINE)

    if start < 1:
        return json.dumps({"error": f"START_LINE must be >= 1, got {start}"})
    if end < start:
        return json.dumps({"error": f"END_LINE ({end}) must be >= START_LINE ({start})"})
    if start > total:
        return json.dumps({"error": f"START_LINE ({start}) exceeds file length", "total_lines": total})
    if end > total:
        return json.dumps({"error": f"END_LINE ({end}) exceeds file length", "total_lines": total})

    return _format_result(lines, start, end)


def write_file(FILE_PATH: str, CONTENT: str) -> str:
    """Write full content to a file, creating it if necessary.

    This is a simpler alternative to ``replace_lines`` for when the LLM
    wants to rewrite the whole file or create a new one.
    """
    try:
        path = _resolve_and_validate(FILE_PATH)
    except ValueError as e:
        return json.dumps({"error": str(e)})

    if path.is_dir():
        return json.dumps({"error": f"path is a directory: {FILE_PATH}"})

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(CONTENT, encoding="utf-8")
    except OSError as e:
        return json.dumps({"error": str(e)})

    line_count = CONTENT.count("\n") + (0 if CONTENT.endswith("\n") or not CONTENT else 1)
    return json.dumps({"ok": True, "lines_written": line_count, "path": FILE_PATH})


def replace_lines(FILE_PATH: str, START_LINE: int, END_LINE: int, CONTENT: str) -> str:
    """Replace lines START_LINE..END_LINE (inclusive) with CONTENT. Empty CONTENT deletes the range."""
    try:
        path = _resolve_and_validate(FILE_PATH)
    except ValueError as e:
        return json.dumps({"error": str(e)})

    if not path.exists():
        return json.dumps({"error": f"file not found: {FILE_PATH}"})
    if not path.is_file():
        return json.dumps({"error": f"not a file: {FILE_PATH}"})

    try:
        lines = _read_lines(path)
    except ValueError as e:
        return json.dumps({"error": str(e)})

    start = int(START_LINE)
    end = int(END_LINE)
    total = len(lines)

    if start < 1:
        return json.dumps({"error": f"START_LINE must be >= 1, got {start}"})
    if end < start:
        return json.dumps({"error": f"END_LINE ({end}) must be >= START_LINE ({start})"})
    if start > total:
        return json.dumps({"error": f"START_LINE ({start}) exceeds file length", "total_lines": total})
    if end > total:
        return json.dumps({"error": f"END_LINE ({end}) exceeds file length", "total_lines": total})

    # Parse CONTENT into replacement lines (strip one trailing newline if present)
    if CONTENT.endswith("\n"):
        CONTENT = CONTENT[:-1]
    new_lines = CONTENT.split("\n") if CONTENT else []

    lines[start - 1 : end] = new_lines

    _write_lines(path, lines)

    # Return context window around the affected region
    ctx_start = max(1, start - 3)
    ctx_end = min(len(lines), (start + max(0, len(new_lines) - 1)) + 3)
    return _format_result(lines, ctx_start, ctx_end)


def insert_lines(FILE_PATH: str, START_LINE: int, CONTENT: str) -> str:
    """Insert CONTENT before START_LINE. Existing lines shift down. Use START_LINE = total_lines + 1 to append."""
    try:
        path = _resolve_and_validate(FILE_PATH)
    except ValueError as e:
        return json.dumps({"error": str(e)})

    if not path.exists():
        return json.dumps({"error": f"file not found: {FILE_PATH}"})
    if not path.is_file():
        return json.dumps({"error": f"not a file: {FILE_PATH}"})

    try:
        lines = _read_lines(path)
    except ValueError as e:
        return json.dumps({"error": str(e)})

    start = int(START_LINE)
    total = len(lines)

    if start < 1 or start > total + 1:
        return json.dumps({"error": f"START_LINE ({start}) out of range, valid range is 1-{total + 1}"})

    if CONTENT.endswith("\n"):
        CONTENT = CONTENT[:-1]
    new_lines = CONTENT.split("\n") if CONTENT else []

    lines[start - 1 : start - 1] = new_lines

    _write_lines(path, lines)

    ctx_start = max(1, start - 3)
    ctx_end = min(len(lines), (start + max(0, len(new_lines) - 1)) + 3)
    return _format_result(lines, ctx_start, ctx_end)


FILE_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read lines from a file. Returns a JSON array of "
                "{line_number: content} objects. Both START_LINE and END_LINE are required — "
                "the tool will not read the entire file automatically."
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
                "Write full content to a file, creating it if it does not exist. "
                "Overwrites the entire file. Use this instead of replace_lines when "
                "rewriting a whole file or creating a new file. "
                "Returns {ok: true, lines_written: N}."
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
                        "description": "The full text content to write to the file.",
                    },
                },
                "required": ["FILE_PATH", "CONTENT"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_lines",
            "description": (
                "Replace a contiguous range of lines in a file with new content. "
                "Lines START_LINE through END_LINE (inclusive) are replaced by CONTENT. "
                "Set CONTENT to an empty string to delete the range. "
                "CONTENT may span multiple lines (use \\n). "
                "Returns the affected region after modification."
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
                        "description": "First line of the range to replace (1-indexed).",
                    },
                    "END_LINE": {
                        "type": "integer",
                        "description": "Last line of the range to replace (1-indexed, inclusive).",
                    },
                    "CONTENT": {
                        "type": "string",
                        "description": "Replacement text. Use \\n to separate multiple lines. Empty string deletes the range.",
                    },
                },
                "required": ["FILE_PATH", "START_LINE", "END_LINE", "CONTENT"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "insert_lines",
            "description": (
                "Insert new content before START_LINE in a file. "
                "Existing lines at and after START_LINE shift down. "
                "Use START_LINE = total_lines + 1 to append at the end. "
                "CONTENT may span multiple lines (use \\n). "
                "Returns the affected region after modification."
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
                        "description": "Line number before which to insert (1-indexed). Use total_lines + 1 to append.",
                    },
                    "CONTENT": {
                        "type": "string",
                        "description": "Text to insert. Use \\n to separate multiple lines.",
                    },
                },
                "required": ["FILE_PATH", "START_LINE", "CONTENT"],
            },
        },
    },
]

FILE_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "read_file",
        "write_file",
        "replace_lines",
        "insert_lines",
    }
)


def get_file_tools(disabled: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Return a name → callable mapping for file editing tools."""
    tools: dict[str, Any] = {
        "read_file": lambda FILE_PATH, START_LINE, END_LINE, **_kw: read_file(FILE_PATH, START_LINE, END_LINE),
        "write_file": lambda FILE_PATH, CONTENT, **_kw: write_file(FILE_PATH, CONTENT),
        "replace_lines": lambda FILE_PATH, START_LINE, END_LINE, CONTENT, **_kw: replace_lines(FILE_PATH, START_LINE, END_LINE, CONTENT),
        "insert_lines": lambda FILE_PATH, START_LINE, CONTENT, **_kw: insert_lines(FILE_PATH, START_LINE, CONTENT),
    }
    return {name: fn for name, fn in tools.items() if name not in disabled}
