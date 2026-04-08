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


def _parse_lines_arg(lines_json: str) -> list[tuple[int, str]]:
    """Parse the LINES JSON array into a sorted list of (line_number, content) tuples."""
    try:
        items = json.loads(lines_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON in LINES: {e}")
    if not isinstance(items, list):
        raise ValueError("LINES must be a JSON array")
    result = []
    for item in items:
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError(f"each element must be a single-key object, got: {item}")
        for k, v in item.items():
            try:
                line_num = int(k)
            except ValueError:
                raise ValueError(f"line number must be an integer, got: {k}")
            result.append((line_num, str(v)))
    return sorted(result, key=lambda x: x[0])


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


def replace_lines(FILE_PATH: str, LINES: str) -> str:
    """Replace content of specified lines. Empty content deletes the line."""
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

    try:
        replacements = _parse_lines_arg(LINES)
    except ValueError as e:
        return json.dumps({"error": str(e)})

    total = len(lines)

    for line_num, _ in replacements:
        if line_num < 1 or line_num > total:
            return json.dumps({"error": f"line {line_num} out of range, file has {total} lines"})

    # Replace content, then remove lines with empty content (in reverse order)
    lines_to_delete: list[int] = []
    for line_num, content in replacements:
        if content == "":
            lines_to_delete.append(line_num)
        else:
            lines[line_num - 1] = content

    # Delete in reverse order to preserve indices
    for line_num in sorted(lines_to_delete, reverse=True):
        del lines[line_num - 1]

    _write_lines(path, lines)

    # Return context window around affected range
    if replacements:
        min_line = min(r[0] for r in replacements)
        max_line = max(r[0] for r in replacements) - len(lines_to_delete)
        ctx_start = max(1, min_line - 3)
        ctx_end = min(len(lines), max_line + 3)
        return _format_result(lines, ctx_start, ctx_end)

    return json.dumps([])


def insert_lines(FILE_PATH: str, LINES: str) -> str:
    """Insert new lines before specified positions. Existing lines shift down."""
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

    try:
        insertions = _parse_lines_arg(LINES)
    except ValueError as e:
        return json.dumps({"error": str(e)})

    total = len(lines)

    for line_num, _ in insertions:
        if line_num < 1 or line_num > total + 1:
            return json.dumps({"error": f"line {line_num} out of range, valid range is 1-{total + 1}"})

    # Insert in forward order (sorted ascending) so that each line number
    # refers to the position in the current state of the file.
    for line_num, content in insertions:
        lines.insert(line_num - 1, content)

    _write_lines(path, lines)

    # Return context window around affected range
    if insertions:
        min_line = min(r[0] for r in insertions)
        max_line = max(r[0] for r in insertions) + len(insertions) - 1
        ctx_start = max(1, min_line - 3)
        ctx_end = min(len(lines), max_line + 3)
        return _format_result(lines, ctx_start, ctx_end)

    return json.dumps([])


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
                "Replace content of specified lines in a file. "
                "LINES is a JSON array of {line_number: new_content} objects. "
                'Set content to "" to delete a line. '
                "Returns the affected region after modification."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "FILE_PATH": {
                        "type": "string",
                        "description": "File path relative to the working directory.",
                    },
                    "LINES": {
                        "type": "string",
                        "description": ('JSON array of objects, e.g. [{"5": "new line 5"}, {"6": "new line 6"}]. ' 'Use {"N": ""} to delete line N.'),
                    },
                },
                "required": ["FILE_PATH", "LINES"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "insert_lines",
            "description": (
                "Insert new lines before specified positions in a file. "
                "Existing lines shift down. Use line_number = total_lines + 1 to append. "
                "LINES is a JSON array of {line_number: content} objects. "
                "Returns the affected region after modification."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "FILE_PATH": {
                        "type": "string",
                        "description": "File path relative to the working directory.",
                    },
                    "LINES": {
                        "type": "string",
                        "description": ('JSON array of objects, e.g. [{"3": "new line before 3"}, {"4": "another line"}].'),
                    },
                },
                "required": ["FILE_PATH", "LINES"],
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
        "replace_lines": lambda FILE_PATH, LINES, **_kw: replace_lines(FILE_PATH, LINES),
        "insert_lines": lambda FILE_PATH, LINES, **_kw: insert_lines(FILE_PATH, LINES),
    }
    return {name: fn for name, fn in tools.items() if name not in disabled}
