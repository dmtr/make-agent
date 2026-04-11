"""Built-in file editing tools: read_file, patch_file.

These tools give every agent structured, line-level file access with
path sandboxing (all paths must resolve within the working directory).

``read_file`` returns numbered lines:

    1. First line of the file
    2. Second line of the file
    ...

``patch_file`` accepts a unified diff whose hunk body uses the same numbered
representation. The numbered prefixes are used to anchor the patch, while the
underlying file is updated without the prefixes.
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
_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?$")


def _format_numbered_lines(lines: list[str], start: int, end: int) -> str:
    """Return lines[start-1..end-1] (1-indexed, inclusive) in numbered format."""
    result = []
    for i in range(max(start, 1), min(end, len(lines)) + 1):
        result.append(f"{i}. {lines[i - 1]}")
    return "\n".join(result)


def _parse_numbered_line(content: str) -> tuple[int, str] | None:
    """Parse one numbered line, returning ``(line_num, text)`` or ``None``."""
    match = _NUMBERED_LINE_RE.match(content)
    if match is None:
        return None
    return int(match.group(1)), match.group(2)


def _parse_patch(diff: str) -> list[tuple[int, int, list[tuple[str, int, str]]]] | str:
    """Parse a unified diff over numbered lines.

    Returns ``[(old_count, new_count, entries), ...]`` where each entry is
    ``(op, line_num, text)`` and ``op`` is one of ``' '``, ``'-'``, ``'+'``.
    """
    if not diff.strip():
        return "error: DIFF must contain at least one hunk"

    raw_lines = diff.splitlines()
    hunks: list[tuple[int, int, list[tuple[str, int, str]]]] = []
    i = 0

    while i < len(raw_lines):
        line = raw_lines[i]
        if line.startswith(("diff --git ", "index ", "--- ", "+++ ")):
            i += 1
            continue

        match = _HUNK_HEADER_RE.match(line)
        if match is None:
            return f"error: invalid unified diff header: {line!r}"

        old_count = int(match.group(2) or "1")
        new_count = int(match.group(4) or "1")
        i += 1
        entries: list[tuple[str, int, str]] = []

        while i < len(raw_lines):
            body_line = raw_lines[i]
            if _HUNK_HEADER_RE.match(body_line):
                break
            if body_line.startswith(("diff --git ", "index ", "--- ", "+++ ")):
                break
            if body_line.startswith("\\ "):
                return "error: '\\ No newline at end of file' is not supported"
            if not body_line:
                return "error: invalid patch line: ''"

            op = body_line[0]
            if op not in {" ", "-", "+"}:
                return f"error: invalid patch line: {body_line!r}"

            parsed = _parse_numbered_line(body_line[1:])
            if parsed is None:
                return f"error: invalid numbered patch line: {body_line!r}"

            entries.append((op, parsed[0], parsed[1]))
            i += 1

        if not entries:
            return "error: each hunk must contain at least one patch line"

        old_entries = [entry for entry in entries if entry[0] != "+"]
        new_entries = [entry for entry in entries if entry[0] != "-"]
        if len(old_entries) != old_count:
            return f"error: hunk old-count mismatch: header says {old_count}, body has {len(old_entries)}"
        if len(new_entries) != new_count:
            return f"error: hunk new-count mismatch: header says {new_count}, body has {len(new_entries)}"

        hunks.append((old_count, new_count, entries))

    if not hunks:
        return "error: DIFF must contain at least one hunk"

    return hunks


def _validate_consecutive(nums: list[int], label: str) -> str | None:
    """Return an error if *nums* are not consecutive."""
    if not nums:
        return None
    expected = list(range(nums[0], nums[0] + len(nums)))
    if nums != expected:
        return f"error: {label} line numbers must be consecutive, got {nums}"
    return None


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


def patch_file(FILE_PATH: str, DIFF: str) -> str:
    """Patch a file using unified diff hunks over numbered ``read_file`` output."""
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

    parsed = _parse_patch(DIFF)
    if isinstance(parsed, str):
        return parsed

    replacements: list[tuple[int, int, list[str]]] = []

    for _old_count, _new_count, entries in parsed:
        old_entries = [(line_num, text) for op, line_num, text in entries if op != "+"]
        new_entries = [(line_num, text) for op, line_num, text in entries if op != "-"]
        old_nums = [line_num for line_num, _text in old_entries]
        new_nums = [line_num for line_num, _text in new_entries]

        old_error = _validate_consecutive(old_nums, "original")
        if old_error is not None:
            return old_error
        new_error = _validate_consecutive(new_nums, "updated")
        if new_error is not None:
            return new_error

        if old_entries:
            start_idx = old_nums[0] - 1
            expected = [text for _line_num, text in old_entries]
            actual = lines[start_idx : start_idx + len(expected)]
            if actual != expected:
                return (
                    f"error: patch context mismatch at line {old_nums[0]}: "
                    f"expected {expected!r}, found {actual!r}"
                )
            old_len = len(expected)
        else:
            if not new_entries:
                return "error: a hunk must modify at least one line"
            start_idx = new_nums[0] - 1
            if start_idx < 0 or start_idx > len(lines):
                return (
                    f"error: insertion point {new_nums[0]} is out of range "
                    f"for a file with {len(lines)} lines"
                )
            old_len = 0

        replacements.append((start_idx, old_len, [text for _line_num, text in new_entries]))

    for start_idx, old_len, new_texts in sorted(replacements, key=lambda item: item[0], reverse=True):
        lines[start_idx : start_idx + old_len] = new_texts

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
            "name": "patch_file",
            "description": (
                "Apply a unified diff to a file using the numbered line format "
                "returned by read_file. Hunk body lines must look like "
                "' 12. unchanged', '-13. old text', '+13. new text'. "
                "The file is updated without the numeric prefixes, and a missing "
                "file may be created from an all-added patch. "
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
                            "Unified diff over numbered read_file output. "
                            "Example: '@@ -2 +2 @@\\n-2. old\\n+2. new'."
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
