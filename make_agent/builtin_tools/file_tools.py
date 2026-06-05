"""File editing tools: write_file, edit_file."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _resolve_safe(path: str, base_dir: Path) -> Path | str:
    """Resolve *path* relative to *base_dir*, rejecting directory traversal.

    Returns the resolved :class:`Path` on success, or an error string if the
    resolved path escapes *base_dir*.
    """
    resolved = (base_dir / path).resolve()
    try:
        resolved.relative_to(base_dir.resolve())
    except ValueError:
        return f"Error: path '{path}' escapes the working directory '{base_dir.resolve()}'."
    return resolved


def write_file(path: str, content: str, base_dir: Path) -> str:
    """Write *content* to *path*, creating or overwriting the file.

    :param path: Path to the file, relative to *base_dir*.
    :param content: Full content to write.
    :param base_dir: Sandbox root directory.
    """
    result = _resolve_safe(path, base_dir)
    if isinstance(result, str):
        return result
    resolved = result
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to '{path}'."
    except PermissionError:
        return f"Error: permission denied writing '{path}'."
    except OSError as e:
        return f"Error: could not write '{path}': {e}"


def edit_file(path: str, old_text: str, new_text: str, base_dir: Path) -> str:
    """Replace the first occurrence of *old_text* with *new_text* in *path*.

    :param path: Path to the file, relative to *base_dir*.
    :param old_text: Exact text to find.
    :param new_text: Replacement text.
    :param base_dir: Sandbox root directory.
    """
    result = _resolve_safe(path, base_dir)
    if isinstance(result, str):
        return result
    resolved = result
    if not resolved.is_file():
        return f"Error: file '{path}' does not exist."
    try:
        content = resolved.read_text(encoding="utf-8")
    except PermissionError:
        return f"Error: permission denied reading '{path}'."
    except OSError as e:
        return f"Error: could not read '{path}': {e}"
    if old_text not in content:
        return f"Error: text not found in '{path}'."
    try:
        resolved.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Successfully replaced text in '{path}'."
    except PermissionError:
        return f"Error: permission denied writing '{path}'."
    except OSError as e:
        return f"Error: could not write '{path}': {e}"


FILE_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write content to a file, creating it if it does not exist or overwriting it if it does. "
                "The path must be relative to the working directory. "
                "Use this to create new files or fully replace the contents of an existing file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file (e.g. 'src/main.py').",
                    },
                    "content": {
                        "type": "string",
                        "description": "The full content to write to the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Find and replace the first occurrence of old_text with new_text in a file. "
                "The path must be relative to the working directory. "
                "Use this for surgical edits when you know the exact text to replace. "
                "Prefer this over write_file when only a small part of the file needs to change."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file (e.g. 'src/main.py').",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "The exact text to find and replace. Must uniquely identify the location.",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "The replacement text.",
                    },
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
]
