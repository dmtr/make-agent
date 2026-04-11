"""Tests for make_agent/builtin_tools/file_tools.py."""

from __future__ import annotations

from difflib import unified_diff
from pathlib import Path

from make_agent.builtin_tools.file_tools import (
    FILE_TOOL_NAMES,
    FILE_TOOL_SCHEMAS,
    get_file_tools,
    patch_file,
    read_file,
)


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _patch(old: str, new: str, context: int = 3) -> str:
    old_lines = [l + "\n" for l in old.splitlines()] if old.strip() else []
    new_lines = [l + "\n" for l in new.splitlines()] if new.strip() else []
    return "".join(unified_diff(old_lines, new_lines, fromfile="a", tofile="b", n=context))


class TestPathSandboxing:
    def test_rejects_parent_traversal(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        secret = tmp_path.parent / "secret.txt"
        secret.write_text("secret\n")
        result = read_file("../secret.txt", 1, 1)
        assert "error" in result
        assert "escapes working directory" in result

    def test_rejects_absolute_path_outside_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = read_file("/etc/hosts", 1, 1)
        assert "error" in result
        assert "escapes working directory" in result

    def test_rejects_symlink_escape(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        outside = tmp_path.parent / "outside.txt"
        outside.write_text("outside\n")
        link = tmp_path / "sneaky"
        link.symlink_to(outside)
        result = read_file("sneaky", 1, 1)
        assert "error" in result
        assert "escapes working directory" in result

    def test_allows_subdirectory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        sub = tmp_path / "sub"
        sub.mkdir()
        _write(sub / "hello.txt", "hello\n")
        result = read_file("sub/hello.txt", 1, 1)
        assert result == "hello"


class TestReadFile:
    def test_happy_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "line1\nline2\nline3\n")
        result = read_file("test.txt", 1, 3)
        assert result == "line1\nline2\nline3"

    def test_partial_range(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "a\nb\nc\nd\ne\n")
        result = read_file("test.txt", 2, 4)
        assert result == "b\nc\nd"

    def test_single_line(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "only\n")
        result = read_file("test.txt", 1, 1)
        assert result == "only"

    def test_blank_line_in_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "a\n\nc\n")
        result = read_file("test.txt", 1, 3)
        assert result == "a\n\nc"

    def test_start_exceeds_file_length(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "a\nb\n")
        result = read_file("test.txt", 5, 10)
        assert "error" in result
        assert "total_lines: 2" in result

    def test_end_exceeds_file_length(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "a\nb\nc\n")
        result = read_file("test.txt", 1, 10)
        assert "error" in result
        assert "total_lines: 3" in result

    def test_start_greater_than_end(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "a\nb\n")
        result = read_file("test.txt", 3, 1)
        assert "error" in result

    def test_start_less_than_one(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "a\n")
        result = read_file("test.txt", 0, 1)
        assert "error" in result

    def test_file_not_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = read_file("nope.txt", 1, 1)
        assert "error" in result
        assert "not found" in result

    def test_binary_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "bin.dat").write_bytes(b"\x00\x01\x02")
        result = read_file("bin.dat", 1, 1)
        assert "error" in result
        assert "binary" in result

    def test_empty_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "empty.txt", "")
        result = read_file("empty.txt", 1, 1)
        assert "error" in result
        assert "total_lines: 0" in result


class TestPatchFile:
    def test_create_new_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = patch_file("new.txt", _patch("", "hello\nworld\n"))
        assert result == "Changes accepted."
        assert _read(tmp_path / "new.txt") == "hello\nworld\n"

    def test_replace_existing_lines(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "a\nb\nc\nd\ne\n")
        result = patch_file("test.txt", _patch("a\nb\nc\nd\ne\n", "a\nBBB\nc\nDDD\ne\n"))
        assert result == "Changes accepted."
        assert _read(tmp_path / "test.txt") == "a\nBBB\nc\nDDD\ne\n"

    def test_insert_line(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "a\nb\n")
        result = patch_file("test.txt", _patch("a\nb\n", "a\nx\nb\n"))
        assert result == "Changes accepted."
        assert _read(tmp_path / "test.txt") == "a\nx\nb\n"

    def test_delete_line(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "a\nb\nc\n")
        result = patch_file("test.txt", _patch("a\nb\nc\n", "a\nc\n"))
        assert result == "Changes accepted."
        assert _read(tmp_path / "test.txt") == "a\nc\n"

    def test_create_in_subdirectory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = patch_file("sub/dir/file.txt", _patch("", "nested\n"))
        assert result == "Changes accepted."
        assert _read(tmp_path / "sub" / "dir" / "file.txt") == "nested\n"

    def test_partial_hunk_no_context(self, tmp_path, monkeypatch):
        """A diff with 0 context lines still correctly targets absolute line positions."""
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "a\nb\nc\nd\n")
        diff = _patch("a\nb\nc\nd\n", "a\nB\nC\nd\n", context=0)
        result = patch_file("test.txt", diff)
        assert result == "Changes accepted."
        assert _read(tmp_path / "test.txt") == "a\nB\nC\nd\n"

    def test_rejects_empty_diff(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "content\n")
        result = patch_file("test.txt", "")
        assert "error" in result

    def test_rejects_directory_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "adir").mkdir()
        result = patch_file("adir", _patch("", "content\n"))
        assert "error" in result

    def test_rejects_path_traversal(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = patch_file("../escape.txt", _patch("", "bad\n"))
        assert "error" in result
        assert "escapes working directory" in result

    def test_rejects_malformed_hunk_body_line(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = patch_file("test.txt", "@@ -1 +1 @@\nx malformed line\n")
        assert "error" in result
        assert "unexpected diff line" in result

    def test_rejects_stale_patch(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "a\nb\nc\n")
        diff = _patch("a\nb\nc\n", "a\nB\nc\n")
        _write(tmp_path / "test.txt", "a\nchanged\nc\n")
        result = patch_file("test.txt", diff)
        assert "error" in result
        assert "context mismatch" in result

    def test_roundtrip_read_patch(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "alpha\nbeta\ngamma\n")
        content = read_file("test.txt", 2, 2)
        assert content == "beta"
        diff = "@@ -2 +2 @@\n-beta\n+BETA\n"
        result = patch_file("test.txt", diff)
        assert result == "Changes accepted."
        assert _read(tmp_path / "test.txt") == "alpha\nBETA\ngamma\n"


class TestSchemas:
    def test_schema_count(self):
        assert len(FILE_TOOL_SCHEMAS) == 2

    def test_schema_names(self):
        names = {s["function"]["name"] for s in FILE_TOOL_SCHEMAS}
        assert names == {"read_file", "patch_file"}

    def test_tool_names_frozenset(self):
        assert FILE_TOOL_NAMES == frozenset({"read_file", "patch_file"})

    def test_get_file_tools_returns_all(self):
        tools = get_file_tools()
        assert set(tools.keys()) == {"read_file", "patch_file"}

    def test_get_file_tools_respects_disabled(self):
        tools = get_file_tools(disabled=frozenset({"read_file"}))
        assert "read_file" not in tools
        assert "patch_file" in tools
