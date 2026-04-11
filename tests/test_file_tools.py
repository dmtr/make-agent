"""Tests for make_agent/builtin_tools/file_tools.py."""

from __future__ import annotations

from pathlib import Path

from make_agent.builtin_tools.file_tools import (
    FILE_TOOL_NAMES,
    FILE_TOOL_SCHEMAS,
    get_file_tools,
    read_file,
    write_file,
)


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


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
        assert result == "1. hello"


class TestReadFile:
    def test_happy_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "line1\nline2\nline3\n")
        result = read_file("test.txt", 1, 3)
        assert result == "1. line1\n2. line2\n3. line3"

    def test_partial_range(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "a\nb\nc\nd\ne\n")
        result = read_file("test.txt", 2, 4)
        assert result == "2. b\n3. c\n4. d"

    def test_single_line(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "only\n")
        result = read_file("test.txt", 1, 1)
        assert result == "1. only"

    def test_blank_line_in_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "a\n\nc\n")
        result = read_file("test.txt", 1, 3)
        assert result == "1. a\n2. \n3. c"

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


class TestWriteFile:
    def test_create_new_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = write_file("new.txt", "1. hello\n2. world")
        assert result == "Changes accepted."
        assert _read(tmp_path / "new.txt") == "hello\nworld\n"

    def test_partial_update(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "a\nb\nc\nd\ne\n")
        result = write_file("test.txt", "2. BBB\n4. DDD")
        assert result == "Changes accepted."
        assert _read(tmp_path / "test.txt") == "a\nBBB\nc\nDDD\ne\n"

    def test_append_line(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "a\nb\n")
        result = write_file("test.txt", "3. c")
        assert result == "Changes accepted."
        assert _read(tmp_path / "test.txt") == "a\nb\nc\n"

    def test_overwrite_whole_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "old\n")
        result = write_file("test.txt", "1. new")
        assert result == "Changes accepted."
        assert _read(tmp_path / "test.txt") == "new\n"

    def test_create_in_subdirectory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = write_file("sub/dir/file.txt", "1. nested")
        assert result == "Changes accepted."
        assert _read(tmp_path / "sub" / "dir" / "file.txt") == "nested\n"

    def test_empty_content_creates_empty_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = write_file("empty.txt", "")
        assert result == "Changes accepted."
        assert _read(tmp_path / "empty.txt") == ""

    def test_empty_content_truncates_existing_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "content\n")
        result = write_file("test.txt", "")
        assert result == "Changes accepted."
        assert _read(tmp_path / "test.txt") == ""

    def test_rejects_directory_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "adir").mkdir()
        result = write_file("adir", "1. content")
        assert "error" in result

    def test_rejects_path_traversal(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = write_file("../escape.txt", "1. bad")
        assert "error" in result
        assert "escapes working directory" in result

    def test_invalid_line_format(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = write_file("test.txt", "no numbers here")
        assert "error" in result
        assert "invalid line format" in result

    def test_roundtrip_read_write(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "alpha\nbeta\ngamma\n")
        content = read_file("test.txt", 2, 2)
        assert content == "2. beta"
        modified = content.replace("beta", "BETA")
        result = write_file("test.txt", modified)
        assert result == "Changes accepted."
        assert _read(tmp_path / "test.txt") == "alpha\nBETA\ngamma\n"


class TestSchemas:
    def test_schema_count(self):
        assert len(FILE_TOOL_SCHEMAS) == 2

    def test_schema_names(self):
        names = {s["function"]["name"] for s in FILE_TOOL_SCHEMAS}
        assert names == {"read_file", "write_file"}

    def test_tool_names_frozenset(self):
        assert FILE_TOOL_NAMES == frozenset({"read_file", "write_file"})

    def test_get_file_tools_returns_all(self):
        tools = get_file_tools()
        assert set(tools.keys()) == {"read_file", "write_file"}

    def test_get_file_tools_respects_disabled(self):
        tools = get_file_tools(disabled=frozenset({"read_file"}))
        assert "read_file" not in tools
        assert "write_file" in tools
