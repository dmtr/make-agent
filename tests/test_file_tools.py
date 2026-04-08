"""Tests for make_agent/file_tools.py."""

from __future__ import annotations

import json
from pathlib import Path

from make_agent.builtin_tools.file_tools import (
    FILE_TOOL_NAMES,
    FILE_TOOL_SCHEMAS,
    get_file_tools,
    insert_lines,
    read_file,
    replace_lines,
    write_file,
)


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _result(output: str):
    return json.loads(output)


class TestPathSandboxing:
    def test_rejects_parent_traversal(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        secret = tmp_path.parent / "secret.txt"
        secret.write_text("secret\n")
        result = _result(read_file("../secret.txt", 1, 1))
        assert "error" in result
        assert "escapes working directory" in result["error"]

    def test_rejects_absolute_path_outside_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _result(read_file("/etc/hosts", 1, 1))
        assert "error" in result
        assert "escapes working directory" in result["error"]

    def test_rejects_symlink_escape(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        outside = tmp_path.parent / "outside.txt"
        outside.write_text("outside\n")
        link = tmp_path / "sneaky"
        link.symlink_to(outside)
        result = _result(read_file("sneaky", 1, 1))
        assert "error" in result
        assert "escapes working directory" in result["error"]

    def test_allows_subdirectory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        sub = tmp_path / "sub"
        sub.mkdir()
        f = sub / "hello.txt"
        _write(f, "hello\n")
        result = _result(read_file("sub/hello.txt", 1, 1))
        assert isinstance(result, list)
        assert result[0]["1"] == "hello"


class TestReadFile:
    def test_happy_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "line1\nline2\nline3\n")
        result = _result(read_file("test.txt", 1, 3))
        assert len(result) == 3
        assert result[0] == {"1": "line1"}
        assert result[1] == {"2": "line2"}
        assert result[2] == {"3": "line3"}

    def test_partial_range(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "a\nb\nc\nd\ne\n")
        result = _result(read_file("test.txt", 2, 4))
        assert len(result) == 3
        assert result[0] == {"2": "b"}
        assert result[2] == {"4": "d"}

    def test_single_line(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "only\n")
        result = _result(read_file("test.txt", 1, 1))
        assert result == [{"1": "only"}]

    def test_start_exceeds_file_length(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "a\nb\n")
        result = _result(read_file("test.txt", 5, 10))
        assert "error" in result
        assert result["total_lines"] == 2

    def test_end_exceeds_file_length(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "a\nb\nc\n")
        result = _result(read_file("test.txt", 1, 10))
        assert "error" in result
        assert result["total_lines"] == 3

    def test_start_greater_than_end(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "a\nb\n")
        result = _result(read_file("test.txt", 3, 1))
        assert "error" in result

    def test_start_less_than_one(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "a\n")
        result = _result(read_file("test.txt", 0, 1))
        assert "error" in result

    def test_file_not_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _result(read_file("nope.txt", 1, 1))
        assert "error" in result
        assert "not found" in result["error"]

    def test_binary_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "bin.dat").write_bytes(b"\x00\x01\x02")
        result = _result(read_file("bin.dat", 1, 1))
        assert "error" in result
        assert "binary" in result["error"]

    def test_empty_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "empty.txt", "")
        result = _result(read_file("empty.txt", 1, 1))
        assert "error" in result
        assert result["total_lines"] == 0


class TestReplaceLines:
    def test_single_line_replace(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "aaa\nbbb\nccc\n")
        result = _result(replace_lines("test.txt", 2, 2, "BBB"))
        assert isinstance(result, list)
        content = _read(tmp_path / "test.txt")
        assert content == "aaa\nBBB\nccc\n"

    def test_replace_range_same_size(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "a\nb\nc\nd\ne\n")
        replace_lines("test.txt", 2, 3, "B\nC")
        content = _read(tmp_path / "test.txt")
        assert content == "a\nB\nC\nd\ne\n"

    def test_replace_range_expand(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "a\nb\nc\n")
        replace_lines("test.txt", 2, 2, "B\nB2\nB3")
        content = _read(tmp_path / "test.txt")
        assert content == "a\nB\nB2\nB3\nc\n"

    def test_replace_range_shrink(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "a\nb\nc\nd\ne\n")
        replace_lines("test.txt", 2, 4, "X")
        content = _read(tmp_path / "test.txt")
        assert content == "a\nX\ne\n"

    def test_delete_range(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "a\nb\nc\nd\n")
        replace_lines("test.txt", 2, 3, "")
        content = _read(tmp_path / "test.txt")
        assert content == "a\nd\n"

    def test_content_with_trailing_newline(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "a\nb\nc\n")
        replace_lines("test.txt", 2, 2, "BBB\n")
        content = _read(tmp_path / "test.txt")
        assert content == "a\nBBB\nc\n"

    def test_out_of_range_start(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "a\nb\n")
        result = _result(replace_lines("test.txt", 5, 5, "x"))
        assert "error" in result

    def test_end_before_start(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "a\nb\nc\n")
        result = _result(replace_lines("test.txt", 3, 2, "x"))
        assert "error" in result

    def test_returns_context_window(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        lines = [f"line{i}" for i in range(1, 11)]
        _write(tmp_path / "test.txt", "\n".join(lines) + "\n")
        result = _result(replace_lines("test.txt", 5, 5, "REPLACED"))
        # Context is ±3 around line 5, so lines 2-8
        assert isinstance(result, list)
        line_nums = [int(list(r.keys())[0]) for r in result]
        assert min(line_nums) == 2
        assert max(line_nums) == 8

    def test_file_not_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _result(replace_lines("nope.txt", 1, 1, "x"))
        assert "error" in result


class TestInsertLines:
    def test_insert_at_beginning(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "b\nc\n")
        insert_lines("test.txt", json.dumps([{"1": "a"}]))
        content = _read(tmp_path / "test.txt")
        assert content == "a\nb\nc\n"

    def test_insert_in_middle(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "a\nc\n")
        insert_lines("test.txt", json.dumps([{"2": "b"}]))
        content = _read(tmp_path / "test.txt")
        assert content == "a\nb\nc\n"

    def test_append_at_end(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "a\nb\n")
        insert_lines("test.txt", json.dumps([{"3": "c"}]))
        content = _read(tmp_path / "test.txt")
        assert content == "a\nb\nc\n"

    def test_insert_multiple_lines(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "a\nd\n")
        insert_lines("test.txt", json.dumps([{"2": "b"}, {"3": "c"}]))
        content = _read(tmp_path / "test.txt")
        assert content == "a\nb\nc\nd\n"

    def test_shift_existing_lines(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "1\n2\n3\n")
        insert_lines("test.txt", json.dumps([{"2": "NEW"}]))
        content = _read(tmp_path / "test.txt")
        assert content == "1\nNEW\n2\n3\n"

    def test_out_of_range(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "a\nb\n")
        result = _result(insert_lines("test.txt", json.dumps([{"5": "x"}])))
        assert "error" in result

    def test_insert_into_empty_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "")
        insert_lines("test.txt", json.dumps([{"1": "first"}]))
        content = _read(tmp_path / "test.txt")
        assert content == "first\n"

    def test_returns_context_window(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        lines = [f"line{i}" for i in range(1, 11)]
        _write(tmp_path / "test.txt", "\n".join(lines) + "\n")
        result = _result(insert_lines("test.txt", json.dumps([{"5": "NEW"}])))
        assert isinstance(result, list)
        # Should contain lines around the insertion point
        contents = [list(r.values())[0] for r in result]
        assert "NEW" in contents

    def test_file_not_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _result(insert_lines("nope.txt", json.dumps([{"1": "x"}])))
        assert "error" in result


class TestWriteFile:
    def test_create_new_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _result(write_file("new.txt", "hello\nworld\n"))
        assert result["ok"] is True
        assert result["lines_written"] == 2
        assert _read(tmp_path / "new.txt") == "hello\nworld\n"

    def test_overwrite_existing_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "test.txt", "old content\n")
        result = _result(write_file("test.txt", "new content\n"))
        assert result["ok"] is True
        assert _read(tmp_path / "test.txt") == "new content\n"

    def test_create_in_subdirectory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _result(write_file("sub/dir/file.txt", "nested\n"))
        assert result["ok"] is True
        assert _read(tmp_path / "sub" / "dir" / "file.txt") == "nested\n"

    def test_empty_content(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _result(write_file("empty.txt", ""))
        assert result["ok"] is True
        assert result["lines_written"] == 0
        assert _read(tmp_path / "empty.txt") == ""

    def test_rejects_directory_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "adir").mkdir()
        result = _result(write_file("adir", "content"))
        assert "error" in result

    def test_rejects_path_traversal(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _result(write_file("../escape.txt", "bad"))
        assert "error" in result
        assert "escapes working directory" in result["error"]

    def test_content_without_trailing_newline(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _result(write_file("notrail.txt", "no newline"))
        assert result["ok"] is True
        assert result["lines_written"] == 1
        assert _read(tmp_path / "notrail.txt") == "no newline"


class TestSchemas:
    def test_schema_count(self):
        assert len(FILE_TOOL_SCHEMAS) == 4

    def test_schema_names(self):
        names = {s["function"]["name"] for s in FILE_TOOL_SCHEMAS}
        assert names == {"read_file", "write_file", "replace_lines", "insert_lines"}

    def test_tool_names_frozenset(self):
        assert FILE_TOOL_NAMES == frozenset({"read_file", "write_file", "replace_lines", "insert_lines"})

    def test_get_file_tools_returns_all(self):
        tools = get_file_tools()
        assert set(tools.keys()) == {"read_file", "write_file", "replace_lines", "insert_lines"}

    def test_get_file_tools_respects_disabled(self):
        tools = get_file_tools(disabled=frozenset({"read_file"}))
        assert "read_file" not in tools
        assert "replace_lines" in tools
