"""Tests for make_agent.create_agent — render() and CLI."""

from __future__ import annotations

import json

import pytest
from make_agent.create_agent import render
from make_agent.parser import parse


def _minimal_spec(**overrides) -> dict:
    spec = {
        "system_prompt": "You are a test agent.",
        "tools": [
            {
                "name": "say-hello",
                "description": "Print a greeting.",
                "recipe": ["@echo hello"],
            }
        ],
    }
    spec.update(overrides)
    return spec


def _roundtrip(spec: dict):
    """Render spec to a Makefile string and parse it back."""
    return parse(render(spec))


class TestRenderSystemPrompt:
    def test_single_line(self):
        mf = _roundtrip(_minimal_spec(system_prompt="You are a bot."))
        assert mf.system_prompt == "You are a bot."

    def test_multiline(self):
        mf = _roundtrip(_minimal_spec(system_prompt="Line one.\nLine two."))
        assert "Line one." in mf.system_prompt
        assert "Line two." in mf.system_prompt

    def test_blank_line_preserved(self):
        prompt = "First paragraph.\n\nSecond paragraph."
        mf = _roundtrip(_minimal_spec(system_prompt=prompt))
        assert "\n\n" in mf.system_prompt


class TestRenderTools:
    def test_phony_targets_declared(self):
        rendered = render(_minimal_spec())
        assert ".PHONY: say-hello" in rendered

    def test_single_tool_no_params(self):
        mf = _roundtrip(_minimal_spec())
        assert len(mf.rules) == 1
        rule = mf.rules[0]
        assert rule.target == "say-hello"
        assert rule.description == "Print a greeting."
        assert rule.params == []

    def test_tool_with_params(self):
        spec = _minimal_spec(
            tools=[
                {
                    "name": "greet",
                    "description": "Greet someone.",
                    "params": [
                        {"name": "NAME", "type": "string", "description": "The name"},
                        {"name": "LOUD", "type": "boolean", "description": "Shout it"},
                    ],
                    "recipe": ["@echo $(NAME)"],
                }
            ]
        )
        mf = _roundtrip(spec)
        rule = mf.rules[0]
        assert len(rule.params) == 2
        assert rule.params[0].name == "NAME"
        assert rule.params[0].type == "string"
        assert rule.params[1].name == "LOUD"
        assert rule.params[1].type == "boolean"

    def test_multiple_tools(self):
        spec = _minimal_spec(
            tools=[
                {"name": "tool-a", "description": "A.", "recipe": ["@echo a"]},
                {"name": "tool-b", "description": "B.", "recipe": ["@echo b"]},
            ]
        )
        mf = _roundtrip(spec)
        targets = [r.target for r in mf.rules]
        assert targets == ["tool-a", "tool-b"]
        assert ".PHONY: tool-a tool-b" in render(spec)

    def test_multiline_tool_description(self):
        spec = _minimal_spec(
            tools=[
                {
                    "name": "search",
                    "description": "Search files.\nReturns matching lines.",
                    "recipe": ["@grep foo ."],
                }
            ]
        )
        mf = _roundtrip(spec)
        assert "Search files." in mf.rules[0].description
        assert "Returns matching lines." in mf.rules[0].description

    def test_recipe_tab_indented(self):
        spec = _minimal_spec(tools=[{"name": "run", "description": "Run.", "recipe": ["@echo step1", "@echo step2"]}])
        rendered = render(spec)
        assert "\t@echo step1" in rendered
        assert "\t@echo step2" in rendered

    def test_tool_without_params_key(self):
        """params key is optional — should not raise."""
        spec = _minimal_spec(tools=[{"name": "no-params", "description": "No params.", "recipe": ["@true"]}])
        mf = _roundtrip(spec)
        assert mf.rules[0].params == []

    def test_make_variable_syntax_preserved(self):
        """$(VAR) in recipe lines must survive round-trip unchanged."""
        spec = _minimal_spec(
            tools=[
                {
                    "name": "cat-file",
                    "description": "Cat a file.",
                    "params": [{"name": "PATH", "type": "string", "description": "File path"}],
                    "recipe": ["@cat $(PATH)"],
                }
            ]
        )
        rendered = render(spec)
        assert "@cat $(PATH)" in rendered


class TestRenderErrors:
    def test_missing_system_prompt_raises(self):
        with pytest.raises(KeyError):
            render({"tools": []})

    def test_missing_tools_raises(self):
        with pytest.raises(KeyError):
            render({"system_prompt": "Hi"})

    def test_missing_tool_name_raises(self):
        with pytest.raises(KeyError):
            render({"system_prompt": "Hi", "tools": [{"description": "D", "recipe": []}]})


class TestCLI:
    def _run(self, args: list[str], stdin: str | None = None) -> tuple[str, int]:
        """Run the CLI via subprocess and return (stdout, returncode)."""
        import subprocess
        import sys

        cmd = [sys.executable, "-m", "make_agent.create_agent"] + args
        result = subprocess.run(
            cmd,
            input=stdin,
            capture_output=True,
            text=True,
        )
        return result.stdout, result.returncode

    def test_reads_from_stdin(self):
        spec = json.dumps(_minimal_spec())
        out, rc = self._run([], stdin=spec)
        assert rc == 0
        assert "say-hello" in out

    def test_reads_from_spec_arg(self):
        spec = json.dumps(_minimal_spec())
        out, rc = self._run(["--spec", spec])
        assert rc == 0
        assert "say-hello" in out

    def test_reads_from_file(self, tmp_path):
        spec_file = tmp_path / "spec.json"
        spec_file.write_text(json.dumps(_minimal_spec()))
        out, rc = self._run(["--file", str(spec_file)])
        assert rc == 0
        assert "say-hello" in out

    def test_writes_to_output_file(self, tmp_path):
        out_file = tmp_path / "agent.mk"
        spec = json.dumps(_minimal_spec())
        _, rc = self._run(["--spec", spec, "-o", str(out_file)])
        assert rc == 0
        assert "say-hello" in out_file.read_text()

    def test_invalid_json_exits_nonzero(self):
        _, rc = self._run(["--spec", "not json"])
        assert rc != 0

    def test_invalid_spec_exits_nonzero(self):
        _, rc = self._run(["--spec", '{"tools": []}'])  # missing system_prompt
        assert rc != 0
