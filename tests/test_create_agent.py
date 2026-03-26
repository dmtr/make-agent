"""Tests for make_agent.create_agent — render() and CLI."""

from __future__ import annotations

import subprocess
import sys
import yaml

import pytest
from unittest.mock import patch
from make_agent.create_agent import render, main
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
                    "recipe": ["@echo $(NAME) loud=$(LOUD)"],
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

        cmd = [sys.executable, "-m", "make_agent.create_agent"] + args
        result = subprocess.run(
            cmd,
            input=stdin,
            capture_output=True,
            text=True,
        )
        return result.stdout, result.returncode

    def test_reads_from_stdin(self):
        spec = yaml.dump(_minimal_spec())
        out, rc = self._run([], stdin=spec)
        assert rc == 0
        assert "say-hello" in out

    def test_reads_from_spec_arg(self):
        spec = yaml.dump(_minimal_spec())
        out, rc = self._run(["--spec", spec])
        assert rc == 0
        assert "say-hello" in out

    def test_reads_from_file(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text(yaml.dump(_minimal_spec()))
        out, rc = self._run(["--file", str(spec_file)])
        assert rc == 0
        assert "say-hello" in out

    def test_writes_to_output_file(self, tmp_path):
        out_file = tmp_path / "agent.mk"
        spec = yaml.dump(_minimal_spec())
        _, rc = self._run(["--spec", spec, "-o", str(out_file)])
        assert rc == 0
        assert "say-hello" in out_file.read_text()

    def test_invalid_yaml_exits_nonzero(self):
        _, rc = self._run(["--spec", "key: [unclosed"])
        assert rc != 0

    def test_invalid_spec_exits_nonzero(self):
        _, rc = self._run(["--spec", "tools: []"])  # missing system_prompt
        assert rc != 0

    def test_symlink_output_exits_nonzero(self, tmp_path):
        real_target = tmp_path / "real.mk"
        out_file = tmp_path / "agent.mk"
        out_file.symlink_to(real_target)
        spec = yaml.dump(_minimal_spec())
        _, rc = self._run(["--spec", spec, "-o", str(out_file)])
        assert rc != 0


class TestParamValidation:
    def _spec_with_broken_recipe(self, param="FILE", recipe="@pip install -r") -> dict:
        return {
            "system_prompt": "You are a test agent.",
            "tools": [
                {
                    "name": "install",
                    "description": "Install deps.",
                    "params": [{"name": param, "type": "string", "description": "A file"}],
                    "recipe": [recipe],
                }
            ],
        }

    def test_param_not_in_recipe_raises(self):
        with pytest.raises(ValueError, match="FILE"):
            render(self._spec_with_broken_recipe("FILE", "@pip install -r"))

    def test_error_message_names_tool(self):
        with pytest.raises(ValueError, match="install"):
            render(self._spec_with_broken_recipe())

    def test_error_message_shows_expected_syntax(self):
        with pytest.raises(ValueError, match=r"\$\(FILE\)"):
            render(self._spec_with_broken_recipe())

    def test_param_in_paren_ref_is_valid(self):
        spec = self._spec_with_broken_recipe("FILE", "@pip install -r $(FILE)")
        rendered = render(spec)  # should not raise
        assert "$(FILE)" in rendered

    def test_param_in_brace_ref_is_valid(self):
        spec = self._spec_with_broken_recipe("FILE", "@pip install -r ${FILE}")
        rendered = render(spec)
        assert "${FILE}" in rendered

    def test_multiple_params_all_referenced_is_valid(self):
        spec = {
            "system_prompt": "Hi.",
            "tools": [
                {
                    "name": "run",
                    "description": "Run.",
                    "params": [
                        {"name": "A", "type": "string", "description": "a"},
                        {"name": "B", "type": "string", "description": "b"},
                    ],
                    "recipe": ["@cmd $(A) $(B)"],
                }
            ],
        }
        assert render(spec)  # should not raise

    def test_content_param_via_file_var_is_valid(self):
        spec = {
            "system_prompt": "Hi.",
            "tools": [
                {
                    "name": "write",
                    "description": "Write file.",
                    "params": [{"name": "BODY", "type": "content", "description": "Content"}],
                    "recipe": ['@cat "$(BODY_FILE)" > out.txt'],
                }
            ],
        }
        rendered = render(spec)  # should not raise
        assert "BODY_FILE" in rendered

    def test_content_param_via_direct_ref_is_valid(self):
        """A content param referenced by $(NAME) directly should also pass."""
        spec = {
            "system_prompt": "Hi.",
            "tools": [
                {
                    "name": "write",
                    "description": "Write file.",
                    "params": [{"name": "BODY", "type": "content", "description": "Content"}],
                    "recipe": ['@echo "$(BODY)"'],
                }
            ],
        }
        assert render(spec)  # should not raise

    def test_content_param_not_referenced_raises(self):
        spec = {
            "system_prompt": "Hi.",
            "tools": [
                {
                    "name": "write",
                    "description": "Write file.",
                    "params": [{"name": "BODY", "type": "content", "description": "Content"}],
                    "recipe": ["@echo nothing"],
                }
            ],
        }
        with pytest.raises(ValueError, match="BODY"):
            render(spec)

    def test_content_param_error_hints_file_var(self):
        """Error message for an unreferenced content param must mention $(BODY_FILE)."""
        spec = {
            "system_prompt": "Hi.",
            "tools": [
                {
                    "name": "write",
                    "description": "Write file.",
                    "params": [{"name": "BODY", "type": "content", "description": "Content"}],
                    "recipe": ["@echo nothing"],
                }
            ],
        }
        with pytest.raises(ValueError, match=r"\$\(BODY_FILE\)"):
            render(spec)

    def test_content_param_error_recommends_file_var(self):
        spec = {
            "system_prompt": "Hi.",
            "tools": [
                {
                    "name": "write",
                    "description": "Write file.",
                    "params": [{"name": "BODY", "type": "content", "description": "Content"}],
                    "recipe": ["@echo nothing"],
                }
            ],
        }
        with pytest.raises(ValueError, match="recommended for content params"):
            render(spec)

    def test_cli_exits_nonzero_on_broken_recipe(self):
        spec = yaml.dump(self._spec_with_broken_recipe())
        result = subprocess.run(
            [sys.executable, "-m", "make_agent.create_agent", "--spec", spec],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "FILE" in result.stderr


class TestOutputSafety:
    def test_refuses_to_overwrite_symlink_output(self, tmp_path):
        out_file = tmp_path / "agent.mk"
        target = tmp_path / "target.mk"
        out_file.symlink_to(target)
        spec_yaml = yaml.dump(_minimal_spec())
        with patch("sys.argv", ["make-agent-create", "--spec", spec_yaml, "-o", str(out_file)]):
            with pytest.raises(SystemExit, match="refusing to overwrite symlink"):
                main()
