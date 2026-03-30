"""Smoke e2e tests — run the real CLI against the live API.

Skip by default; opt in with:  pytest --e2e -m e2e
"""

import subprocess
import sys

import pytest

MODEL = "anthropic/claude-haiku-4-5"
MAKEFILE = "examples/orchestra.mk"


def _run(prompt: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "make_agent.main", "run", "-f", MAKEFILE, "--model", MODEL, "--prompt", prompt],
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.e2e
def test_list_available_tools():
    result = _run("What tools are available to you? Just list their names.")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip(), "Expected non-empty output"
    assert "list_agent" in result.stdout, "Expected 'list_agents' tool to be available"


@pytest.mark.e2e
def test_list_available_agents():
    result = _run("List available agents.")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip(), "Expected non-empty output"
    assert "test-agent" in result.stdout, "Expected list of agents in output"
