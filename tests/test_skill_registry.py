"""Tests for SkillRegistry internals, focusing on _ast_trust_check."""

from __future__ import annotations


from make_agent.skill_registry import _ast_trust_check


# ── dangerous patterns ─────────────────────────────────────────────────────────


def test_subprocess_import_untrusted():
    is_trusted, patterns = _ast_trust_check("import subprocess\n")
    assert not is_trusted
    assert "subprocess" in patterns


def test_subprocess_from_import_untrusted():
    is_trusted, patterns = _ast_trust_check("from subprocess import run\n")
    assert not is_trusted
    assert "subprocess" in patterns


def test_socket_import_untrusted():
    is_trusted, patterns = _ast_trust_check("import socket\n")
    assert not is_trusted
    assert "socket" in patterns


def test_requests_import_untrusted():
    is_trusted, patterns = _ast_trust_check("import requests\n")
    assert not is_trusted
    assert "requests" in patterns


def test_httpx_import_untrusted():
    is_trusted, patterns = _ast_trust_check("import httpx\n")
    assert not is_trusted
    assert "httpx" in patterns


def test_urllib_from_import_untrusted():
    is_trusted, patterns = _ast_trust_check("from urllib.request import urlopen\n")
    assert not is_trusted
    assert "urllib" in patterns


def test_http_from_import_untrusted():
    is_trusted, patterns = _ast_trust_check("from http.client import HTTPConnection\n")
    assert not is_trusted
    assert "http" in patterns


def test_exec_call_untrusted():
    is_trusted, patterns = _ast_trust_check('exec("rm -rf /")\n')
    assert not is_trusted
    assert "exec" in patterns


def test_eval_call_untrusted():
    is_trusted, patterns = _ast_trust_check('eval("1+1")\n')
    assert not is_trusted
    assert "eval" in patterns


def test_dunder_import_untrusted():
    is_trusted, patterns = _ast_trust_check('__import__("os")\n')
    assert not is_trusted
    assert "__import__" in patterns


def test_os_system_call_untrusted():
    is_trusted, patterns = _ast_trust_check("import os\nos.system('ls')\n")
    assert not is_trusted
    assert "os.system" in patterns


def test_os_popen_call_untrusted():
    is_trusted, patterns = _ast_trust_check("import os\nos.popen('ls')\n")
    assert not is_trusted
    assert "os.popen" in patterns


def test_asyncio_subprocess_exec_untrusted():
    is_trusted, patterns = _ast_trust_check(
        "import asyncio\nawait asyncio.create_subprocess_exec('ls')\n"
    )
    assert not is_trusted
    assert "asyncio.create_subprocess_exec" in patterns


def test_asyncio_subprocess_shell_untrusted():
    is_trusted, patterns = _ast_trust_check(
        "import asyncio\nawait asyncio.create_subprocess_shell('ls')\n"
    )
    assert not is_trusted
    assert "asyncio.create_subprocess_shell" in patterns


def test_multiple_patterns_deduplicated():
    code = "import subprocess\nimport subprocess\n"
    is_trusted, patterns = _ast_trust_check(code)
    assert not is_trusted
    assert patterns.count("subprocess") == 1


def test_multiple_distinct_patterns():
    code = "import subprocess\nimport socket\n"
    is_trusted, patterns = _ast_trust_check(code)
    assert not is_trusted
    assert "subprocess" in patterns
    assert "socket" in patterns


# ── safe patterns ──────────────────────────────────────────────────────────────


def test_clean_code_trusted():
    code = "x = 1\nprint(x)\n"
    is_trusted, patterns = _ast_trust_check(code)
    assert is_trusted
    assert patterns == []


def test_safe_stdlib_imports_trusted():
    code = "import json\nimport re\nfrom pathlib import Path\nfrom typing import Any\n"
    is_trusted, patterns = _ast_trust_check(code)
    assert is_trusted
    assert patterns == []


def test_os_import_without_dangerous_calls_trusted():
    """import os alone is safe; only os.system/popen/exec* are flagged."""
    code = "import os\npath = os.path.join('a', 'b')\n"
    is_trusted, patterns = _ast_trust_check(code)
    assert is_trusted
    assert patterns == []


def test_target_decorator_skill_trusted():
    code = """\
from make_agent import target

@target
def read_file(path: str) -> str:
    \"\"\"Read a file.\"\"\"
    with open(path) as f:
        return f.read()
"""
    is_trusted, patterns = _ast_trust_check(code)
    assert is_trusted
    assert patterns == []


def test_aiohttp_import_untrusted():
    is_trusted, patterns = _ast_trust_check("import aiohttp\n")
    assert not is_trusted
    assert "aiohttp" in patterns


def test_syntax_error_returns_unparseable():
    is_trusted, patterns = _ast_trust_check("def (:\n")
    assert not is_trusted
    assert "unparseable" in patterns
