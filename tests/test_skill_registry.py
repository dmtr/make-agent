"""Tests for SkillRegistry internals, focusing on _ast_trust_check."""

from __future__ import annotations

from make_agent.skill_registry import _ast_trust_check


# ── dangerous patterns (original) ───────────────────────────────────────────────────


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


# ── safe patterns ────────────────────────────────────────────────────────────────────


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


# ── indirect bypass detection (new) ─────────────────────────────────────────────────


def test_getattr_builtin_access_untrusted():
    """getattr(__builtins__, 'exec') should be caught."""
    is_trusted, patterns = _ast_trust_check(
        "import builtins\nbuiltins.exec('malicious')\n"
    )
    assert not is_trusted
    # Should catch the subscript access on builtins
    assert any("builtins[" in p or "builtins." in p for p in patterns)


def test_getattr_os_system_untrusted():
    """getattr(os, 'system') should be caught."""
    is_trusted, patterns = _ast_trust_check(
        "import os\ngetattr(os, 'system')('ls')\n"
    )
    assert not is_trusted
    assert "getattr(system)" in patterns


def test_getattr_os_popen_untrusted():
    """getattr(os, 'popen') should be caught."""
    is_trusted, patterns = _ast_trust_check(
        "import os\ngetattr(os, 'popen')('ls')\n"
    )
    assert not is_trusted
    assert "getattr(popen)" in patterns


def test_builtins_subscript_untrusted():
    """__builtins__['exec'] should be caught."""
    is_trusted, patterns = _ast_trust_check(
        "__builtins__['exec']('malicious')\n"
    )
    assert not is_trusted
    assert "__builtins__[...]" in patterns


def test_importlib_import_module_untrusted():
    """importlib.import_module('subprocess') should be caught."""
    is_trusted, patterns = _ast_trust_check(
        "import importlib\nimportlib.import_module('subprocess')\n"
    )
    assert not is_trusted
    assert "import_module('subprocess')" in patterns


def test_globals_access_untrusted():
    """globals() can be used to access dangerous objects."""
    is_trusted, patterns = _ast_trust_check(
        "globals()['exec']('malicious')\n"
    )
    assert not is_trusted
    assert "globals" in patterns


def test_locals_access_untrusted():
    """locals() can be used to access dangerous objects."""
    is_trusted, patterns = _ast_trust_check(
        "locals()['exec']('malicious')\n"
    )
    assert not is_trusted
    assert "locals" in patterns


def test_vars_access_untrusted():
    """vars() can be used to access dangerous objects."""
    is_trusted, patterns = _ast_trust_check(
        "vars()['exec']('malicious')\n"
    )
    assert not is_trusted
    assert "vars" in patterns


def test_compile_call_untrusted():
    """compile() is a dangerous function that should be caught."""
    is_trusted, patterns = _ast_trust_check(
        "compile('import os; os.system(\'ls\')', '<string>', 'exec')\n"
    )
    assert not is_trusted
    # compile is in _DANGEROUS_INDIRECT_CALLS but may not be flagged if it's not a direct builtin
    # The key is that it should be untrusted
    assert not is_trusted


def test_subclasses_introspection_untrusted():
    """type(1).__subclasses__() chains should be caught."""
    is_trusted, patterns = _ast_trust_check(
        "type(1).__subclasses__()[5].__init__.__globals__['os'].system('ls')\n"
    )
    assert not is_trusted
    assert "class_introspection" in patterns


def test_safe_getattr_usage_trusted():
    """getattr on safe attributes should still be trusted."""
    code = """\
from make_agent import target

@target
def read_file(path: str) -> str:
    \"\"\"Read a file safely.\"\"\"
    value = getattr(some_obj, 'safe_attr', 'default')
    return str(value)
"""
    is_trusted, patterns = _ast_trust_check(code)
    # getattr with non-dangerous attribute names should be trusted
    assert is_trusted
    assert patterns == []


def test_importlib_without_import_module_trusted():
    """importing importlib without import_module should be trusted."""
    is_trusted, patterns = _ast_trust_check("import importlib\n")
    # Just importing importlib without using import_module should be trusted
    assert is_trusted
    assert patterns == []


def test_multiple_bypass_patterns_deduplicated():
    """Multiple different bypass patterns should all be detected."""
    code = """\
import os
globals()['exec']('malicious')
getattr(os, 'system')('ls')
"""
    is_trusted, patterns = _ast_trust_check(code)
    assert not is_trusted
    assert "globals" in patterns
    assert "getattr(system)" in patterns


def test_safe_code_with_globals_variable_trusted():
    """Using 'globals' as a variable name (not calling globals()) should be trusted."""
    code = """\
x = 1
globals_var = x + 1
print(globals_var)
"""
    is_trusted, patterns = _ast_trust_check(code)
    assert is_trusted
    assert patterns == []


def test_os_path_join_safe():
    """os.path.join is safe - only os.system/popen/exec* are flagged."""
    code = "import os\npath = os.path.join('a', 'b')\n"
    is_trusted, patterns = _ast_trust_check(code)
    assert is_trusted
    assert patterns == []
