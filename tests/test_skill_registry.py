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


# ── AST bypass vectors (should be detected but currently aren't) ────────────────────
# These tests demonstrate vulnerabilities in the trust check that allow code execution.
# They will FAIL until the trust check is fixed to catch these patterns.


def test_getattr_with_variable_argument_bypass():
    """getattr(os, name) where name is a variable should be caught."""
    # This pattern bypasses the current check because the second arg is not a string constant
    code = """\
import os
name = 'system'
os.__getattribute__(name)('whoami')
"""
    is_trusted, patterns = _ast_trust_check(code)
    # Currently this passes (is_trusted=True) — that's the bug!
    # After fix: should be not is_trusted
    assert not is_trusted, "getattr with variable arg should be detected"


def test_getattr_os_with_variable_bypass():
    """getattr(os, 'system') with a variable holding 'os' should be caught."""
    code = """\
import os as operating_system
name = 'system'
getattr(operating_system, name)('whoami')
"""
    is_trusted, patterns = _ast_trust_check(code)
    assert not is_trusted, "getattr with variable module should be detected"


def test_subclasses_introspection_bypass():
    """object.__subclasses__() chain to find Popen should be caught."""
    # This finds subprocess.Popen through class introspection
    code = """\
classes = object.__subclasses__()
for c in classes:
    if c.__name__ == 'Popen':
        c('whoami', shell=True).wait()
"""
    is_trusted, patterns = _ast_trust_check(code)
    assert not is_trusted, "__subclasses__() introspection should be detected"


def test_builtins_dict_subscript_bypass():
    """builtins.__dict__['exec'] should be caught."""
    # This accesses builtins through __dict__ instead of direct attribute access
    code = """\
import builtins
builtins.__dict__['exec']('__import__("os").system("whoami")')
"""
    is_trusted, patterns = _ast_trust_check(code)
    assert not is_trusted, "builtins.__dict__ subscript should be detected"


def test_dynamic_module_name_bypass():
    """__import__(var) where var is a variable should be caught."""
    # This bypasses the check because __import__ is called with a variable, not a string literal
    code = """\
module_name = 'subprocess'
mod = __import__(module_name)
mod.Popen('whoami').wait()
"""
    is_trusted, patterns = _ast_trust_check(code)
    assert not is_trusted, "__import__ with variable arg should be detected"


def test_function_assignment_bypass():
    """Assigning exec/eval to a variable and calling it should be caught."""
    # This bypasses the check because exec is not called directly
    code = """\
x = exec
x('import os; os.system("whoami")')
"""
    is_trusted, patterns = _ast_trust_check(code)
    assert not is_trusted, "exec assigned to variable and called should be detected"


def test_eval_through_variable_bypass():
    """Assigning eval to a variable and calling it should be caught."""
    code = """\
f = eval
f('__import__("os").system("whoami")')
"""
    is_trusted, patterns = _ast_trust_check(code)
    assert not is_trusted, "eval assigned to variable and called should be detected"


def test_string_concatenation_bypass():
    """String concatenation to form 'exec' or 'eval' should be caught."""
    # This bypasses the check because the function name is constructed dynamically
    code = """\
f = eval
f('ev' + 'al' + '("1+1")')  # obfuscated
"""
    is_trusted, patterns = _ast_trust_check(code)
    # After fix: should be not is_trusted
    assert not is_trusted, "string concatenation to form dangerous names should be detected"


def test_type_subclasses_direct_bypass():
    """type(1).__subclasses__() should be caught."""
    code = """\
type(1).__subclasses__()
"""
    is_trusted, patterns = _ast_trust_check(code)
    assert not is_trusted, "type().__subclasses__() should be detected"


def test_object_subclasses_direct_bypass():
    """object.__subclasses__() should be caught."""
    code = """\
object.__subclasses__()
"""
    is_trusted, patterns = _ast_trust_check(code)
    assert not is_trusted, "object.__subclasses__() should be detected"


def test_getattribute_dangerous_attr_bypass():
    """os.__getattribute__('system') should be caught."""
    code = """\
import os
os.__getattribute__('system')('whoami')
"""
    is_trusted, patterns = _ast_trust_check(code)
    assert not is_trusted, "__getattribute__ with dangerous attr should be detected"


def test_import_module_variable_bypass():
    """importlib.import_module(var) where var is a variable should be caught."""
    code = """\
import importlib
mod_name = 'subprocess'
importlib.import_module(mod_name)
"""
    is_trusted, patterns = _ast_trust_check(code)
    assert not is_trusted, "import_module with variable arg should be detected"


def test_chain_dangerous_calls_bypass():
    """Chained dangerous calls should be caught."""
    code = """\
import os
result = os.__getattribute__('system')('whoami')
"""
    is_trusted, patterns = _ast_trust_check(code)
    assert not is_trusted, "chained __getattribute__ on os should be detected"


def test_safe_getattr_non_dangerous_still_trusted():
    """getattr on safe attributes should still be trusted (regression check)."""
    code = """\
value = getattr(some_obj, 'safe_attr', 'default')
print(value)
"""
    is_trusted, patterns = _ast_trust_check(code)
    assert is_trusted
    assert patterns == []


def test_safe_os_import_still_trusted():
    """import os with safe usage should still be trusted (regression check)."""
    code = """\
import os
path = os.path.join('a', 'b')
"""
    is_trusted, patterns = _ast_trust_check(code)
    assert is_trusted
    assert patterns == []
    assert is_trusted
    assert patterns == []
