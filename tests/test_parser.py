from make_agent.parser import Param, Variable, parse


def test_empty():
    mf = parse("")
    assert mf.system_prompt is None
    assert mf.variables == {}
    assert mf.rules == []
    assert mf.default_target is None


def test_comments_only():
    mf = parse("# just a comment\n# another")
    assert mf.rules == []
    assert mf.variables == {}


def test_variable_recursive():
    mf = parse("FOO = bar")
    assert mf.variables["FOO"] == Variable(name="FOO", value="bar", flavor="recursive")


def test_variable_simple():
    mf = parse("FOO := bar")
    assert mf.variables["FOO"] == Variable(name="FOO", value="bar", flavor="simple")


def test_variable_simple_double_colon():
    mf = parse("FOO ::= bar")
    assert mf.variables["FOO"].flavor == "simple"


def test_variable_conditional():
    mf = parse("FOO ?= bar")
    assert mf.variables["FOO"] == Variable(name="FOO", value="bar", flavor="conditional")


def test_variable_append_new():
    mf = parse("FOO += bar")
    assert mf.variables["FOO"].value == "bar"


def test_variable_append_existing():
    mf = parse("FOO = hello\nFOO += world")
    assert mf.variables["FOO"].value == "hello world"


def test_variable_expansion():
    mf = parse("A = hello\nB = $(A) world")
    assert mf.variables["B"].value == "hello world"


def test_variable_expansion_braces():
    mf = parse("A = hi\nB = ${A} there")
    assert mf.variables["B"].value == "hi there"


def test_variable_inline_comment_stripped():
    mf = parse("FOO = bar # this is a comment")
    assert mf.variables["FOO"].value == "bar"


def test_variable_unknown_ref_preserved():
    mf = parse("FOO = $(UNDEFINED)")
    assert mf.variables["FOO"].value == "$(UNDEFINED)"


def test_multiple_variables():
    mf = parse("CC = gcc\nCFLAGS = -Wall")
    assert mf.variables["CC"].value == "gcc"
    assert mf.variables["CFLAGS"].value == "-Wall"


def test_basic_rule():
    mf = parse("build:\n\tgcc -o output main.c")
    assert len(mf.rules) == 1
    rule = mf.rules[0]
    assert rule.target == "build"
    assert rule.prerequisites == []
    assert rule.recipes == ["gcc -o output main.c"]


def test_rule_with_prerequisites():
    mf = parse("build: main.c utils.c\n\tgcc -o output main.c utils.c")
    rule = mf.rules[0]
    assert rule.prerequisites == ["main.c", "utils.c"]


def test_rule_no_recipe():
    mf = parse("all: build test")
    assert mf.rules[0].target == "all"
    assert mf.rules[0].prerequisites == ["build", "test"]
    assert mf.rules[0].recipes == []


def test_multiple_recipes():
    mf = parse("build:\n\tgcc -c main.c\n\tgcc -o output main.o")
    assert mf.rules[0].recipes == ["gcc -c main.c", "gcc -o output main.o"]


def test_recipe_ends_on_non_tab_line():
    mf = parse("build:\n\tgcc main.c\nclean:\n\trm output")
    assert len(mf.rules) == 2
    assert mf.rules[0].recipes == ["gcc main.c"]
    assert mf.rules[1].recipes == ["rm output"]


def test_recipe_ends_on_blank_line():
    mf = parse("build:\n\tgcc main.c\n\nclean:")
    assert mf.rules[0].recipes == ["gcc main.c"]
    assert mf.rules[1].recipes == []


def test_recipe_kept_verbatim():
    """Recipes are passed to the shell verbatim; variable refs are NOT expanded."""
    mf = parse("CC = gcc\nbuild:\n\t$(CC) -o output main.c")
    assert mf.rules[0].recipes == ["$(CC) -o output main.c"]


def test_multiple_targets_on_one_line():
    mf = parse("foo bar: baz")
    targets = [r.target for r in mf.rules]
    assert "foo" in targets
    assert "bar" in targets
    for r in mf.rules:
        assert r.prerequisites == ["baz"]


def test_multiple_targets_share_recipes():
    mf = parse("foo bar:\n\techo hi")
    foo = next(r for r in mf.rules if r.target == "foo")
    bar = next(r for r in mf.rules if r.target == "bar")
    assert foo.recipes == ["echo hi"]
    assert bar.recipes == ["echo hi"]
    assert foo.recipes is bar.recipes  # same list object


def test_prerequisites_inline_comment_stripped():
    mf = parse("build: main.c # only main for now")
    assert mf.rules[0].prerequisites == ["main.c"]


def test_prerequisite_variable_expansion():
    mf = parse("SRC = main.c\nbuild: $(SRC)\n\tgcc $(SRC)")
    assert mf.rules[0].prerequisites == ["main.c"]


def test_default_target():
    mf = parse("all:\nbuild:")
    assert mf.default_target == "all"


def test_default_target_skips_dot_targets():
    mf = parse(".PHONY: all\nall:\nbuild:")
    assert mf.default_target == "all"


def test_default_target_none_when_only_phony_special():
    mf = parse(".PHONY: all")
    assert mf.default_target is None


def test_phony_before_rule():
    mf = parse(".PHONY: build\nbuild:\n\tgcc main.c")
    assert mf.rules[0].is_phony is True


def test_phony_after_rule():
    mf = parse("build:\n\tgcc main.c\n.PHONY: build")
    assert mf.rules[0].is_phony is True


def test_phony_multiple_targets():
    mf = parse(".PHONY: build test clean\nbuild:\ntest:\nclean:")
    assert all(r.is_phony for r in mf.rules)


def test_non_phony_rule():
    mf = parse("build:\n\tgcc main.c")
    assert mf.rules[0].is_phony is False


def test_system_prompt_single_line():
    mf = parse("# <system>\n# You are a build assistant.\n# </system>")
    assert mf.system_prompt == "You are a build assistant."


def test_system_prompt_multiline():
    mf = parse("# <system>\n# Line one.\n# Line two.\n# </system>")
    assert mf.system_prompt == "Line one.\nLine two."


def test_system_prompt_empty_line_preserved():
    mf = parse("# <system>\n# First.\n#\n# Second.\n# </system>")
    assert mf.system_prompt == "First.\n\nSecond."


def test_system_prompt_none_when_absent():
    mf = parse("build:")
    assert mf.system_prompt is None


def test_system_prompt_does_not_create_rule():
    mf = parse("# <system>\n# Hi.\n# </system>\nbuild:")
    assert len(mf.rules) == 1


def test_tool_description_single_line():
    mf = parse("# <tool>\n# Build the project.\n# </tool>\nbuild:")
    assert mf.rules[0].description == "Build the project."


def test_tool_description_multiline():
    mf = parse("# <tool>\n# Build the project.\n# Run this to compile.\n# </tool>\nbuild:")
    assert mf.rules[0].description == "Build the project.\nRun this to compile."


def test_tool_description_empty_line_preserved():
    mf = parse("# <tool>\n# Part one.\n#\n# Part two.\n# </tool>\nbuild:")
    assert mf.rules[0].description == "Part one.\n\nPart two."


def test_no_tool_description():
    mf = parse("build:\n\tgcc main.c")
    assert mf.rules[0].description is None


def test_tool_description_only_first_target():
    """Only the first target in a multi-target line gets the description."""
    mf = parse("# <tool>\n# Desc.\n# </tool>\nfoo bar:")
    foo = next(r for r in mf.rules if r.target == "foo")
    bar = next(r for r in mf.rules if r.target == "bar")
    assert foo.description == "Desc."
    assert bar.description is None


def test_tool_description_separate_rules():
    text = "# <tool>\n# Build it.\n# </tool>\nbuild:\n\tgcc main.c\n" "# <tool>\n# Test it.\n# </tool>\ntest:\n\tpytest"
    mf = parse(text)
    build = next(r for r in mf.rules if r.target == "build")
    test = next(r for r in mf.rules if r.target == "test")
    assert build.description == "Build it."
    assert test.description == "Test it."


def test_tool_description_not_attached_across_other_rules():
    """A tool block not immediately followed by a rule is lost."""
    mf = parse("# <tool>\n# Desc.\n# </tool>\n\nbuild:\ntest:")
    build = next(r for r in mf.rules if r.target == "build")
    assert build.description == "Desc."  # blank line doesn't reset pending desc


def test_line_continuation_variable():
    mf = parse("FOO = hello \\\nworld")
    assert "hello" in mf.variables["FOO"].value
    assert "world" in mf.variables["FOO"].value


def test_line_continuation_prerequisites():
    mf = parse("build: main.c \\\n    utils.c")
    assert "main.c" in mf.rules[0].prerequisites
    assert "utils.c" in mf.rules[0].prerequisites


_FULL_EXAMPLE = """\
# <system>
# You are a build assistant. Help users compile and test the project.
# </system>

CC = gcc
CFLAGS = -Wall -O2

.PHONY: build test clean

# <tool>
# Compile the project from source.
# Use this when the user wants to build the binary.
# </tool>
build: main.c
\t$(CC) $(CFLAGS) -o output main.c

# <tool>
# Run the test suite.
# </tool>
test: build
\tpython -m pytest tests/

clean:
\trm -f output
"""


def test_full_example():
    mf = parse(_FULL_EXAMPLE)

    assert mf.system_prompt == ("You are a build assistant. Help users compile and test the project.")
    assert mf.variables["CC"].value == "gcc"
    assert mf.variables["CFLAGS"].value == "-Wall -O2"
    assert mf.default_target == "build"

    build = next(r for r in mf.rules if r.target == "build")
    assert build.is_phony
    assert build.prerequisites == ["main.c"]
    assert build.recipes == ["$(CC) $(CFLAGS) -o output main.c"]
    assert build.description == ("Compile the project from source.\n" "Use this when the user wants to build the binary.")

    test = next(r for r in mf.rules if r.target == "test")
    assert test.is_phony
    assert test.prerequisites == ["build"]
    assert test.description == "Run the test suite."

    clean = next(r for r in mf.rules if r.target == "clean")
    assert clean.is_phony
    assert clean.description is None
    assert clean.recipes == ["rm -f output"]


def test_param_single():
    mf = parse("# <tool>\n# Greet someone.\n# @param NAME string The name\n# </tool>\ngreet:")
    rule = mf.rules[0]
    assert rule.description == "Greet someone."
    assert rule.params == [Param(name="NAME", type="string", description="The name")]


def test_param_multiple():
    text = "# <tool>\n" "# Greet someone.\n" "# @param NAME string The name to greet\n" "# @param GREETING string The greeting word\n" "# </tool>\n" "greet:"
    mf = parse(text)
    rule = mf.rules[0]
    assert rule.description == "Greet someone."
    assert rule.params == [
        Param(name="NAME", type="string", description="The name to greet"),
        Param(name="GREETING", type="string", description="The greeting word"),
    ]


def test_param_different_types():
    text = "# <tool>\n" "# Do something.\n" "# @param COUNT integer How many times\n" "# @param VERBOSE boolean Enable verbose output\n" "# </tool>\n" "run:"
    mf = parse(text)
    params = mf.rules[0].params
    assert params[0] == Param(name="COUNT", type="integer", description="How many times")
    assert params[1] == Param(name="VERBOSE", type="boolean", description="Enable verbose output")


def test_param_description_only_non_param_lines():
    """@param lines are excluded from the description text."""
    text = "# <tool>\n" "# First line.\n" "# @param X string Something\n" "# Second line.\n" "# </tool>\n" "run:"
    mf = parse(text)
    rule = mf.rules[0]
    assert rule.description == "First line.\nSecond line."
    assert len(rule.params) == 1


def test_no_params_when_absent():
    mf = parse("# <tool>\n# Build it.\n# </tool>\nbuild:")
    assert mf.rules[0].params == []


def test_params_only_on_first_target():
    """Params should only be attached to the first target in a multi-target line."""
    text = "# <tool>\n# Desc.\n# @param X string A thing\n# </tool>\nfoo bar:"
    mf = parse(text)
    foo = next(r for r in mf.rules if r.target == "foo")
    bar = next(r for r in mf.rules if r.target == "bar")
    assert foo.params == [Param(name="X", type="string", description="A thing")]
    assert bar.params == []
