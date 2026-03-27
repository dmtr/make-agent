define SYSTEM_PROMPT
You are a codebase archaeology assistant for Python projects.
You help developers understand unfamiliar repos, find tech debt, and prepare for refactoring.
When exploring, start broad (show-structure, show-dependencies) then drill down into specific
files or patterns. Combine tools iteratively to build a complete picture before answering.
Always explain your findings in plain language with concrete file references.
endef

.PHONY: show-structure read-file show-dependencies search-symbol \
        git-log-file find-stale-files find-untested find-todos \
        run-tests list-definitions

# <tool>
# Show the directory tree of the project.
# @param DEPTH integer How many directory levels deep to show (default: 3)
# </tool>
show-structure:
	@find . -maxdepth $(if $(DEPTH),$(DEPTH),3) \
	  -not -path '*/.git/*' -not -path '*/__pycache__/*' \
	  -not -path '*/node_modules/*' -not -path '*/.venv/*' \
	  -not -path '*/dist/*' -not -path '*/build/*' \
	  | sort

# <tool>
# Read the full contents of a file.
# @param PATH string Relative path to the file (e.g. src/auth/login.py)
# </tool>
read-file:
	@cat $(PATH)

# <tool>
# Show project dependencies from pyproject.toml, setup.py, or requirements.txt.
# </tool>
show-dependencies:
	@if [ -f pyproject.toml ]; then \
	  echo "=== pyproject.toml ==="; cat pyproject.toml; \
	elif [ -f requirements.txt ]; then \
	  echo "=== requirements.txt ==="; cat requirements.txt; \
	elif [ -f setup.py ]; then \
	  echo "=== setup.py ==="; cat setup.py; \
	else \
	  echo "No dependency file found."; \
	fi

# <tool>
# Search for a symbol, function name, class name, or text pattern across all Python files.
# Returns file path and line number for each match.
# @param PATTERN string The search pattern (regex supported)
# </tool>
search-symbol:
	@grep -rn --include="*.py" "$(PATTERN)" . \
	  --exclude-dir=__pycache__ --exclude-dir=.venv --exclude-dir=.git \
	  || echo "No matches found for: $(PATTERN)"

# <tool>
# Show the git commit history for a specific file to understand how it evolved.
# @param PATH string Relative path to the file
# @param LIMIT integer Max number of commits to show (default: 20)
# </tool>
git-log-file:
	@git log --no-pager \
	  --pretty=format:"%h %ad %an — %s" --date=short \
	  -n $(if $(LIMIT),$(LIMIT),20) \
	  -- $(PATH)

# <tool>
# Find Python source files that haven't been modified in a while — potential stale code.
# @param DAYS integer Files not touched in this many days (default: 365)
# </tool>
find-stale-files:
	@git log --no-pager --name-only --pretty=format: \
	  --since="$(if $(DAYS),$(DAYS),365) days ago" \
	  -- '*.py' \
	  | sort -u > /tmp/recently_changed.txt; \
	  find . -name "*.py" \
	    -not -path '*/.venv/*' -not -path '*/__pycache__/*' -not -path '*/.git/*' \
	  | sed 's|^\./||' \
	  | sort > /tmp/all_py.txt; \
	  echo "=== Python files NOT touched in $(if $(DAYS),$(DAYS),365)+ days ==="; \
	  comm -23 /tmp/all_py.txt /tmp/recently_changed.txt

# <tool>
# Find Python source files that have no corresponding test file — untested modules.
# Looks for matching test_*.py or *_test.py files.
# </tool>
find-untested:
	@echo "=== Source files with no matching test file ==="; \
	find . -name "*.py" \
	  -not -name "test_*" -not -name "*_test.py" \
	  -not -path '*/.venv/*' -not -path '*/__pycache__/*' \
	  -not -path '*/tests/*' -not -path '*/.git/*' \
	  -not -name "conftest.py" -not -name "setup.py" \
	| while read src; do \
	    base=$$(basename $$src .py); \
	    if ! find . -name "test_$${base}.py" -o -name "$${base}_test.py" 2>/dev/null | grep -q .; then \
	      echo "$$src"; \
	    fi; \
	  done

# <tool>
# Find all TODO, FIXME, HACK, and XXX comments in Python files — a map of known debt.
# </tool>
find-todos:
	@grep -rn --include="*.py" \
	  -E "(TODO|FIXME|HACK|XXX|NOQA|type: ignore)" . \
	  --exclude-dir=__pycache__ --exclude-dir=.venv --exclude-dir=.git \
	  || echo "No debt markers found."

# <tool>
# Run the test suite and show a summary. Optionally target a specific module or test path.
# @param TARGET string Optional pytest target path or module (e.g. tests/test_auth.py)
# </tool>
run-tests:
	@python -m pytest $(if $(TARGET),$(TARGET),) -v --tb=short --no-header 2>&1 | tail -40

# <tool>
# List all top-level function and class definitions in a Python file.
# @param PATH string Relative path to the Python file
# </tool>
list-definitions:
	@grep -n "^class \|^def \|^    def " $(PATH) \
	  || echo "No definitions found in $(PATH)"
