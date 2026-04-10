define SYSTEM_PROMPT
You are a file explorer agent. You help users navigate and search file systems. Use the provided tools to list directories, find files by name or extension, and search file contents with grep. Always report errors by echoing a message starting with ERROR:. If you are unsure about how to complete a task, ask for help rather than making assumptions.
endef

define  DISABLED_BUILTINS
all
endef


.PHONY: list_files search_by_name search_by_extension grep_in_files

# <tool>
# List files and directories in the specified directory.
# @param DIR string Directory to list contents of
# </tool>
list_files:
	@ls -la "$(DIR)"


# <tool>
# Search for files by name pattern in a directory recursively.
# @param NAME string Filename or pattern to search for (supports wildcards)
# @param DIR string Directory to search in
# </tool>
search_by_name:
	@find "$(DIR)" -type f -name "$(NAME)" 2>/dev/null || echo "No matches found"


# <tool>
# Search for files by extension in a directory recursively.
# @param EXT string File extension to search for (without dot, e.g., "txt" or "py")
# @param DIR string Directory to search in
# </tool>
search_by_extension:
	@find "$(DIR)" -type f -name "*.$(EXT)" 2>/dev/null || echo "No matches found"


# <tool>
# Search for a text pattern in files within a directory recursively.
# @param PATTERN string Pattern to search for (regex supported)
# @param DIR string Directory to search in
# </tool>
grep_in_files:
	@grep -rn "$(PATTERN)" "$(DIR)" 2>/dev/null || echo "No matches found"

