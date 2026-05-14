.PHONY: list-files search-by-name search-by-extension grep-in-files

# <tool>
# List files and directories in the specified directory.
# @param DIR string Directory to list contents of
# </tool>
list-files:
	@ls -la "$$DIR"

# <tool>
# Search for files by name pattern in a directory recursively.
# @param NAME string Filename or pattern to search for (supports wildcards)
# @param DIR string Directory to search in
# </tool>
search-by-name:
	@find "$$DIR" -type f -name "$$NAME" 2>/dev/null || echo "No matches found"

# <tool>
# Search for files by extension in a directory recursively.
# @param EXT string File extension to search for (without dot, e.g., "txt" or "py")
# @param DIR string Directory to search in
# </tool>
search-by-extension:
	@find "$$DIR" -type f -name "*.$$EXT" 2>/dev/null || echo "No matches found"

# <tool>
# Search for a text pattern in files within a directory recursively.
# @param PATTERN string Pattern to search for (regex supported)
# @param DIR string Directory to search in
# </tool>
grep-in-files:
	@grep -rn "$$PATTERN" "$$DIR" 2>/dev/null || echo "No matches found"
