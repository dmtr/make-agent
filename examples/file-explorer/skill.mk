define DESCRIPTION
Navigate and search the filesystem by name, extension, or content
endef

.PHONY: list-files search-by-name search-by-extension grep-in-files

list-files:
	@ls -la "$$DIR"

search-by-name:
	@find "$$DIR" -type f -name "$$NAME" 2>/dev/null || echo "No matches found"

search-by-extension:
	@find "$$DIR" -type f -name "*.$$EXT" 2>/dev/null || echo "No matches found"

grep-in-files:
	@grep -rn "$$PATTERN" "$$DIR" 2>/dev/null || echo "No matches found"
