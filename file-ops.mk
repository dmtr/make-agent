define SYSTEM_PROMPT
You are a file operations specialist. Your role is to help with reading, writing, searching, and managing files.
You have tools for:
- Reading file contents (with safeguards against large files)
- Writing and appending to files
- Listing and exploring directories
- Searching for files and patterns
- Deleting and renaming files
- Getting file information
- Counting lines, words, and bytes in files
- Previewing files with head and tail

IMPORTANT: To prevent context pollution, do NOT read large files directly. Use the wc-file tool first to check file size.
If a file is larger than 10KB, use head-file or tail-file instead, or ask the orchestrator for help.

Always report errors by echoing a message that starts with "ERROR:" if something goes wrong.
If you're unsure about how to complete a task, ask the orchestrator for help rather than making assumptions.
endef

.PHONY: read-file wc-file head-file tail-file write-file append-file list-directory search-files search-content delete-file rename-file file-info copy-file

# <tool>
# Read and display the contents of a file. WARNING - only use on small files to avoid context pollution. Check file size first with wc-file.
# @param FILE_PATH string Path to the file to read
# </tool>
read-file:
	@cat $(FILE_PATH) 2>&1 || echo 'ERROR: Could not read file $(FILE_PATH)'


# <tool>
# Count lines, words, and bytes in a file. Use this to check file size before reading.
# @param FILE_PATH string Path to the file to count
# </tool>
wc-file:
	@wc -l -w -c $(FILE_PATH) 2>&1 || echo 'ERROR: Could not count file $(FILE_PATH)'


# <tool>
# Display the first N lines of a file. Useful for previewing large files.
# @param FILE_PATH string Path to the file to preview
# @param LINES integer Number of lines to display (default 10)
# </tool>
head-file:
	@head -n $(LINES) $(FILE_PATH) 2>&1 || echo 'ERROR: Could not read file $(FILE_PATH)'


# <tool>
# Display the last N lines of a file. Useful for viewing end of large files or logs.
# @param FILE_PATH string Path to the file to preview
# @param LINES integer Number of lines to display (default 10)
# </tool>
tail-file:
	@tail -n $(LINES) $(FILE_PATH) 2>&1 || echo 'ERROR: Could not read file $(FILE_PATH)'


# <tool>
# Write content to a file (overwrites if exists).
# @param FILE_PATH string Path to the file to write to
# @param CONTENT string Content to write to the file
# </tool>
write-file:
	@mkdir -p $(dirname $(FILE_PATH))
	@echo $(CONTENT_FILE) > $(FILE_PATH)


# <tool>
# Append content to a file.
# @param FILE_PATH string Path to the file to append to
# @param CONTENT string Content to append to the file
# </tool>
append-file:
	@mkdir -p $(dirname $(FILE_PATH))
	@echo $(CONTENT_FILE) >> $(FILE_PATH)


# <tool>
# List files and directories in a given path.
# @param DIR_PATH string Path to the directory to list
# </tool>
list-directory:
	@ls -la $(DIR_PATH) 2>&1 || echo 'ERROR: Could not list directory $(DIR_PATH)'


# <tool>
# Search for files matching a pattern or name in a directory.
# @param PATTERN string File name pattern or regex to search for
# @param DIR_PATH string Directory to search in
# </tool>
search-files:
	@find $(DIR_PATH) -name '$(PATTERN)' 2>&1 || echo 'ERROR: Search failed in $(DIR_PATH)'


# <tool>
# Search for text content within files in a directory.
# @param PATTERN string Text pattern to search for (regex)
# @param DIR_PATH string Directory to search in
# </tool>
search-content:
	@grep -rn '$(PATTERN)' $(DIR_PATH) 2>&1 || echo 'No matches found for pattern: $(PATTERN)'


# <tool>
# Delete a file.
# @param FILE_PATH string Path to the file to delete
# </tool>
delete-file:
	@rm -f $(FILE_PATH) && echo 'File deleted: $(FILE_PATH)' || echo 'ERROR: Could not delete file $(FILE_PATH)'


# <tool>
# Rename or move a file.
# @param OLD_PATH string Current file path
# @param NEW_PATH string New file path
# </tool>
rename-file:
	@mv $(OLD_PATH) $(NEW_PATH) 2>&1 && echo 'File renamed from $(OLD_PATH) to $(NEW_PATH)' || echo 'ERROR: Could not rename file'


# <tool>
# Get detailed information about a file.
# @param FILE_PATH string Path to the file
# </tool>
file-info:
	@stat $(FILE_PATH) 2>&1 || echo 'ERROR: Could not get file info for $(FILE_PATH)'


# <tool>
# Copy a file to a new location.
# @param SOURCE_PATH string Source file path
# @param DEST_PATH string Destination file path
# </tool>
copy-file:
	@cp -v $(SOURCE_PATH) $(DEST_PATH) 2>&1 || echo 'ERROR: Could not copy file'

