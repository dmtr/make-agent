define DESCRIPTION
A file editing agent with tools to list, read, write, and modify files.
endef

define SYSTEM_PROMPT
You are a file editing agent. Use the provided tools to read, create, and
modify files. When editing an existing file, first read it with read-file or
read-lines to understand its content, then apply changes with write-file (full
rewrite) or replace-in-file (targeted replacement). Always verify the result
by reading the file again after making changes. Report all errors clearly.
endef

define  DISABLED_BUILTINS
all
endef


.PHONY: list-files count-lines read-file read-lines write-file append-to-file replace-in-file

# <tool>
# List files and directories at the given path.
# @param DIR string Directory to list (use . for the current directory)
# </tool>
list-files:
	@ls -la "$$DIR"

# <tool>
# Count the total number of lines in a file.
# @param FILE string Path to the file
# </tool>
count-lines:
	@[ -f "$$FILE" ] || { echo "Error: file not found: $$FILE"; exit 1; }
	@wc -l < "$$FILE"

# <tool>
# Read and print the full contents of a file.
# @param FILE string Path to the file
# </tool>
read-file:
	@[ -f "$$FILE" ] || { echo "Error: file not found: $$FILE"; exit 1; }
	@cat "$$FILE"

# <tool>
# Read a range of lines from a file (line numbers are 1-based).
# If END is 0, reads from START to the last line of the file.
# @param FILE string Path to the file
# @param START integer First line to read (1-based)
# @param END integer Last line to read inclusive; 0 means end of file
# </tool>
read-lines:
	@[ -f "$$FILE" ] || { echo "Error: file not found: $$FILE"; exit 1; }
	@if [ "$$END" = "0" ] || [ -z "$$END" ]; then \
		tail -n "+$$START" "$$FILE"; \
	else \
		awk "NR>=$$START && NR<=$$END" "$$FILE"; \
	fi

# <tool>
# Create or overwrite a file with the given content.
# Creates parent directories automatically if they do not exist.
# @param FILE string Path of the file to write
# @param CONTENT string Text content to write (may be multiline)
# </tool>
write-file:
	@mkdir -p "$$(dirname $$FILE)"
	@printf '%s' "$$CONTENT" > "$$FILE" && echo "Written: $$FILE"

# <tool>
# Append content to the end of a file, creating the file if it does not exist.
# @param FILE string Path of the file to append to
# @param CONTENT string Content to append (may be multiline)
# </tool>
append-to-file:
	@printf '%s' "$$CONTENT" >> "$$FILE" && echo "Appended to: $$FILE"

# <tool>
# Replace the first occurrence of literal string OLD with NEW inside FILE.
# Exits with a non-zero status and prints an error if OLD is not found.
# @param FILE string Path to the file to modify
# @param OLD string Exact text to find (literal string, not a regex)
# @param NEW string Replacement text
# </tool>
replace-in-file:
	@[ -f "$$FILE" ] || { echo "Error: file not found: $$FILE"; exit 1; }
	@python3 -c "\
import sys; \
p, o, n = sys.argv[1:]; \
t = open(p).read(); \
(print('Error: text not found in ' + p), sys.exit(1)) if o not in t \
else (open(p, 'w').write(t.replace(o, n, 1)), print('Replaced in: ' + p)) \
" "$$FILE" "$$OLD" "$$NEW"
