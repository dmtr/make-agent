---
description: "Read, write, and edit text files on the filesystem"
---

# File Edit Skill

This skill provides tools to read, write, and precisely modify text files on the local filesystem.

## Usage

Before editing, read the file first to understand its current content. Use `replace-in-file` for targeted changes and `write-file` only when rewriting the whole file. Always verify results by reading the file again after making changes.

## Available tools

- `list-files DIR=path` — List files and directories at the given path
- `count-lines FILE=path` — Count the total number of lines in a file
- `read-file FILE=path` — Read and print the full contents of a file
- `read-lines FILE=path START=n END=n` — Read a specific line range (END=0 means end of file)
- `write-file FILE=path CONTENT=text` — Create or overwrite a file with the given content
- `append-to-file FILE=path CONTENT=text` — Append content to the end of a file
- `replace-in-file FILE=path OLD=text NEW=text` — Replace the first occurrence of a literal string in a file
