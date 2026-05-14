---
description: "Navigate and search the filesystem by name, extension, or content"
---

# File Explorer Skill

This skill provides tools to navigate directories and search the local filesystem by filename, extension, or text content.

## Usage

Start with `list-files` to orient yourself in a directory. Use `search-by-name` or `search-by-extension` to locate files, and `grep-in-files` to find files containing a specific pattern.

## Available tools

- `list-files DIR=path` — List files and directories in the specified directory
- `search-by-name DIR=path NAME=pattern` — Search for files by name pattern recursively (supports wildcards)
- `search-by-extension DIR=path EXT=ext` — Search for files by extension recursively (e.g., `EXT=py`)
- `grep-in-files DIR=path PATTERN=regex` — Search for a text pattern in files recursively
