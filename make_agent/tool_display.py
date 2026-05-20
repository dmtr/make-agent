"""Structured tool call display formatter.

Formats tool execution events for clear, readable output in the terminal.
Supports color coding, collapsible panels, and timing information.
"""

from __future__ import annotations

from typing import Any


# ANSI escape codes
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_GREEN = "\033[92m"
ANSI_RED = "\033[91m"
ANSI_YELLOW = "\033[93m"
ANSI_CYAN = "\033[96m"
ANSI_DIM = "\033[2m"


class ToolDisplayFormatter:
    """Formats tool call events for structured display."""

    def __init__(self, max_output_preview: int = 200) -> None:
        self.max_output_preview = max_output_preview

    def format_start(self, tool_name: str, args: dict[str, Any]) -> str:
        """Format a tool call before execution.

        Returns a formatted string showing the tool name and parameters.
        """
        lines = [f"{ANSI_CYAN}{ANSI_BOLD}🔧 {tool_name}{ANSI_RESET}"]

        # Format each parameter
        if isinstance(args, dict):
            for key, value in args.items():
                str_value = self._truncate(str(value), 50)
                lines.append(f"  {key}: {str_value}")
        elif args:
            lines.append(f"  {self._truncate(str(args), 50)}")

        return "\n".join(lines)

    def format_done(
        self,
        tool_name: str,
        output: str,
        is_error: bool,
        duration_ms: float | None = None,
    ) -> str:
        """Format a tool call after execution.

        Returns a formatted string showing status, timing, and output preview.
        """
        # Status indicator
        if is_error:
            status_icon = "❌"
            status_color = ANSI_RED
        else:
            status_icon = "✅"
            status_color = ANSI_GREEN

        # Timing info
        time_str = ""
        if duration_ms is not None:
            if duration_ms < 1000:
                time_str = f"{duration_ms:.0f}ms"
            else:
                time_str = f"{duration_ms / 1000:.1f}s"

        # Build output
        lines = [f"{status_color}{ANSI_BOLD}{status_icon} {tool_name} (completed in {time_str}){ANSI_RESET}"]

        # Output preview (truncated if needed)
        if output:
            preview = self._truncate(output, self.max_output_preview)
            if len(output) > self.max_output_preview:
                preview += f"\n{ANSI_DIM}[... {len(output) - self.max_output_preview} more characters]{ANSI_RESET}"
            lines.append(f"  {preview}")
        else:
            lines.append(f"  {ANSI_DIM}(no output){ANSI_RESET}")

        return "\n".join(lines)

    def format_collapsible(
        self, tool_name: str, args: dict[str, Any], output: str, is_error: bool, duration_ms: float | None = None
    ) -> str:
        """Format a collapsible tool call panel.

        The output starts collapsed and can be expanded by the terminal.
        """
        # Start with collapsed view
        start_lines = [
            f"{ANSI_CYAN}{ANSI_BOLD}🔧 {tool_name}{ANSI_RESET}",
        ]
        if isinstance(args, dict):
            for key, value in args.items():
                str_value = self._truncate(str(value), 50)
                start_lines.append(f"  {key}: {str_value}")
        elif args:
            start_lines.append(f"  {self._truncate(str(args), 50)}")

        # Add expand/collapse marker
        expand_marker = f"{ANSI_DIM}[Click to expand]{ANSI_RESET}"
        start_lines.append(f"\n{expand_marker}")

        # Output preview (if any)
        if output:
            preview = self._truncate(output, 100)
            start_lines.append(f"\n{ANSI_DIM}Output: {self._truncate(preview, 80)}...{ANSI_RESET}")

        return "\n".join(start_lines)

    def _truncate(self, text: str, max_len: int) -> str:
        """Truncate text to max_len with ellipsis if needed."""
        if len(text) <= max_len:
            return text
        return text[:max_len - 3] + "..."

    def format_error_suggestion(self, error_msg: str) -> str | None:
        """Generate helpful suggestions based on common error patterns.

        Returns a formatted suggestion string or None if no pattern matches.
        """
        error_lower = error_msg.lower()

        # Common error patterns and suggestions
        if "permission denied" in error_lower:
            return f"{ANSI_YELLOW}💡 Suggestion: Check file permissions with 'ls -la'{ANSI_RESET}"
        elif "not found" in error_lower or "no such file" in error_lower:
            return f"{ANSI_YELLOW}💡 Suggestion: Verify the path exists and check current directory with 'pwd'{ANSI_RESET}"
        elif "timeout" in error_lower:
            return f"{ANSI_YELLOW}💡 Suggestion: The tool may be taking too long. Try with smaller input or check for infinite loops{ANSI_RESET}"
        elif "syntax error" in error_lower:
            return f"{ANSI_YELLOW}💡 Suggestion: Check the syntax of your input for typos or missing characters{ANSI_RESET}"

        return None


def create_formatter(max_output_preview: int = 200) -> ToolDisplayFormatter:
    """Factory function to create a ToolDisplayFormatter instance."""
    return ToolDisplayFormatter(max_output_preview=max_output_preview)
