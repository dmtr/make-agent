"""User message history manager for the agent shell.

Provides navigation through past user messages backed by persistent Memory.
The shell interacts with this manager, not Memory directly.
"""

from __future__ import annotations

from typing import Optional

from make_agent.memory import Memory


class UserMessagesManager:
    """Manages user message history for shell navigation.

    Refreshes messages from Memory on each navigation start so that
    messages from the current session are always included.  Supports
    forward/backward navigation with index tracking.
    """

    def __init__(self, memory: Memory, limit: int = 200) -> None:
        """Initialize with a Memory instance.

        Args:
            memory: The Memory backend to fetch messages from.
            limit: Maximum number of historical messages to load (default: 200).
        """
        self._memory = memory
        self._limit = limit
        self._messages: list[str] = []  # Refreshed on each nav start, newest first
        self._index: int = -1  # -1 = not navigating; 0 = most recent
        self._original_text: str = ""  # Text user had typed before pressing UP

    def _refresh(self) -> None:
        """Reload user messages from Memory so current-session messages
        are always visible."""
        self._messages = self._memory.recent_user(self._limit)

    def start_navigating(self, current_text: str) -> Optional[str]:
        """Begin navigation from the current composer text.

        Called when user first presses UP.  Reloads messages from Memory
        so current-session messages are included.  Saves the current text
        as the restoration point (returned when DOWN exits navigation).

        Args:
            current_text: What the user has typed so far.

        Returns:
            The most recent past user message, or None if history is empty.
        """
        self._refresh()
        if not self._messages:
            return None
        self._original_text = current_text
        self._index = 0
        return self._messages[0]

    def previous(self, current_text: str = "") -> Optional[str]:
        """Navigate to the next older message.

        Args:
            current_text: What the user has typed so far (saved as restoration point
                          on first UP press).

        Returns:
            The previous message, or None if already at the oldest.
        """
        if self._index < 0:
            # First press — start navigating, save current text as restoration point
            return self.start_navigating(current_text)
        if self._index < len(self._messages) - 1:
            self._index += 1
            return self._messages[self._index]
        return None

    def next(self) -> Optional[str]:
        """Navigate to the next newer message or restore original text.

        Returns:
            The next newer message, or the original typed text when
            navigation returns to the start, or None if not navigating.
        """
        if self._index < 0:
            return None
        if self._index > 0:
            self._index -= 1
            return self._messages[self._index]
        # Back at start — restore original text
        text = self._original_text
        self._index = -1
        self._original_text = ""
        return text

    def cancel(self) -> None:
        """Cancel navigation and restore the original typed text."""
        self._index = -1

    def submit(self) -> None:
        """Called when the user sends a message (presses Enter).

        Resets navigation state so the next UP press starts fresh.
        """
        self._index = -1
        self._original_text = ""

    def is_navigating(self) -> bool:
        """Check if the user is currently in navigation mode."""
        return self._index >= 0
