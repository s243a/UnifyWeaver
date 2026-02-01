"""Conversation history management with edit/delete support."""

from dataclasses import dataclass
from typing import Literal
from context import ContextManager, Message


@dataclass
class HistoryEntry:
    """A numbered entry in the conversation history."""
    index: int
    role: str
    content: str
    tokens: int
    truncated: bool = False


class HistoryManager:
    """Manages conversation history with edit/delete capabilities."""

    def __init__(self, context: ContextManager):
        self.context = context
        self._undo_stack: list[list[Message]] = []
        self._max_undo = 10

    def get_entries(self, limit: int = 20) -> list[HistoryEntry]:
        """Get recent history entries with indices."""
        entries = []
        messages = self.context.messages[-limit:] if limit else self.context.messages
        start_idx = len(self.context.messages) - len(messages)

        for i, msg in enumerate(messages):
            content = msg.content
            truncated = False
            if len(content) > 100:
                content = content[:97] + "..."
                truncated = True

            entries.append(HistoryEntry(
                index=start_idx + i,
                role=msg.role,
                content=content,
                tokens=msg.tokens,
                truncated=truncated
            ))

        return entries

    def format_history(self, limit: int = 10) -> str:
        """Format history for display."""
        entries = self.get_entries(limit)
        if not entries:
            return "No messages in history."

        lines = [f"Last {len(entries)} messages:"]
        for e in entries:
            role = "You" if e.role == "user" else "Asst"
            # Show first line only, truncated
            first_line = e.content.split('\n')[0]
            if len(first_line) > 60:
                first_line = first_line[:57] + "..."
            lines.append(f"  [{e.index}] {role}: {first_line}")

        return "\n".join(lines)

    def get_message(self, index: int) -> Message | None:
        """Get a message by index."""
        if 0 <= index < len(self.context.messages):
            return self.context.messages[index]
        return None

    def get_full_content(self, index: int) -> str | None:
        """Get full content of a message by index."""
        msg = self.get_message(index)
        return msg.content if msg else None

    def edit_message(self, index: int, new_content: str) -> bool:
        """Edit a message's content."""
        if not 0 <= index < len(self.context.messages):
            return False

        # Save state for undo
        self._save_state()

        # Edit the message
        old_msg = self.context.messages[index]
        self.context.messages[index] = Message(
            role=old_msg.role,
            content=new_content,
            tokens=old_msg.tokens  # Keep original token count
        )
        return True

    def delete_message(self, index: int) -> bool:
        """Delete a message by index."""
        if not 0 <= index < len(self.context.messages):
            return False

        # Save state for undo
        self._save_state()

        # Remove the message
        removed = self.context.messages.pop(index)
        self.context._token_count -= removed.tokens
        return True

    def delete_range(self, start: int, end: int) -> int:
        """Delete a range of messages [start, end]. Returns count deleted."""
        if start < 0:
            start = 0
        if end >= len(self.context.messages):
            end = len(self.context.messages) - 1
        if start > end:
            return 0

        # Save state for undo
        self._save_state()

        # Remove messages in reverse order to preserve indices
        count = 0
        for i in range(end, start - 1, -1):
            if 0 <= i < len(self.context.messages):
                removed = self.context.messages.pop(i)
                self.context._token_count -= removed.tokens
                count += 1

        return count

    def delete_last(self, n: int = 1) -> int:
        """Delete the last N messages. Returns count deleted."""
        if n <= 0:
            return 0

        # Save state for undo
        self._save_state()

        count = 0
        for _ in range(n):
            if self.context.messages:
                removed = self.context.messages.pop()
                self.context._token_count -= removed.tokens
                count += 1

        return count

    def undo(self) -> bool:
        """Undo the last edit/delete operation."""
        if not self._undo_stack:
            return False

        # Restore previous state
        self.context.messages = self._undo_stack.pop()
        self.context._token_count = sum(m.tokens for m in self.context.messages)
        return True

    def _save_state(self) -> None:
        """Save current state for undo."""
        # Deep copy messages
        state = [
            Message(role=m.role, content=m.content, tokens=m.tokens)
            for m in self.context.messages
        ]
        self._undo_stack.append(state)

        # Limit undo stack size
        while len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self._undo_stack) > 0

    def truncate_after(self, index: int) -> int:
        """Delete all messages after index. Returns count deleted."""
        if index < 0 or index >= len(self.context.messages) - 1:
            return 0

        # Save state for undo
        self._save_state()

        count = 0
        while len(self.context.messages) > index + 1:
            removed = self.context.messages.pop()
            self.context._token_count -= removed.tokens
            count += 1

        return count

    def replay_from(self, index: int) -> str | None:
        """Get the user message at index for replay.

        Truncates history after that point so the message can be re-sent.
        """
        msg = self.get_message(index)
        if msg is None or msg.role != "user":
            return None

        # Get the content before truncating
        content = msg.content

        # Truncate to just before this message
        self.truncate_after(index - 1)

        return content
