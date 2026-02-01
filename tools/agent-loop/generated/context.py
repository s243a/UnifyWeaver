"""Context manager for conversation history."""

from dataclasses import dataclass, field
from typing import Literal
from enum import Enum


class ContextBehavior(Enum):
    """How to handle context across messages."""
    CONTINUE = "continue"    # Full context, continue conversation
    FRESH = "fresh"          # No context, each query independent
    SLIDING = "sliding"      # Keep only last N messages
    SUMMARIZE = "summarize"  # Summarize old context (future)


class ContextFormat(Enum):
    """How to format context for the backend."""
    PLAIN = "plain"          # Simple text with role prefixes
    MARKDOWN = "markdown"    # Markdown formatted
    JSON = "json"            # JSON array of messages
    XML = "xml"              # XML structure


@dataclass
class Message:
    """A single message in the conversation."""
    role: Literal["user", "assistant", "system"]
    content: str
    tokens: int = 0


class ContextManager:
    """Manages conversation history and context window."""

    def __init__(
        self,
        max_tokens: int = 100000,
        max_messages: int = 50,
        behavior: ContextBehavior = ContextBehavior.CONTINUE,
        format: ContextFormat = ContextFormat.PLAIN
    ):
        self.messages: list[Message] = []
        self.max_tokens = max_tokens
        self.max_messages = max_messages
        self.behavior = behavior
        self.format = format
        self._token_count = 0

    def add_message(self, role: str, content: str, tokens: int = 0) -> None:
        """Add a message to the context."""
        msg = Message(role=role, content=content, tokens=tokens)
        self.messages.append(msg)
        self._token_count += tokens
        self._trim_if_needed()

    def _trim_if_needed(self) -> None:
        """Remove old messages if context is too large."""
        # Trim by message count
        while len(self.messages) > self.max_messages:
            removed = self.messages.pop(0)
            self._token_count -= removed.tokens

        # Trim by token count (keep at least 2 messages)
        while self._token_count > self.max_tokens and len(self.messages) > 2:
            removed = self.messages.pop(0)
            self._token_count -= removed.tokens

    def get_context(self) -> list[dict]:
        """Get context formatted for the backend."""
        if self.behavior == ContextBehavior.FRESH:
            return []

        messages = self.messages
        if self.behavior == ContextBehavior.SLIDING:
            # Keep only recent messages
            messages = self.messages[-10:]

        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

    def get_formatted_context(self) -> str:
        """Get context as a formatted string."""
        context = self.get_context()

        if self.format == ContextFormat.PLAIN:
            return self._format_plain(context)
        elif self.format == ContextFormat.MARKDOWN:
            return self._format_markdown(context)
        elif self.format == ContextFormat.JSON:
            return self._format_json(context)
        elif self.format == ContextFormat.XML:
            return self._format_xml(context)

        return self._format_plain(context)

    def _format_plain(self, context: list[dict]) -> str:
        """Format context as plain text."""
        lines = []
        for msg in context:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def _format_markdown(self, context: list[dict]) -> str:
        """Format context as markdown."""
        lines = []
        for msg in context:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"**{role}:**\n{msg['content']}\n")
        return "\n".join(lines)

    def _format_json(self, context: list[dict]) -> str:
        """Format context as JSON."""
        import json
        return json.dumps(context, indent=2)

    def _format_xml(self, context: list[dict]) -> str:
        """Format context as XML."""
        lines = ["<conversation>"]
        for msg in context:
            lines.append(f'  <message role="{msg["role"]}">')
            lines.append(f"    {msg['content']}")
            lines.append("  </message>")
        lines.append("</conversation>")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all messages from context."""
        self.messages.clear()
        self._token_count = 0

    @property
    def token_count(self) -> int:
        """Current total token count."""
        return self._token_count

    @property
    def message_count(self) -> int:
        """Current message count."""
        return len(self.messages)

    def __len__(self) -> int:
        return len(self.messages)
