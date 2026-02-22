"""Context manager for conversation history."""

from dataclasses import dataclass, field
from typing import Literal
from enum import Enum


class ContextBehavior(Enum):
    """How to handle context across messages."""
    CONTINUE = "continue"    # Full context, continue conversation
    FRESH = "fresh"    # No context, each query independent
    SLIDING = "sliding"    # Keep only last N messages
    SUMMARIZE = "summarize"    # Summarize old context (future)


class ContextFormat(Enum):
    """How to format context for the backend."""
    PLAIN = "plain"    # Simple text with role prefixes
    MARKDOWN = "markdown"    # Markdown formatted
    JSON = "json"    # JSON array of messages
    XML = "xml"    # XML structure


@dataclass
class Message:
    """A single message in the conversation."""
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    tokens: int = 0
    tool_call_id: str = ""
    tool_calls: list = field(default_factory=list)



def _count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text (chars / 4 heuristic)."""
    return max(1, len(text) // 4)


class ContextManager:
    """Manages conversation history and context window."""

    def __init__(
        self,
        max_tokens: int = 100000,
        max_messages: int = 50,
        max_chars: int = 0,
        max_words: int = 0,
        behavior: ContextBehavior = ContextBehavior.CONTINUE,
        format: ContextFormat = ContextFormat.PLAIN
    ):
        self.messages: list[Message] = []
        self.max_tokens = max_tokens
        self.max_messages = max_messages
        self.max_chars = max_chars
        self.max_words = max_words
        self.behavior = behavior
        self.format = format
        self._token_count = 0
        self._char_count = 0
        self._word_count = 0

    def add_message(self, role: str, content: str, tokens: int = 0,
                    tool_call_id: str = "", tool_calls: list | None = None) -> None:
        """Add a message to the context."""
        if tokens == 0:
            tokens = _estimate_tokens(content)

        msg = Message(role=role, content=content, tokens=tokens,
                      tool_call_id=tool_call_id,
                      tool_calls=tool_calls or [])
        self.messages.append(msg)
        self._token_count += tokens
        self._char_count += len(content)
        self._word_count += _count_words(content)
        self._trim_if_needed()

    def _trim_if_needed(self) -> None:
        """Remove old messages if any context limit is exceeded."""
        while len(self.messages) > self.max_messages:
            self._remove_oldest()

        while self._token_count > self.max_tokens and len(self.messages) > 1:
            self._remove_oldest()

        if self.max_chars > 0:
            while self._char_count > self.max_chars and len(self.messages) > 1:
                self._remove_oldest()

        if self.max_words > 0:
            while self._word_count > self.max_words and len(self.messages) > 1:
                self._remove_oldest()

    def _remove_oldest(self) -> None:
        """Remove the oldest message and update counters."""
        removed = self.messages.pop(0)
        self._token_count -= removed.tokens
        self._char_count -= len(removed.content)
        self._word_count -= _count_words(removed.content)

    @staticmethod
    def _msg_to_dict(msg: 'Message') -> dict:
        """Convert a Message to a dict, including tool fields when present."""
        d = {"role": msg.role, "content": msg.content}
        if msg.role == 'tool' and msg.tool_call_id:
            d["tool_call_id"] = msg.tool_call_id
        if msg.role == 'assistant' and msg.tool_calls:
            d["tool_calls"] = msg.tool_calls
        return d

    def get_context(self) -> list[dict]:
        """Get context formatted for the backend."""
        if self.behavior == ContextBehavior.FRESH:
            return []

        messages = self.messages
        if self.behavior == ContextBehavior.SLIDING:
            messages = self.messages[-self.max_messages:]

        if self.max_chars <= 0 and self.max_words <= 0:
            return [self._msg_to_dict(msg) for msg in messages]

        result = []
        total_chars = 0
        total_words = 0
        for msg in reversed(messages):
            msg_chars = len(msg.content)
            msg_words = _count_words(msg.content)

            if self.max_chars > 0 and total_chars + msg_chars > self.max_chars and result:
                break
            if self.max_words > 0 and total_words + msg_words > self.max_words and result:
                break

            result.append(self._msg_to_dict(msg))
            total_chars += msg_chars
            total_words += msg_words

        result.reverse()
        return result

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
            lines.append(f'  <message role="{msg["role"]}">'  )
            lines.append(f"    {msg['content']}")
            lines.append("  </message>")
        lines.append("</conversation>")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all messages from context."""
        self.messages.clear()
        self._token_count = 0
        self._char_count = 0
        self._word_count = 0

    @property
    def token_count(self) -> int:
        """Current total token count."""
        return self._token_count

    @property
    def char_count(self) -> int:
        """Current total character count."""
        return self._char_count

    @property
    def word_count(self) -> int:
        """Current total word count."""
        return self._word_count

    @property
    def message_count(self) -> int:
        """Current message count."""
        return len(self.messages)

    def __len__(self) -> int:
        return len(self.messages)

