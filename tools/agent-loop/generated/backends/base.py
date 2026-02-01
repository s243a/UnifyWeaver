"""Abstract base class for agent backends."""

from dataclasses import dataclass, field
from typing import Any
from abc import ABC, abstractmethod


@dataclass
class ToolCall:
    """Represents a tool call from the agent."""
    name: str
    arguments: dict[str, Any]
    id: str = ""


@dataclass
class AgentResponse:
    """Response from an agent backend."""
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tokens: dict[str, int] = field(default_factory=dict)
    raw: Any = None


class AgentBackend(ABC):
    """Abstract interface for agent backends."""

    @abstractmethod
    def send_message(self, message: str, context: list[dict]) -> AgentResponse:
        """Send a message with context, get response."""
        raise NotImplementedError

    def parse_tool_calls(self, response: str) -> list[ToolCall]:
        """Extract tool calls from response. Override in subclasses."""
        return []

    def supports_streaming(self) -> bool:
        """Whether this backend supports streaming output."""
        return False

    @property
    def name(self) -> str:
        """Backend name for display."""
        return self.__class__.__name__
