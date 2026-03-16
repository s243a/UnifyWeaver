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
    def send_message(self, message: str, context: list[dict], **kwargs) -> AgentResponse:
        """Send a message with context, get response.

        Optional kwargs:
            on_status: Callback for status updates (e.g. tool call progress)
        """
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



class AsyncApiBackend(AgentBackend):
    """Async HTTP backend using aiohttp for non-blocking API calls."""

    def __init__(self, name: str, endpoint: str, api_key: str | None = None,
                 model: str = "", stream: bool = False, api_format: str = "openai"):
        self._name = name
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.stream = stream
        self.api_format = api_format

    @property
    def name(self) -> str:
        return self._name

    def _build_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            if self.api_format == "anthropic":
                headers["x-api-key"] = self.api_key
                headers["anthropic-version"] = "2023-06-01"
            else:
                headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_body(self, message: str, context: list[dict]) -> dict:
        messages = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in context]
        messages.append({"role": "user", "content": message})
        if self.api_format == "anthropic":
            return {"model": self.model, "messages": messages, "max_tokens": 4096, "stream": self.stream}
        return {"model": self.model, "messages": messages, "stream": self.stream}

    async def send_message_async(self, message: str, context: list[dict], **kwargs) -> AgentResponse:
        """Send message asynchronously using aiohttp."""
        try:
            import aiohttp
        except ImportError:
            # Fallback to sync if aiohttp not available
            return self.send_message(message, context, **kwargs)

        headers = self._build_headers()
        body = self._build_body(message, context)

        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoint, json=body, headers=headers) as resp:
                data = await resp.json()

        return self._parse_response(data)

    def send_message(self, message: str, context: list[dict], **kwargs) -> AgentResponse:
        """Synchronous fallback using urllib."""
        import urllib.request
        import json as _json
        headers = self._build_headers()
        body = self._build_body(message, context)
        req = urllib.request.Request(
            self.endpoint,
            data=_json.dumps(body).encode(),
            headers=headers,
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = _json.loads(resp.read())
        return self._parse_response(data)

    def _parse_response(self, data: dict) -> AgentResponse:
        tool_calls = []
        if self.api_format == "anthropic":
            content = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    content = block.get("text", "")
                elif block.get("type") == "tool_use":
                    tool_calls.append(ToolCall(
                        name=block.get("name", ""),
                        arguments=block.get("input", {}),
                        id=block.get("id", "")
                    ))
            usage = data.get("usage", {})
            tokens = {"input": usage.get("input_tokens", 0), "output": usage.get("output_tokens", 0)}
        else:
            choice = data.get("choices", [{}])[0].get("message", {})
            content = choice.get("content", "") or ""
            for tc in choice.get("tool_calls", []):
                import json as _tc_json
                args = tc.get("function", {}).get("arguments", "{}")
                try:
                    parsed_args = _tc_json.loads(args) if isinstance(args, str) else args
                except (ValueError, TypeError):
                    parsed_args = {}
                tool_calls.append(ToolCall(
                    name=tc.get("function", {}).get("name", ""),
                    arguments=parsed_args,
                    id=tc.get("id", "")
                ))
            usage = data.get("usage", {})
            tokens = {"input": usage.get("prompt_tokens", 0), "output": usage.get("completion_tokens", 0)}
        return AgentResponse(content=content, tool_calls=tool_calls, tokens=tokens, raw=data)

    def supports_streaming(self) -> bool:
        return self.stream
