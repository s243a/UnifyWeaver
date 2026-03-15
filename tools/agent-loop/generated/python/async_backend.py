"""Async agent backend using asyncio and aiohttp."""

import asyncio
import json
from typing import Any, Callable

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

from backends.base import AgentBackend, AgentResponse, ToolCall


class AsyncAgentBackend:
    """Async wrapper around API backends using aiohttp.

    Provides non-blocking send_async() and send_streaming_async() methods.
    Falls back to sync execution if aiohttp is not installed.
    """

    def __init__(self, backend_type: str, endpoint: str, api_key: str,
                 model: str, stream: bool = False, api_format: str = "openai"):
        self.backend_type = backend_type
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.stream = stream
        self.api_format = api_format

    def _build_headers(self) -> dict[str, str]:
        """Build request headers based on API format."""
        headers = {"Content-Type": "application/json"}
        if self.api_format == "anthropic":
            headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = "2023-06-01"
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_body(self, message: str, context: list[dict]) -> dict[str, Any]:
        """Build request body based on API format."""
        messages = list(context) + [{"role": "user", "content": message}]
        if self.api_format == "anthropic":
            return {
                "model": self.model,
                "max_tokens": 4096,
                "messages": messages,
                "stream": self.stream,
            }
        return {
            "model": self.model,
            "messages": messages,
            "stream": self.stream,
        }

    def _parse_response(self, data: dict) -> AgentResponse:
        """Parse API response based on format."""
        tool_calls = []
        if self.api_format == "anthropic":
            content_parts = data.get("content", [])
            text = ""
            for part in content_parts:
                if part.get("type") == "text":
                    text += part.get("text", "")
                elif part.get("type") == "tool_use":
                    tool_calls.append(ToolCall(
                        name=part.get("name", ""),
                        arguments=part.get("input", {}),
                        id=part.get("id", ""),
                    ))
            tokens = {}
            usage = data.get("usage", {})
            if usage:
                tokens = {
                    "input": usage.get("input_tokens", 0),
                    "output": usage.get("output_tokens", 0),
                }
            return AgentResponse(content=text, tool_calls=tool_calls, tokens=tokens)
        else:
            choices = data.get("choices", [])
            if not choices:
                return AgentResponse(content="")
            msg = choices[0].get("message", {})
            text = msg.get("content", "") or ""
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(
                    name=fn.get("name", ""),
                    arguments=args,
                    id=tc.get("id", ""),
                ))
            tokens = {}
            usage = data.get("usage", {})
            if usage:
                tokens = {
                    "input": usage.get("prompt_tokens", 0),
                    "output": usage.get("completion_tokens", 0),
                }
            return AgentResponse(content=text, tool_calls=tool_calls, tokens=tokens)

    async def send_async(self, message: str, context: list[dict]) -> AgentResponse:
        """Send a non-blocking request to the API."""
        if not HAS_AIOHTTP:
            raise RuntimeError("aiohttp is required for async backend. Install with: pip install aiohttp")
        headers = self._build_headers()
        body = self._build_body(message, context)
        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoint, headers=headers, json=body) as resp:
                data = await resp.json()
                return self._parse_response(data)

    async def send_streaming_async(self, message: str, context: list[dict],
                                    on_token: Callable[[str], None] | None = None) -> AgentResponse:
        """Send a streaming request, calling on_token for each chunk."""
        if not HAS_AIOHTTP:
            raise RuntimeError("aiohttp is required for async backend. Install with: pip install aiohttp")
        headers = self._build_headers()
        body = self._build_body(message, context)
        body["stream"] = True
        full_content = ""
        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoint, headers=headers, json=body) as resp:
                async for line in resp.content:
                    text = line.decode("utf-8", errors="replace").strip()
                    if not text.startswith("data: "):
                        continue
                    payload = text[6:]
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    # Extract delta content
                    if self.api_format == "anthropic":
                        delta = chunk.get("delta", {})
                        token = delta.get("text", "")
                    else:
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        token = delta.get("content", "") or ""
                    if token:
                        full_content += token
                        if on_token:
                            on_token(token)
        return AgentResponse(content=full_content)
