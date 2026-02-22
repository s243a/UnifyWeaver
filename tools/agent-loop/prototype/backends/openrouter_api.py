"""OpenRouter API backend using urllib (no pip dependencies)."""

import json
import os
import sys
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from .base import AgentBackend, AgentResponse, ToolCall


# Default tool schemas for function calling (OpenAI format)
DEFAULT_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute"
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative file path to read"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write content to a file (creates or overwrites).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to write to"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Edit a file by replacing a unique string with a new string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to edit"
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact string to find (must be unique in the file)"
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement string"
                    }
                },
                "required": ["path", "old_string", "new_string"]
            }
        }
    },
]


class OpenRouterBackend(AgentBackend):
    """OpenRouter API backend (OpenAI-compatible, no pip deps)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        system_prompt: str | None = None,
        tools: list[dict] | None = None,
    ):
        self.api_key = api_key
        self.model = model or 'moonshotai/kimi-k2.5'
        self.base_url = base_url or 'https://openrouter.ai/api/v1'
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt or (
            "You are a helpful AI coding assistant. "
            "Answer questions directly and concisely. "
            "When asked to perform tasks, use the available tools."
        )
        self.tool_schemas = tools  # None = no tools, [] = explicit empty

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY, "
                "provide --api-key, or add api_key to uwsal.json / coro.json."
            )

    def send_message(self, message: str, context: list[dict], **kwargs) -> AgentResponse:
        """Send message to OpenRouter API."""
        on_status = kwargs.get('on_status')

        # Build messages array
        messages = [{"role": "system", "content": self.system_prompt}]

        for msg in context:
            role = msg.get('role')
            if role == 'tool':
                messages.append({
                    "role": "tool",
                    "tool_call_id": msg.get('tool_call_id', ''),
                    "content": msg['content'],
                })
            elif role == 'assistant' and msg.get('tool_calls'):
                messages.append({
                    "role": "assistant",
                    "content": msg.get('content') or '',
                    "tool_calls": msg['tool_calls'],
                })
            elif role in ('user', 'assistant'):
                messages.append({"role": role, "content": msg['content']})

        # Add current message (only if non-empty and not already in context)
        if message and (not context or context[-1].get('content') != message):
            messages.append({"role": "user", "content": message})

        # Build request body
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        if self.tool_schemas:
            body["tools"] = self.tool_schemas
            body["tool_choice"] = "auto"

        # Make API request
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        req_data = json.dumps(body).encode('utf-8')

        req = Request(
            url,
            data=req_data,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}',
                'HTTP-Referer': 'https://github.com/s243a/UnifyWeaver',
                'X-Title': 'UnifyWeaver Agent Loop',
            },
            method='POST'
        )

        if on_status:
            on_status("Waiting for response...")

        try:
            with urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode('utf-8'))
        except HTTPError as e:
            error_body = e.read().decode('utf-8', errors='replace')
            try:
                err_data = json.loads(error_body)
                err_msg = err_data.get('error', {}).get('message', error_body[:200])
            except json.JSONDecodeError:
                err_msg = error_body[:200]
            return AgentResponse(
                content=f"[API Error {e.code}: {err_msg}]",
                tokens={}
            )
        except URLError as e:
            return AgentResponse(
                content=f"[Network Error: {e.reason}]",
                tokens={}
            )
        except Exception as e:
            return AgentResponse(
                content=f"[Error: {e}]",
                tokens={}
            )

        # Parse response
        content = ""
        tool_calls = []
        tokens = {}

        if data.get('choices'):
            choice = data['choices'][0]
            msg = choice.get('message', {})

            # Text content
            content = msg.get('content') or ''

            # Tool calls
            for tc in msg.get('tool_calls', []):
                if tc.get('type') == 'function':
                    func = tc.get('function', {})
                    try:
                        arguments = json.loads(func.get('arguments', '{}'))
                    except json.JSONDecodeError:
                        arguments = {"raw": func.get('arguments', '')}
                    tool_calls.append(ToolCall(
                        name=func.get('name', ''),
                        arguments=arguments,
                        id=tc.get('id', '')
                    ))
                    if on_status:
                        desc = self._describe_tool_call(
                            func.get('name', '?'), arguments)
                        on_status(f"[{len(tool_calls)}] {desc}")

        # Token usage
        usage = data.get('usage', {})
        if usage:
            tokens = {
                'input': usage.get('prompt_tokens', 0),
                'output': usage.get('completion_tokens', 0),
                'total': usage.get('total_tokens', 0),
            }

        return AgentResponse(
            content=content,
            tool_calls=tool_calls,
            tokens=tokens,
            raw=data
        )

    def supports_streaming(self) -> bool:
        return True

    def send_message_streaming(self, message: str, context: list[dict],
                               on_token=None, **kwargs) -> AgentResponse:
        """Send message with SSE streaming. Tokens arrive via on_token callback."""
        # Build messages array (same as non-streaming)
        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in context:
            role = msg.get('role')
            if role == 'tool':
                messages.append({
                    "role": "tool",
                    "tool_call_id": msg.get('tool_call_id', ''),
                    "content": msg['content'],
                })
            elif role == 'assistant' and msg.get('tool_calls'):
                messages.append({
                    "role": "assistant",
                    "content": msg.get('content') or '',
                    "tool_calls": msg['tool_calls'],
                })
            elif role in ('user', 'assistant'):
                messages.append({"role": role, "content": msg['content']})
        if message and (not context or context[-1].get('content') != message):
            messages.append({"role": "user", "content": message})

        # Build request body with stream=True
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }
        if self.tool_schemas:
            body["tools"] = self.tool_schemas
            body["tool_choice"] = "auto"

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        req = Request(
            url,
            data=json.dumps(body).encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}',
                'HTTP-Referer': 'https://github.com/s243a/UnifyWeaver',
                'X-Title': 'UnifyWeaver Agent Loop',
            },
            method='POST'
        )

        content = ""
        tool_calls_acc = {}  # index -> {id, name, arguments}
        tokens = {}

        try:
            with urlopen(req, timeout=300) as resp:
                for raw_line in resp:
                    line = raw_line.decode('utf-8', errors='replace').rstrip('\n\r')
                    if not line.startswith('data: '):
                        continue
                    payload = line[6:]
                    if payload == '[DONE]':
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get('choices', [])
                    if not choices:
                        # Usage-only chunk (some providers send this)
                        if chunk.get('usage'):
                            usage = chunk['usage']
                            tokens = {
                                'input': usage.get('prompt_tokens', 0),
                                'output': usage.get('completion_tokens', 0),
                                'total': usage.get('total_tokens', 0),
                            }
                        continue

                    delta = choices[0].get('delta', {})

                    # Text content
                    text = delta.get('content')
                    if text:
                        content += text
                        if on_token:
                            on_token(text)

                    # Tool call deltas — accumulate by index
                    for tc_delta in delta.get('tool_calls', []):
                        idx = tc_delta.get('index', 0)
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                'id': '', 'name': '', 'arguments': ''
                            }
                        acc = tool_calls_acc[idx]
                        if tc_delta.get('id'):
                            acc['id'] = tc_delta['id']
                        func = tc_delta.get('function', {})
                        if func.get('name'):
                            acc['name'] = func['name']
                        if func.get('arguments'):
                            acc['arguments'] += func['arguments']

                    # Usage in final chunk
                    if chunk.get('usage'):
                        usage = chunk['usage']
                        tokens = {
                            'input': usage.get('prompt_tokens', 0),
                            'output': usage.get('completion_tokens', 0),
                            'total': usage.get('total_tokens', 0),
                        }

        except HTTPError as e:
            error_body = e.read().decode('utf-8', errors='replace')
            try:
                err_data = json.loads(error_body)
                err_msg = err_data.get('error', {}).get('message', error_body[:200])
            except json.JSONDecodeError:
                err_msg = error_body[:200]
            return AgentResponse(content=f"[API Error {e.code}: {err_msg}]", tokens={})
        except URLError as e:
            return AgentResponse(content=f"[Network Error: {e.reason}]", tokens={})
        except Exception as e:
            return AgentResponse(content=f"[Error: {e}]", tokens={})

        # Convert accumulated tool call deltas to ToolCall objects
        tool_calls = []
        for acc in tool_calls_acc.values():
            try:
                arguments = json.loads(acc['arguments']) if acc['arguments'] else {}
            except json.JSONDecodeError:
                arguments = {"raw": acc['arguments']}
            tool_calls.append(ToolCall(
                name=acc['name'],
                arguments=arguments,
                id=acc['id']
            ))

        return AgentResponse(
            content=content,
            tool_calls=tool_calls,
            tokens=tokens,
        )

    def _describe_tool_call(self, tool_name: str, params: dict) -> str:
        """Create a short description of a tool call."""
        if tool_name == 'read':
            return f"reading {os.path.basename(params.get('path', '?'))}"
        elif tool_name == 'write':
            return f"writing {os.path.basename(params.get('path', '?'))}"
        elif tool_name == 'edit':
            return f"editing {os.path.basename(params.get('path', '?'))}"
        elif tool_name == 'bash':
            cmd = params.get('command', '?')
            if len(cmd) > 72:
                cmd = cmd[:69] + '...'
            return f"$ {cmd}"
        return tool_name

    @property
    def name(self) -> str:
        return f"OpenRouter ({self.model})"
