"""Ollama REST API backend for local models"""

import json
import urllib.request
import urllib.error
from .base import AgentBackend, AgentResponse, ToolCall


class OllamaAPIBackend(AgentBackend):
    """Ollama REST API backend for local models."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 11434,
        model: str = "llama3",
        system_prompt: str | None = None,
        timeout: int = 300
    ):
        self.host = host
        self.port = port
        self.model = model
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.timeout = timeout
        self.base_url = f"http://{host}:{port}"

    def send_message(self, message: str, context: list[dict], **kwargs) -> AgentResponse:
        """Send message to Ollama API and get response."""
        # Build messages array
        messages = []

        # Add system prompt
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })

        # Add context messages
        for msg in context:
            if msg.get('role') in ('user', 'assistant'):
                messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })

        # Add current message
        messages.append({
            "role": "user",
            "content": message
        })

        # Build request
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }

        try:
            url = f"{self.base_url}/api/chat"
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode('utf-8'))

            content = result.get("message", {}).get("content", "")

            tokens = {}
            if "eval_count" in result:
                tokens["output"] = result["eval_count"]
            if "prompt_eval_count" in result:
                tokens["input"] = result["prompt_eval_count"]

            return AgentResponse(
                content=content,
                tool_calls=[],
                tokens=tokens,
                raw=result
            )

        except urllib.error.URLError as e:
            return AgentResponse(
                content=f"[Error: Cannot connect to Ollama at {self.base_url}. Is Ollama running? Error: {e}]",
                tokens={}
            )
        except json.JSONDecodeError as e:
            return AgentResponse(
                content=f"[Error: Invalid JSON response from Ollama: {e}]",
                tokens={}
            )
        except Exception as e:
            return AgentResponse(
                content=f"[Error: {e}]",
                tokens={}
            )

    def list_models(self) -> list[str]:
        """List available models on the Ollama server."""
        try:
            url = f"{self.base_url}/api/tags"
            req = urllib.request.Request(url, method="GET")

            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))

            return [m["name"] for m in result.get("models", [])]

        except Exception:
            return []

    @property
    def name(self) -> str:
        return f"Ollama API ({self.model}@{self.host}:{self.port})"
