"""Claude API backend using the Anthropic SDK."""

import os
from .base import AgentBackend, AgentResponse, ToolCall

# Try to import anthropic, but don't fail if not installed
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


class ClaudeAPIBackend(AgentBackend):
    """Anthropic Claude API backend."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        system_prompt: str | None = None
    ):
        if not HAS_ANTHROPIC:
            raise ImportError(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            )

        # Get API key from parameter or environment
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError(
                "API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or (
            "You are a helpful AI assistant. Be concise and direct."
        )

        self.client = anthropic.Anthropic(api_key=self.api_key)

    def send_message(self, message: str, context: list[dict]) -> AgentResponse:
        """Send message to Claude API and get response."""
        # Build messages array
        messages = []

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

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                messages=messages
            )

            # Extract content
            content = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    content += block.text

            # Extract token usage
            tokens = {}
            if hasattr(response, 'usage'):
                tokens = {
                    'input': response.usage.input_tokens,
                    'output': response.usage.output_tokens
                }

            # Extract tool use (for future)
            tool_calls = self._extract_tool_calls(response)

            return AgentResponse(
                content=content,
                tool_calls=tool_calls,
                tokens=tokens,
                raw=response
            )

        except anthropic.APIError as e:
            return AgentResponse(
                content=f"[API Error: {e}]",
                tokens={}
            )
        except Exception as e:
            return AgentResponse(
                content=f"[Error: {e}]",
                tokens={}
            )

    def _extract_tool_calls(self, response) -> list[ToolCall]:
        """Extract tool use blocks from response."""
        tool_calls = []

        for block in response.content:
            if hasattr(block, 'type') and block.type == 'tool_use':
                tool_calls.append(ToolCall(
                    name=block.name,
                    arguments=block.input,
                    id=block.id
                ))

        return tool_calls

    def supports_streaming(self) -> bool:
        """Claude API supports streaming."""
        return True

    @property
    def name(self) -> str:
        return f"Claude API ({self.model})"
