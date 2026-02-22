"""Anthropic Claude API backend"""

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

    def send_message(self, message: str, context: list[dict], **kwargs) -> AgentResponse:
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

        # Note: current message is already in context (added by agent_loop)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                messages=messages
            )

            content = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    content += block.text

            tokens = {}
            if hasattr(response, 'usage'):
                tokens = {
                    'input': response.usage.input_tokens,
                    'output': response.usage.output_tokens
                }

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

    def send_message_streaming(
        self,
        message: str,
        context: list[dict],
        on_token: callable = None
    ) -> AgentResponse:
        """Send message with streaming response.

        Args:
            message: The message to send
            context: Conversation context
            on_token: Callback called for each token chunk (str) -> None
        """
        messages = []
        for msg in context:
            if msg.get('role') in ('user', 'assistant'):
                messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
        # Note: current message is already in context (added by agent_loop)

        try:
            content_parts = []
            input_tokens = 0
            output_tokens = 0

            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                messages=messages
            ) as stream:
                for text in stream.text_stream:
                    content_parts.append(text)
                    if on_token:
                        on_token(text)

                response = stream.get_final_message()
                if hasattr(response, 'usage'):
                    input_tokens = response.usage.input_tokens
                    output_tokens = response.usage.output_tokens

            content = "".join(content_parts)
            return AgentResponse(
                content=content,
                tool_calls=[],
                tokens={'input': input_tokens, 'output': output_tokens},
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

    def supports_streaming(self) -> bool:
        """Claude API supports streaming."""
        return True

    @property
    def name(self) -> str:
        return f"Claude API ({self.model})"
