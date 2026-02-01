"""OpenAI API backend using the OpenAI SDK."""

import os
from .base import AgentBackend, AgentResponse, ToolCall

# Try to import openai, but don't fail if not installed
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class OpenAIBackend(AgentBackend):
    """OpenAI API backend (GPT-4, GPT-3.5, etc.)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        max_tokens: int = 4096,
        system_prompt: str | None = None,
        base_url: str | None = None
    ):
        if not HAS_OPENAI:
            raise ImportError(
                "openai package not installed. "
                "Install with: pip install openai"
            )

        # Get API key from parameter or environment
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or (
            "You are a helpful AI assistant. Be concise and direct."
        )
        self.base_url = base_url  # For OpenAI-compatible APIs

        # Initialize client
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = openai.OpenAI(**client_kwargs)

    def send_message(self, message: str, context: list[dict]) -> AgentResponse:
        """Send message to OpenAI API and get response."""
        # Build messages array
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

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
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=messages
            )

            # Extract content
            content = ""
            if response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                if choice.message and choice.message.content:
                    content = choice.message.content

            # Extract token usage
            tokens = {}
            if response.usage:
                tokens = {
                    'input': response.usage.prompt_tokens,
                    'output': response.usage.completion_tokens,
                    'total': response.usage.total_tokens
                }

            # Extract tool calls
            tool_calls = self._extract_tool_calls(response)

            return AgentResponse(
                content=content,
                tool_calls=tool_calls,
                tokens=tokens,
                raw=response
            )

        except openai.APIError as e:
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
        """Extract tool calls from response."""
        tool_calls = []

        if not response.choices or len(response.choices) == 0:
            return tool_calls

        choice = response.choices[0]
        if not choice.message or not choice.message.tool_calls:
            return tool_calls

        for tc in choice.message.tool_calls:
            if tc.type == "function":
                import json
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": tc.function.arguments}

                tool_calls.append(ToolCall(
                    name=tc.function.name,
                    arguments=arguments,
                    id=tc.id
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
        # Build messages array
        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in context:
            if msg.get('role') in ('user', 'assistant'):
                messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
        messages.append({"role": "user", "content": message})

        try:
            content_parts = []

            stream = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True}
            )

            input_tokens = 0
            output_tokens = 0

            for chunk in stream:
                # Handle content chunks
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        content_parts.append(delta.content)
                        if on_token:
                            on_token(delta.content)

                # Handle usage (comes at the end)
                if chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens

            content = "".join(content_parts)
            return AgentResponse(
                content=content,
                tool_calls=[],
                tokens={'input': input_tokens, 'output': output_tokens},
                raw=None
            )

        except openai.APIError as e:
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
        """OpenAI API supports streaming."""
        return True

    @property
    def name(self) -> str:
        return f"OpenAI ({self.model})"
