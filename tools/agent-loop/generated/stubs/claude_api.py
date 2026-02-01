"""Anthropic Claude API backend"""

import os
import requests
from .base import AgentBackend, AgentResponse

class ClaudeApiBackend(AgentBackend):
    """API backend for https://api.anthropic.com/v1/messages"""

    ENDPOINT = "https://api.anthropic.com/v1/messages"
    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model or self.DEFAULT_MODEL

    def send_message(self, message: str, context: list[dict]) -> AgentResponse:
        # Implementation varies by API
        raise NotImplementedError("Implement for specific API")
