"""OpenRouter API backend with model routing"""

import os
import requests
from .base import AgentBackend, AgentResponse

class OpenRouterBackend(AgentBackend):
    """API backend for https://openrouter.ai/api/v1/chat/completions"""

    ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
    DEFAULT_MODEL = "anthropic/claude-sonnet-4-20250514"

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model or self.DEFAULT_MODEL

    def send_message(self, message: str, context: list[dict]) -> AgentResponse:
        raise NotImplementedError("See prototype for full implementation")
