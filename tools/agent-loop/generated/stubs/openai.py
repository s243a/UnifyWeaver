"""OpenAI API backend"""

import os
import requests
from .base import AgentBackend, AgentResponse

class OpenaiBackend(AgentBackend):
    """API backend for https://api.openai.com/v1/chat/completions"""

    ENDPOINT = "https://api.openai.com/v1/chat/completions"
    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model or self.DEFAULT_MODEL

    def send_message(self, message: str, context: list[dict]) -> AgentResponse:
        # Implementation varies by API
        raise NotImplementedError("Implement for specific API")
