"""Ollama REST API backend for local models"""

import os
import requests
from .base import AgentBackend, AgentResponse

class OllamaApiBackend(AgentBackend):
    """API backend for http://localhost:11434/api/chat"""

    ENDPOINT = "http://localhost:11434/api/chat"
    DEFAULT_MODEL = "llama3"

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL

    def send_message(self, message: str, context: list[dict]) -> AgentResponse:
        # Implementation varies by API
        raise NotImplementedError("Implement for specific API")
