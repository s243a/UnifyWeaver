"""Gemini CLI backend"""

import subprocess
from .base import AgentBackend, AgentResponse

class GeminiBackend(AgentBackend):
    """CLI backend using gemini command."""

    def __init__(self, command: str = "gemini"):
        self.command = command

    def send_message(self, message: str, context: list[dict]) -> AgentResponse:
        raise NotImplementedError("See prototype for full implementation")
