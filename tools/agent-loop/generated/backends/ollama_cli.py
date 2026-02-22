"""Ollama CLI backend using 'ollama run' command"""

import subprocess
from .base import AgentBackend, AgentResponse

class OllamaCLIBackend(AgentBackend):
    """CLI backend using ollama command."""

    def __init__(self, command: str = "ollama"):
        self.command = command

    def send_message(self, message: str, context: list[dict]) -> AgentResponse:
        raise NotImplementedError("See prototype for full implementation")
