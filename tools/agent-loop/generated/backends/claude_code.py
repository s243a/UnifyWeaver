"""Claude Code CLI backend using print mode"""

import subprocess
from .base import AgentBackend, AgentResponse

class ClaudeCodeBackend(AgentBackend):
    """CLI backend using claude command."""

    def __init__(self, command: str = "claude"):
        self.command = command

    def send_message(self, message: str, context: list[dict]) -> AgentResponse:
        raise NotImplementedError("See prototype for full implementation")
