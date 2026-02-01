"""Coro-code CLI backend using single-task mode"""

import subprocess
from .base import AgentBackend, AgentResponse

class CoroBackend(AgentBackend):
    """CLI backend using claude command."""

    def __init__(self, command: str = "claude"):
        self.command = command

    def send_message(self, message: str, context: list[dict]) -> AgentResponse:
        # Implementation in coro.py
        raise NotImplementedError("See coro.py for full implementation")
