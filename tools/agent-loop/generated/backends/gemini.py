"""Gemini CLI backend using print mode."""

import subprocess
from .base import AgentBackend, AgentResponse, ToolCall


class GeminiBackend(AgentBackend):
    """Gemini CLI backend using -p (print) mode."""

    def __init__(self, command: str = "gemini", model: str = "gemini-2.5-flash"):
        self.command = command
        self.model = model

    def send_message(self, message: str, context: list[dict]) -> AgentResponse:
        """Send message to Gemini CLI and get response."""
        # Format the prompt with context
        prompt = self._format_prompt(message, context)

        # Build command: gemini -p <prompt> -m <model> --output-format text
        cmd = [
            self.command,
            "-p", prompt,
            "-m", self.model,
            "--output-format", "text"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            output = result.stdout
            if result.returncode != 0 and result.stderr:
                output = f"[Error: {result.stderr.strip()}]"

        except subprocess.TimeoutExpired:
            return AgentResponse(
                content="[Error: Command timed out after 2 minutes]",
                tokens={}
            )
        except FileNotFoundError:
            return AgentResponse(
                content=f"[Error: Command '{self.command}' not found. Install Gemini CLI from Google.]",
                tokens={}
            )
        except Exception as e:
            return AgentResponse(
                content=f"[Error: {e}]",
                tokens={}
            )

        # Clean the output
        content = self._clean_output(output)

        return AgentResponse(
            content=content,
            tool_calls=[],
            tokens={},  # Gemini CLI doesn't report tokens
            raw=output
        )

    def _format_prompt(self, message: str, context: list[dict]) -> str:
        """Format message with conversation context."""
        if not context:
            return message

        # Format last few messages as context
        history_lines = []
        for msg in context[-6:]:  # Last 6 messages (3 exchanges)
            role = "User" if msg.get('role') == 'user' else "Assistant"
            content = msg.get('content', '')
            # Truncate very long messages in context
            if len(content) > 500:
                content = content[:500] + "..."
            history_lines.append(f"{role}: {content}")

        history = "\n".join(history_lines)

        return f"""Previous conversation:
{history}

Current request: {message}"""

    def _clean_output(self, output: str) -> str:
        """Clean up output."""
        result = output.strip()

        # Remove common assistant prefixes
        for prefix in ['A:', 'Assistant:', 'Gemini:']:
            if result.startswith(prefix):
                result = result[len(prefix):].strip()
                break

        return result

    @property
    def name(self) -> str:
        return f"Gemini ({self.model})"
