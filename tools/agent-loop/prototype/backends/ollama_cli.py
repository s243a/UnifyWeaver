"""Ollama CLI backend using the ollama run command."""

import subprocess
from .base import AgentBackend, AgentResponse, ToolCall


class OllamaCLIBackend(AgentBackend):
    """Ollama CLI backend using 'ollama run' command."""

    def __init__(
        self,
        command: str = "ollama",
        model: str = "llama3",
        timeout: int = 300
    ):
        self.command = command
        self.model = model
        self.timeout = timeout

    def send_message(self, message: str, context: list[dict], **kwargs) -> AgentResponse:
        """Send message to Ollama CLI and get response."""
        # Format the prompt with context
        prompt = self._format_prompt(message, context)

        # Build command: ollama run <model> "<prompt>"
        # Note: ollama run reads from stdin if no prompt given
        cmd = [self.command, "run", self.model]

        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            output = result.stdout
            if result.returncode != 0 and result.stderr:
                output = f"[Error: {result.stderr.strip()}]"

        except subprocess.TimeoutExpired:
            return AgentResponse(
                content=f"[Error: Command timed out after {self.timeout} seconds]",
                tokens={}
            )
        except FileNotFoundError:
            return AgentResponse(
                content=f"[Error: Command '{self.command}' not found. Install Ollama from https://ollama.ai]",
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
            tokens={},  # CLI doesn't report tokens
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

        # Remove common prefixes
        for prefix in ['A:', 'Assistant:']:
            if result.startswith(prefix):
                result = result[len(prefix):].strip()
                break

        return result

    def list_models(self) -> list[str]:
        """List available models using 'ollama list'."""
        try:
            result = subprocess.run(
                [self.command, "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                return [line.split()[0] for line in lines if line.strip()]
        except Exception:
            pass
        return []

    @property
    def name(self) -> str:
        return f"Ollama CLI ({self.model})"
