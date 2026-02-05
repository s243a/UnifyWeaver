"""Claude Code CLI backend using print mode."""

import subprocess
import re
from .base import AgentBackend, AgentResponse, ToolCall


class ClaudeCodeBackend(AgentBackend):
    """Claude Code CLI backend using -p (print) mode."""

    def __init__(self, command: str = "claude", model: str = "sonnet"):
        self.command = command
        self.model = model
        # Pattern for token counts if present
        self.token_pattern = re.compile(
            r'(?:Input|Output|Total):\s*([\d,]+)\s*tokens?',
            re.IGNORECASE
        )

    def send_message(self, message: str, context: list[dict], **kwargs) -> AgentResponse:
        """Send message to Claude Code CLI and get response."""
        # Format the prompt with context
        prompt = self._format_prompt(message, context)

        # Build command: claude -p --model <model> <prompt>
        cmd = [
            self.command,
            "-p",  # Print mode (non-interactive)
            "--model", self.model,
            prompt
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            output = result.stdout
            if result.returncode != 0 and result.stderr:
                output = f"[Error: {result.stderr.strip()}]"

        except subprocess.TimeoutExpired:
            return AgentResponse(
                content="[Error: Command timed out after 5 minutes]",
                tokens={}
            )
        except FileNotFoundError:
            return AgentResponse(
                content=f"[Error: Command '{self.command}' not found. Install with: npm install -g @anthropic-ai/claude-code]",
                tokens={}
            )
        except Exception as e:
            return AgentResponse(
                content=f"[Error: {e}]",
                tokens={}
            )

        # Parse the output
        content = self._clean_output(output)
        tokens = self._parse_tokens(output)

        return AgentResponse(
            content=content,
            tool_calls=[],
            tokens=tokens,
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
        """Clean up output, removing noise."""
        lines = output.split('\n')
        cleaned = []

        for line in lines:
            # Skip token report lines
            if self.token_pattern.search(line):
                continue
            cleaned.append(line)

        result = '\n'.join(cleaned).strip()

        # Remove common assistant prefixes
        for prefix in ['A:', 'Assistant:', 'Claude:']:
            if result.startswith(prefix):
                result = result[len(prefix):].strip()
                break

        return result

    def _parse_tokens(self, output: str) -> dict[str, int]:
        """Extract token counts from output if present."""
        tokens = {}
        for match in self.token_pattern.finditer(output):
            line = output[max(0, match.start()-20):match.end()]
            count = int(match.group(1).replace(',', ''))

            if 'input' in line.lower():
                tokens['input'] = count
            elif 'output' in line.lower():
                tokens['output'] = count

        return tokens

    @property
    def name(self) -> str:
        return f"Claude Code ({self.model})"
