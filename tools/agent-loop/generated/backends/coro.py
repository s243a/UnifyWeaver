"""Coro-code CLI backend using single-task mode."""

import subprocess
import re
from .base import AgentBackend, AgentResponse, ToolCall


class CoroBackend(AgentBackend):
    """Coro-code CLI backend using single-task mode."""

    def __init__(self, command: str = "claude", verbose: bool = True):
        self.command = command
        self.verbose = verbose
        # Patterns for parsing coro output
        self.token_pattern = re.compile(
            r'(?:Input|Output|Total):\s*([\d,]+)\s*tokens?',
            re.IGNORECASE
        )

    def send_message(self, message: str, context: list[dict], **kwargs) -> AgentResponse:
        """Send message to coro CLI and get response."""
        # Format the prompt with context
        prompt = self._format_prompt(message, context)

        # Build command
        cmd = [self.command]
        if self.verbose:
            cmd.append('--verbose')
        cmd.append(prompt)

        # Run coro in single-task mode
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            output = result.stdout
            if result.stderr:
                output += "\n" + result.stderr

        except subprocess.TimeoutExpired:
            return AgentResponse(
                content="[Error: Command timed out after 5 minutes]",
                tokens={}
            )
        except FileNotFoundError:
            return AgentResponse(
                content=f"[Error: Command '{self.command}' not found]",
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
        tool_calls = self.parse_tool_calls(output)

        return AgentResponse(
            content=content,
            tool_calls=tool_calls,
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
        """Clean up coro output, removing noise."""
        lines = output.split('\n')
        cleaned = []
        blank_count = 0

        for line in lines:
            # Skip token report lines
            if self.token_pattern.search(line):
                continue
            # Skip common noise patterns
            if line.strip() in ['', '...', '─' * 10]:
                blank_count += 1
                if blank_count <= 1:
                    cleaned.append(line)
                continue

            blank_count = 0
            cleaned.append(line)

        # Remove leading/trailing blank lines
        result = '\n'.join(cleaned).strip()

        # Remove common assistant prefixes from claude output
        for prefix in ['A:', 'Assistant:', 'Claude:']:
            if result.startswith(prefix):
                result = result[len(prefix):].strip()
                break

        return result

    def _parse_tokens(self, output: str) -> dict[str, int]:
        """Extract token counts from output."""
        tokens = {}
        for match in self.token_pattern.finditer(output):
            # Get the type (Input/Output/Total) from context
            line = output[max(0, match.start()-20):match.end()]
            count = int(match.group(1).replace(',', ''))

            if 'input' in line.lower():
                tokens['input'] = tokens.get('input', 0) + count
            elif 'output' in line.lower():
                tokens['output'] = tokens.get('output', 0) + count
            elif 'total' in line.lower():
                tokens['total'] = count

        return tokens

    def parse_tool_calls(self, response: str) -> list[ToolCall]:
        """Extract tool calls from coro output.

        Note: This is a placeholder - actual implementation depends
        on how coro formats tool calls in single-task mode.
        """
        # TODO: Parse actual tool call format from coro
        return []

    @property
    def name(self) -> str:
        return f"Coro ({self.command})"
