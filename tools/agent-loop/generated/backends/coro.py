"""Coro-code CLI backend using single-task mode."""

import json as _json
import os
import subprocess
import re
import tempfile
from .base import AgentBackend, AgentResponse, ToolCall


class CoroBackend(AgentBackend):
    """Coro-code CLI backend using single-task mode."""

    def __init__(self, command: str = "coro", verbose: bool = True,
                 debug: bool = True, max_steps: int = 5,
                 config: str | None = None, no_fallback: bool = False,
                 max_context_tokens: int = 0):
        self.command = command
        self.verbose = verbose
        self.debug = debug
        self.max_steps = max_steps
        self.config = config or self._find_config(no_fallback=no_fallback)
        self.max_context_tokens = max_context_tokens  # 0 = use coro's default
        self._temp_config = None  # temp config file with max_token override
        self.model = self._read_model_from_config()

        # If max_context_tokens is set, create a temp config with max_token
        if self.max_context_tokens > 0:
            self._temp_config = self._create_limited_config()
        # Patterns for parsing coro output
        self.token_pattern = re.compile(
            r'(?:Input|Output|Total):\s*([\d,]+)\s*tokens?',
            re.IGNORECASE
        )
        # Coro debug format: "Tokens: 3639 input + 80 output = 3719 total"
        self._coro_token_pattern = re.compile(
            r'Tokens:\s*(\d+)\s*input\s*\+\s*(\d+)\s*output\s*=\s*(\d+)\s*total'
        )
        # Duration: "Duration: 2.26s"
        self._duration_pattern = re.compile(r'Duration:\s*([\d.]+)s')
        # Pattern for ANSI escape codes (colors, cursor movement, etc.)
        self._ansi_pattern = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')

    def _find_config(self, no_fallback: bool = False) -> str | None:
        """Find coro config, checking CWD then home directory."""
        # Check current directory first
        if os.path.isfile('coro.json'):
            return None  # coro will find it itself
        if no_fallback:
            return None
        # Check home directory as fallback
        home_config = os.path.expanduser('~/coro.json')
        if os.path.isfile(home_config):
            return home_config
        return None

    def _read_coro_config(self) -> dict:
        """Read the full coro config from the best available path."""
        for path in filter(None, [
            self.config,
            'coro.json' if os.path.isfile('coro.json') else None,
            os.path.expanduser('~/coro.json'),
        ]):
            try:
                with open(path) as f:
                    return _json.load(f)
            except Exception:
                continue
        return {}

    def _read_model_from_config(self) -> str:
        """Read model name from coro config file."""
        return self._read_coro_config().get('model', 'unknown')

    def _create_limited_config(self) -> str | None:
        """Create a temp coro.json with max_token set to limit context."""
        base = self._read_coro_config()
        if not base:
            return None
        base['max_token'] = self.max_context_tokens
        try:
            tf = tempfile.NamedTemporaryFile(
                mode='w', suffix='.json', prefix='coro_',
                delete=False
            )
            _json.dump(base, tf, indent=2)
            tf.close()
            return tf.name
        except Exception:
            return None

    def send_message(self, message: str, context: list[dict], **kwargs) -> AgentResponse:
        """Send message to coro CLI and get response."""
        prompt = self._format_prompt(message, context)

        # Build command
        cmd = [self.command]
        # Use temp config (with max_token) if available, else original
        config_path = self._temp_config or self.config
        if config_path:
            cmd.extend(['-c', config_path])
        if self.debug:
            cmd.append('--debug')
        elif self.verbose:
            cmd.append('--verbose')
        if self.max_steps > 0:
            cmd.extend(['--max-steps', str(self.max_steps)])
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

    def _strip_ansi(self, text: str) -> str:
        """Remove ANSI escape codes from text."""
        return self._ansi_pattern.sub('', text)

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
            plain = self._strip_ansi(line)
            # Skip token report lines
            if self.token_pattern.search(plain):
                continue
            if self._coro_token_pattern.search(plain):
                continue
            if self._duration_pattern.search(plain):
                continue
            # Skip verbose/debug log lines (timestamps like 2026-...)
            if re.match(r'^\s*\d{4}-\d{2}-\d{2}T', plain):
                continue
            stripped = plain.strip()
            # Skip residual cursor control fragments after ANSI stripping
            if stripped and all(c in '[0123456789ABCDJKHfm;' for c in stripped):
                continue
            if stripped in ['Previous conversation:', 'Current request:'] or \
               stripped.startswith('Current request: ') or \
               stripped.startswith('User: ') or \
               stripped.startswith('Assistant: '):
                continue
            # Skip common noise patterns
            if stripped in ['', '...', '─' * 10]:
                blank_count += 1
                if blank_count <= 1:
                    cleaned.append(line)
                continue

            blank_count = 0
            cleaned.append(line)

        # Remove leading/trailing blank lines
        result = '\n'.join(cleaned).strip()
        # Strip any remaining ANSI codes
        result = self._strip_ansi(result)

        # Remove common assistant prefixes from claude output
        for prefix in ['A:', 'Assistant:', 'Claude:']:
            if result.startswith(prefix):
                result = result[len(prefix):].strip()
                break

        return result

    def _parse_tokens(self, output: str) -> dict[str, int]:
        """Extract token counts from output."""
        plain = self._strip_ansi(output)
        tokens = {}

        # Try coro debug format first: "Tokens: N input + N output = N total"
        match = self._coro_token_pattern.search(plain)
        if match:
            tokens['input'] = int(match.group(1))
            tokens['output'] = int(match.group(2))
            tokens['total'] = int(match.group(3))
            # Also extract duration if available
            dur = self._duration_pattern.search(plain)
            if dur:
                tokens['duration'] = float(dur.group(1))
            return tokens

        # Fall back to generic pattern
        for match in self.token_pattern.finditer(plain):
            line = plain[max(0, match.start()-20):match.end()]
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
