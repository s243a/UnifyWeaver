"""Claude Code CLI backend using print mode"""

import json
import os
import subprocess
from .base import AgentBackend, AgentResponse, ToolCall


class ClaudeCodeBackend(AgentBackend):
    """Claude Code CLI backend with streaming JSON output."""

    def __init__(self, command: str = "claude", model: str = "sonnet"):
        self.command = command
        self.model = model
        self._on_status = None

    def send_message(self, message: str, context: list[dict], **kwargs) -> AgentResponse:
        """Send message to Claude Code CLI with live streaming output."""
        prompt = self._format_prompt(message, context)
        self._on_status = kwargs.get('on_status')

        cmd = [
            self.command,
            "-p",
            "--verbose",
            "--output-format", "stream-json",
            "--model", self.model,
            prompt
        ]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                stdin=subprocess.DEVNULL
            )

            content_parts = []
            tokens = {}
            tool_count = 0
            last_tool_desc = None

            for line in proc.stdout:
                line = line.strip()
                if not line or not line.startswith('{'):
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                etype = event.get('type', '')

                if etype == 'assistant':
                    # Assistant messages contain content blocks
                    message_data = event.get('message', {})
                    content_blocks = message_data.get('content', [])

                    for block in content_blocks:
                        btype = block.get('type', '')

                        if btype == 'text':
                            content_parts.append(block.get('text', ''))

                        elif btype == 'tool_use':
                            tool_count += 1
                            tool_name = block.get('name', '?')
                            tool_input = block.get('input', {})
                            last_tool_desc = self._describe_tool_call(tool_name, tool_input)
                            if self._on_status:
                                self._on_status(f"[{tool_count}] {last_tool_desc}")

                elif etype == 'user':
                    # Tool result — include tool description so it's visible
                    if self._on_status and tool_count > 0:
                        self._on_status(f"[{tool_count}] {last_tool_desc} done")

                elif etype == 'result':
                    # Final result with usage stats
                    usage = event.get('usage', {})
                    tokens = {
                        'input': usage.get('input_tokens', 0),
                        'output': usage.get('output_tokens', 0),
                    }
                    cache_read = usage.get('cache_read_input_tokens', 0)
                    cache_create = usage.get('cache_creation_input_tokens', 0)
                    if cache_read:
                        tokens['cache_read'] = cache_read
                    if cache_create:
                        tokens['cache_create'] = cache_create

                    # Use result text if we didn't collect content from events
                    if not content_parts:
                        result_text = event.get('result', '')
                        if result_text:
                            content_parts.append(result_text)

            proc.wait(timeout=10)
            content = ''.join(content_parts).strip()

            if not content and proc.returncode != 0:
                stderr = proc.stderr.read()
                content = f"[Error: {stderr.strip()}]"

        except subprocess.TimeoutExpired:
            proc.kill()
            return AgentResponse(content="[Error: Command timed out]", tokens={})
        except FileNotFoundError:
            return AgentResponse(
                content=f"[Error: Command '{self.command}' not found. Install with: npm install -g @anthropic-ai/claude-code]",
                tokens={})
        except Exception as e:
            return AgentResponse(content=f"[Error: {e}]", tokens={})

        return AgentResponse(
            content=content,
            tool_calls=[],
            tokens=tokens,
        )

    def _describe_tool_call(self, tool_name: str, params: dict) -> str:
        """Create a short description of a tool call."""
        if tool_name == 'Read':
            return f"reading {os.path.basename(params.get('file_path', '?'))}"
        elif tool_name == 'Glob':
            return f"searching {params.get('pattern', '?')}"
        elif tool_name == 'Grep':
            return f"grep {params.get('pattern', '?')}"
        elif tool_name == 'Bash':
            cmd = params.get('command', '?')
            if len(cmd) > 72:
                cmd = cmd[:69] + '...'
            return f"$ {cmd}"
        elif tool_name == 'Write':
            return f"writing {os.path.basename(params.get('file_path', '?'))}"
        elif tool_name == 'Edit':
            return f"editing {os.path.basename(params.get('file_path', '?'))}"
        elif tool_name == 'Task':
            return f"agent: {params.get('description', '?')}"
        elif tool_name == 'WebFetch':
            return f"fetching {params.get('url', '?')}"
        elif tool_name == 'WebSearch':
            return f"searching: {params.get('query', '?')}"
        else:
            return tool_name

    def _format_prompt(self, message: str, context: list[dict]) -> str:
        """Format message with conversation context."""
        if not context:
            return message

        history_lines = []
        for msg in context[-6:]:
            role = "User" if msg.get('role') == 'user' else "Assistant"
            content = msg.get('content', '')
            if len(content) > 500:
                content = content[:500] + "..."
            history_lines.append(f"{role}: {content}")

        history = "\n".join(history_lines)

        return f"""Previous conversation:
{history}

Current request: {message}"""

    @property
    def name(self) -> str:
        return f"Claude Code ({self.model})"
