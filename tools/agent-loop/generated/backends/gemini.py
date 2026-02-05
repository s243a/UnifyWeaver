"""Gemini CLI backend using stream-json for live progress."""

import json
import subprocess
import sys
from .base import AgentBackend, AgentResponse, ToolCall


class GeminiBackend(AgentBackend):
    """Gemini CLI backend with streaming JSON output."""

    def __init__(self, command: str = "gemini", model: str = "gemini-3-flash-preview",
                 sandbox: bool = False, approval_mode: str = "yolo",
                 allowed_tools: list[str] | None = None):
        self.command = command
        self.model = model
        self.sandbox = sandbox
        self.approval_mode = approval_mode
        self.allowed_tools = allowed_tools or []
        self._on_status = None  # Callback for status updates

    def send_message(self, message: str, context: list[dict], **kwargs) -> AgentResponse:
        """Send message to Gemini CLI with live streaming output."""
        prompt = self._format_prompt(message, context)
        self._on_status = kwargs.get('on_status')

        # Use stream-json for live progress
        cmd = [
            self.command,
            "-m", self.model,
            "-o", "stream-json",
        ]
        if self.sandbox:
            cmd.append("-s")
        if self.approval_mode:
            cmd.extend(["--approval-mode", self.approval_mode])
        for tool in self.allowed_tools:
            cmd.extend(["--allowed-tools", tool])
        cmd.extend(["-p", prompt])

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

            for line in proc.stdout:
                line = line.strip()
                if not line or not line.startswith('{'):
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                etype = event.get('type', '')

                if etype == 'message' and event.get('role') == 'assistant':
                    text = event.get('content', '')
                    content_parts.append(text)

                elif etype == 'tool_use':
                    tool_count += 1
                    tool_name = event.get('tool_name', '?')
                    params = event.get('parameters', {})
                    # Show what tool is being called
                    desc = self._describe_tool_call(tool_name, params)
                    if self._on_status:
                        self._on_status(f"[{tool_count}] {desc}")

                elif etype == 'tool_result':
                    status = event.get('status', '')
                    if self._on_status:
                        self._on_status(f"[{tool_count}] done ({status})")

                elif etype == 'result':
                    stats = event.get('stats', {})
                    tokens = {
                        'input': stats.get('input_tokens', 0),
                        'output': stats.get('output_tokens', 0),
                    }

            proc.wait(timeout=10)
            content = ''.join(content_parts).strip()

            if not content and proc.returncode != 0:
                stderr = proc.stderr.read()
                content = f"[Error: {stderr.strip()}]"

        except subprocess.TimeoutExpired:
            proc.kill()
            return AgentResponse(
                content="[Error: Command timed out]",
                tokens={}
            )
        except FileNotFoundError:
            return AgentResponse(
                content=f"[Error: Command '{self.command}' not found. Install Gemini CLI.]",
                tokens={}
            )
        except Exception as e:
            return AgentResponse(
                content=f"[Error: {e}]",
                tokens={}
            )

        return AgentResponse(
            content=content,
            tool_calls=[],
            tokens=tokens,
        )

    def _describe_tool_call(self, tool_name: str, params: dict) -> str:
        """Create a short description of a tool call."""
        if tool_name == 'read_file':
            return f"reading {params.get('file_path', '?')}"
        elif tool_name == 'glob':
            return f"searching {params.get('pattern', '?')}"
        elif tool_name == 'grep':
            return f"grep {params.get('pattern', '?')}"
        elif tool_name == 'run_shell_command':
            cmd = params.get('command', '?')
            if len(cmd) > 40:
                cmd = cmd[:37] + '...'
            return f"$ {cmd}"
        elif tool_name == 'write_file':
            return f"writing {params.get('file_path', '?')}"
        elif tool_name == 'edit':
            return f"editing {params.get('file_path', '?')}"
        elif tool_name == 'list_directory':
            return f"ls {params.get('path', '?')}"
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
        return f"Gemini ({self.model})"
