"""Tool handler for executing agent tool calls."""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from backends.base import ToolCall


@dataclass
class ToolResult:
    """Result of executing a tool."""
    success: bool
    output: str
    tool_name: str


class ToolHandler:
    """Handles execution of tool calls from agent responses."""

    def __init__(
        self,
        allowed_tools: list[str] | None = None,
        confirm_destructive: bool = True,
        working_dir: str | None = None
    ):
        self.allowed_tools = allowed_tools or ['bash', 'read', 'write', 'edit']
        self.confirm_destructive = confirm_destructive
        self.working_dir = working_dir or os.getcwd()

        # Tool implementations
        self.tools = {
            'bash': self._execute_bash,
            'read': self._read_file,
            'write': self._write_file,
            'edit': self._edit_file,
        }

        # Tools that require confirmation
        self.destructive_tools = {'bash', 'write', 'edit'}

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call and return the result."""
        if tool_call.name not in self.allowed_tools:
            return ToolResult(
                success=False,
                output=f"Tool '{tool_call.name}' is not allowed",
                tool_name=tool_call.name
            )

        if tool_call.name not in self.tools:
            return ToolResult(
                success=False,
                output=f"Unknown tool: {tool_call.name}",
                tool_name=tool_call.name
            )

        # Check for confirmation if destructive
        if self.confirm_destructive and tool_call.name in self.destructive_tools:
            if not self._confirm_execution(tool_call):
                return ToolResult(
                    success=False,
                    output="User declined to execute tool",
                    tool_name=tool_call.name
                )

        try:
            return self.tools[tool_call.name](tool_call.arguments)
        except Exception as e:
            return ToolResult(
                success=False,
                output=f"Error executing {tool_call.name}: {e}",
                tool_name=tool_call.name
            )

    def _confirm_execution(self, tool_call: ToolCall) -> bool:
        """Ask user to confirm execution of a destructive tool."""
        print(f"\n[Tool: {tool_call.name}]")

        if tool_call.name == 'bash':
            cmd = tool_call.arguments.get('command', '')
            print(f"  Command: {cmd}")
        elif tool_call.name == 'write':
            path = tool_call.arguments.get('path', '')
            print(f"  File: {path}")
        elif tool_call.name == 'edit':
            path = tool_call.arguments.get('path', '')
            old = tool_call.arguments.get('old_string', '')[:50]
            print(f"  File: {path}")
            print(f"  Replace: {old}...")

        try:
            response = input("Execute? [y/N]: ").strip().lower()
            return response in ('y', 'yes')
        except (EOFError, KeyboardInterrupt):
            return False

    def _execute_bash(self, args: dict) -> ToolResult:
        """Execute a bash command."""
        command = args.get('command', '')
        if not command:
            return ToolResult(
                success=False,
                output="No command provided",
                tool_name='bash'
            )

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=self.working_dir
            )

            output = result.stdout
            if result.stderr:
                output += "\n" + result.stderr

            return ToolResult(
                success=result.returncode == 0,
                output=output.strip() or "(no output)",
                tool_name='bash'
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="Command timed out after 120 seconds",
                tool_name='bash'
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=f"Error: {e}",
                tool_name='bash'
            )

    def _read_file(self, args: dict) -> ToolResult:
        """Read a file."""
        path = args.get('path', '')
        if not path:
            return ToolResult(
                success=False,
                output="No path provided",
                tool_name='read'
            )

        # Resolve path relative to working directory
        file_path = Path(self.working_dir) / path
        if path.startswith('/'):
            file_path = Path(path)

        try:
            content = file_path.read_text()
            return ToolResult(
                success=True,
                output=content,
                tool_name='read'
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                output=f"File not found: {path}",
                tool_name='read'
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=f"Error reading file: {e}",
                tool_name='read'
            )

    def _write_file(self, args: dict) -> ToolResult:
        """Write content to a file."""
        path = args.get('path', '')
        content = args.get('content', '')

        if not path:
            return ToolResult(
                success=False,
                output="No path provided",
                tool_name='write'
            )

        # Resolve path
        file_path = Path(self.working_dir) / path
        if path.startswith('/'):
            file_path = Path(path)

        try:
            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            return ToolResult(
                success=True,
                output=f"Wrote {len(content)} bytes to {path}",
                tool_name='write'
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=f"Error writing file: {e}",
                tool_name='write'
            )

    def _edit_file(self, args: dict) -> ToolResult:
        """Edit a file with search/replace."""
        path = args.get('path', '')
        old_string = args.get('old_string', '')
        new_string = args.get('new_string', '')

        if not path:
            return ToolResult(
                success=False,
                output="No path provided",
                tool_name='edit'
            )

        if not old_string:
            return ToolResult(
                success=False,
                output="No old_string provided",
                tool_name='edit'
            )

        # Resolve path
        file_path = Path(self.working_dir) / path
        if path.startswith('/'):
            file_path = Path(path)

        try:
            content = file_path.read_text()

            if old_string not in content:
                return ToolResult(
                    success=False,
                    output=f"old_string not found in {path}",
                    tool_name='edit'
                )

            # Count occurrences
            count = content.count(old_string)
            if count > 1:
                return ToolResult(
                    success=False,
                    output=f"old_string found {count} times - must be unique",
                    tool_name='edit'
                )

            # Perform replacement
            new_content = content.replace(old_string, new_string, 1)
            file_path.write_text(new_content)

            return ToolResult(
                success=True,
                output=f"Edited {path}",
                tool_name='edit'
            )

        except FileNotFoundError:
            return ToolResult(
                success=False,
                output=f"File not found: {path}",
                tool_name='edit'
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=f"Error editing file: {e}",
                tool_name='edit'
            )

    def format_result_for_agent(self, result: ToolResult) -> str:
        """Format a tool result for sending back to the agent."""
        status = "Success" if result.success else "Failed"
        return f"[Tool {result.tool_name} - {status}]\n{result.output}"
