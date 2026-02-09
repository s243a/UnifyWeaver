"""Tool handler for executing agent tool calls."""

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from backends.base import ToolCall


@dataclass
class ToolResult:
    """Result of executing a tool."""
    success: bool
    output: str
    tool_name: str


# ── Path validation ────────────────────────────────────────────────────────

# Sensitive paths that should never be accessed by agent tools
_BLOCKED_PATHS = {
    '/etc/shadow', '/etc/gshadow', '/etc/sudoers',
}
_BLOCKED_PREFIXES = (
    '/proc/', '/sys/', '/dev/',
)
_BLOCKED_HOME_PATTERNS = (
    '.ssh/', '.gnupg/', '.aws/', '.config/gcloud/',
    '.env', '.netrc', '.npmrc',
)


def validate_path(raw_path: str, working_dir: str,
                  extra_blocked: list[str] | None = None,
                  extra_allowed: list[str] | None = None) -> tuple[str, str | None]:
    """Resolve and validate a file path.

    Returns (resolved_path, error_message).
    error_message is None if the path is safe.
    """
    # Resolve relative paths and symlinks
    if raw_path.startswith('/'):
        resolved = os.path.realpath(raw_path)
    else:
        resolved = os.path.realpath(os.path.join(working_dir, raw_path))

    # Check both raw (as typed) and resolved (after symlinks) paths,
    # because on Termux /etc -> /system/etc and ~ -> /data/data/.../home
    raw_abs = os.path.abspath(os.path.join(working_dir, raw_path)) if not raw_path.startswith('/') else raw_path
    check_paths = {resolved, raw_abs}

    # Check allowlist first — explicit allows override blocks
    if extra_allowed:
        for pattern in extra_allowed:
            expanded = os.path.realpath(os.path.expanduser(pattern))
            for p in check_paths:
                if p == expanded or p.startswith(expanded.rstrip('/') + '/'):
                    return resolved, None

    # Check user-provided extra blocks
    if extra_blocked:
        for pattern in extra_blocked:
            expanded = os.path.realpath(os.path.expanduser(pattern))
            for p in check_paths:
                if p == expanded or p.startswith(expanded.rstrip('/') + '/'):
                    return resolved, f"Blocked by config: {raw_path}"

    # Built-in blocks — check both raw and resolved
    for p in check_paths:
        if p in _BLOCKED_PATHS:
            return resolved, f"Blocked: {raw_path} is a sensitive system file"
        for prefix in _BLOCKED_PREFIXES:
            if p.startswith(prefix):
                return resolved, f"Blocked: {raw_path} is in a system directory"

    # Sensitive home directory patterns — check against resolved path
    home = os.path.expanduser('~')
    for p in check_paths:
        if p.startswith(home + '/'):
            rel = p[len(home) + 1:]
            for pattern in _BLOCKED_HOME_PATTERNS:
                if rel == pattern.rstrip('/') or rel.startswith(pattern):
                    return resolved, f"Blocked: ~/{pattern} may contain credentials"

    return resolved, None


# ── Command blocklist ──────────────────────────────────────────────────────

_BLOCKED_COMMAND_PATTERNS = [
    (r'\brm\s+-[rf]*\s+/', "rm with absolute path and force/recursive flags"),
    (r'\bmkfs\b', "filesystem format"),
    (r'\bdd\s+.*of=/dev/', "write to block device"),
    (r'>\s*/dev/sd', "redirect to block device"),
    (r'\bcurl\b.*\|\s*(?:ba)?sh', "pipe remote script to shell"),
    (r'\bwget\b.*\|\s*(?:ba)?sh', "pipe remote script to shell"),
    (r'\bchmod\s+777\b', "world-writable permissions"),
    (r':\(\)\s*\{\s*:\|:\s*&\s*\}\s*;', "fork bomb"),
    (r'\b>\s*/etc/', "overwrite system config"),
]


def is_command_blocked(command: str,
                       extra_blocked: list[str] | None = None,
                       extra_allowed: list[str] | None = None) -> str | None:
    """Return reason if command is blocked, None if allowed."""
    # Check allowlist first
    if extra_allowed:
        for pattern in extra_allowed:
            if re.search(pattern, command, re.IGNORECASE):
                return None

    # Check user-provided extra blocks
    if extra_blocked:
        for pattern in extra_blocked:
            if re.search(pattern, command, re.IGNORECASE):
                return f"Blocked by config: matches '{pattern}'"

    # Built-in blocks
    for pattern, description in _BLOCKED_COMMAND_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return f"Blocked: {description}"
    return None


# ── Security config ────────────────────────────────────────────────────────

@dataclass
class SecurityConfig:
    """Security settings for tool execution."""
    path_validation: bool = True
    command_blocklist: bool = True
    blocked_paths: list[str] = field(default_factory=list)
    blocked_commands: list[str] = field(default_factory=list)
    allowed_paths: list[str] = field(default_factory=list)
    allowed_commands: list[str] = field(default_factory=list)

    @classmethod
    def from_profile(cls, profile: str) -> 'SecurityConfig':
        """Create config from a named profile."""
        if profile == 'open':
            return cls(path_validation=False, command_blocklist=False)
        elif profile == 'cautious':
            return cls(path_validation=True, command_blocklist=True)
        elif profile in ('sandboxed', 'paranoid'):
            return cls(path_validation=True, command_blocklist=True)
        # Default to cautious
        return cls(path_validation=True, command_blocklist=True)


class ToolHandler:
    """Handles execution of tool calls from agent responses."""

    def __init__(
        self,
        allowed_tools: list[str] | None = None,
        confirm_destructive: bool = True,
        working_dir: str | None = None,
        security: SecurityConfig | None = None
    ):
        self.allowed_tools = allowed_tools or ['bash', 'read', 'write', 'edit']
        self.confirm_destructive = confirm_destructive
        self.working_dir = working_dir or os.getcwd()
        self.security = security or SecurityConfig()

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

        # Command blocklist check
        if self.security.command_blocklist:
            reason = is_command_blocked(
                command,
                extra_blocked=self.security.blocked_commands,
                extra_allowed=self.security.allowed_commands,
            )
            if reason:
                return ToolResult(
                    success=False,
                    output=f"[Security] {reason}",
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

    def _validate_file_path(self, raw_path: str, tool_name: str) -> tuple[Path, ToolResult | None]:
        """Validate and resolve a file path. Returns (resolved, error_or_None)."""
        if not raw_path:
            return Path(), ToolResult(False, "No path provided", tool_name)

        if self.security.path_validation:
            resolved, error = validate_path(
                raw_path, self.working_dir,
                extra_blocked=self.security.blocked_paths,
                extra_allowed=self.security.allowed_paths,
            )
            if error:
                return Path(resolved), ToolResult(False, f"[Security] {error}", tool_name)
            return Path(resolved), None

        # No validation — resolve path the simple way
        if raw_path.startswith('/'):
            return Path(raw_path), None
        return Path(self.working_dir) / raw_path, None

    def _read_file(self, args: dict) -> ToolResult:
        """Read a file."""
        path = args.get('path', '')
        file_path, error = self._validate_file_path(path, 'read')
        if error:
            return error

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

        file_path, error = self._validate_file_path(path, 'write')
        if error:
            return error

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

        file_path, error = self._validate_file_path(path, 'edit')
        if error:
            return error

        if not old_string:
            return ToolResult(
                success=False,
                output="No old_string provided",
                tool_name='edit'
            )

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
