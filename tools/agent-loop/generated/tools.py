"""Tool handler for executing agent tool calls."""

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from backends.base import ToolCall
from security.audit import AuditLogger
from security.profiles import SecurityProfile, get_profile
from security.proxy import CommandProxyManager


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

    # Enhanced profile settings (Phase 2)
    allowed_commands_only: bool = False   # paranoid: allowlist mode
    safe_commands: list[str] = field(default_factory=list)  # skip confirmation
    command_proxying: str = 'disabled'    # disabled, optional, enabled, strict
    max_file_read_size: int | None = None
    max_file_write_size: int | None = None

    # Optional execution layers (opt-in via CLI or config)
    path_proxying: bool = False                                 # Layer 3.5
    proot_sandbox: bool = False                                 # Layer 4
    proot_allowed_dirs: list[str] = field(default_factory=list)

    @classmethod
    def from_profile(cls, profile: str) -> 'SecurityConfig':
        """Create config from a named security profile."""
        sp = get_profile(profile)
        return cls(
            path_validation=sp.path_validation,
            command_blocklist=sp.command_blocklist,
            blocked_paths=list(sp.blocked_paths),
            blocked_commands=list(sp.blocked_commands),
            allowed_paths=list(sp.allowed_paths),
            allowed_commands=list(sp.allowed_commands),
            allowed_commands_only=sp.allowed_commands_only,
            safe_commands=list(sp.safe_commands),
            command_proxying=sp.command_proxying,
            max_file_read_size=sp.max_file_read_size,
            max_file_write_size=sp.max_file_write_size,
            path_proxying=sp.path_proxying,
            proot_sandbox=sp.proot_isolation,
            proot_allowed_dirs=list(sp.proot_allowed_dirs),
        )


class ToolHandler:
    """Handles execution of tool calls from agent responses."""

    def __init__(
        self,
        allowed_tools: list[str] | None = None,
        confirm_destructive: bool = True,
        working_dir: str | None = None,
        security: SecurityConfig | None = None,
        audit: AuditLogger | None = None
    ):
        self.allowed_tools = allowed_tools or ['bash', 'read', 'write', 'edit']
        self.confirm_destructive = confirm_destructive
        self.working_dir = working_dir or os.getcwd()
        self.security = security or SecurityConfig()
        self.audit = audit or AuditLogger(level='disabled')

        # Command proxy (active when command_proxying != 'disabled')
        self.proxy: CommandProxyManager | None = None
        if self.security.command_proxying != 'disabled':
            self.proxy = CommandProxyManager()

        # PATH-based proxy (optional — prepends wrapper scripts to PATH)
        self.path_proxy = None
        if self.security.path_proxying:
            from security.path_proxy import PathProxyManager
            self.path_proxy = PathProxyManager()
            if self.proxy:
                generated = self.path_proxy.generate_wrappers(self.proxy)
                if generated:
                    self.audit.log_proxy_action(
                        '', 'path_proxy', 'init',
                        f'Generated wrappers: {", ".join(generated)}')

        # proot sandbox (optional — wraps commands in proot)
        self.proot = None
        if self.security.proot_sandbox:
            from security.proot_sandbox import ProotSandbox, ProotConfig
            proot_cfg = ProotConfig(
                allowed_dirs=self.security.proot_allowed_dirs)
            sandbox = ProotSandbox(self.working_dir, proot_cfg)
            if sandbox.is_available():
                self.proot = sandbox
            else:
                import sys
                print("[Warning] proot sandbox requested but proot not "
                      "found. Install with: pkg install proot",
                      file=sys.stderr)

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

        # Run security pre-checks BEFORE asking for confirmation.
        # No point prompting the user if the command will be blocked.
        if tool_call.name == 'bash':
            blocked = self._pre_check_bash(tool_call.arguments)
            if blocked:
                self.audit.log_tool_call(
                    tool_call.name, False,
                    args_summary=str(tool_call.arguments)[:256]
                )
                return blocked

        # Check for confirmation if destructive — but skip for safe commands
        if self.confirm_destructive and tool_call.name in self.destructive_tools:
            if not self._is_safe_command(tool_call):
                if not self._confirm_execution(tool_call):
                    return ToolResult(
                        success=False,
                        output="User declined to execute tool",
                        tool_name=tool_call.name
                    )

        try:
            result = self.tools[tool_call.name](tool_call.arguments)
            self.audit.log_tool_call(
                tool_call.name, result.success,
                args_summary=str(tool_call.arguments)[:256]
            )
            return result
        except Exception as e:
            self.audit.log_tool_call(tool_call.name, False, args_summary=str(e))
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

    def _is_safe_command(self, tool_call: ToolCall) -> bool:
        """Check if a tool call is safe enough to skip confirmation.

        Only applies to bash commands that match the safe_commands patterns
        (read-only commands like ls, cat, grep, git status, etc.).
        """
        if tool_call.name != 'bash':
            return False
        if not self.security.safe_commands:
            return False

        command = tool_call.arguments.get('command', '').strip()
        return any(
            re.search(pat, command, re.IGNORECASE)
            for pat in self.security.safe_commands
        )

    def _pre_check_bash(self, args: dict) -> ToolResult | None:
        """Run security checks on a bash command BEFORE confirmation.

        Returns a ToolResult if the command is blocked, or None if it
        passes all security checks.  Called from execute() so the user
        is never prompted to approve a command that will be rejected.
        """
        command = args.get('command', '').strip()
        if not command:
            return ToolResult(
                success=False,
                output="No command provided",
                tool_name='bash'
            )

        # Allowlist-only mode (paranoid): reject unless explicitly allowed
        if self.security.allowed_commands_only:
            matched = any(
                re.search(pat, command, re.IGNORECASE)
                for pat in self.security.allowed_commands
            )
            if not matched:
                reason = "Command not in allowlist"
                self.audit.log_command(command, allowed=False, reason=reason)
                return ToolResult(
                    success=False,
                    output=f"[Security] {reason}",
                    tool_name='bash'
                )

        # Command blocklist check
        if self.security.command_blocklist:
            reason = is_command_blocked(
                command,
                extra_blocked=self.security.blocked_commands,
                extra_allowed=self.security.allowed_commands if not self.security.allowed_commands_only else None,
            )
            if reason:
                self.audit.log_command(command, allowed=False, reason=reason)
                return ToolResult(
                    success=False,
                    output=f"[Security] {reason}",
                    tool_name='bash'
                )

        # Command proxy check (Layer 3)
        if self.proxy:
            proxy_mode = self.security.command_proxying
            allowed, reason = self.proxy.check(command, proxy_mode)
            if not allowed:
                self.audit.log_command(command, allowed=False, reason=reason)
                self.audit.log_proxy_action(command, '', 'block', reason or '')
                return ToolResult(
                    success=False,
                    output=f"[Security/Proxy] {reason}",
                    tool_name='bash'
                )

        return None  # All checks passed

    def _execute_bash(self, args: dict) -> ToolResult:
        """Execute a bash command."""
        command = args.get('command', '')
        if not command:
            return ToolResult(
                success=False,
                output="No command provided",
                tool_name='bash'
            )

        # Security checks already ran in _pre_check_bash() before
        # the confirmation prompt, so skip them here.

        # Build execution environment and command — when neither layer
        # is enabled, exec_env stays None and exec_cmd stays unchanged,
        # so subprocess.run() behaves identically to before.
        exec_env = None
        exec_cmd = command

        # Layer 3.5: PATH proxy — prepend wrapper dir to PATH
        if self.path_proxy:
            log_file = self.audit._log_file if self.audit._log_file else None
            exec_env = self.path_proxy.build_env(
                proxy_mode=self.security.command_proxying,
                audit_log=str(log_file) if log_file else None,
            )

        # Layer 4: proot — wrap command in proot invocation
        if self.proot:
            exec_cmd = self.proot.wrap_command(command)
            proot_env = self.proot.build_env_overrides()
            if exec_env is None:
                exec_env = dict(os.environ)
            exec_env.update(proot_env)

        try:
            result = subprocess.run(
                exec_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=self.working_dir,
                env=exec_env,
            )

            output = result.stdout
            if result.stderr:
                output += "\n" + result.stderr

            self.audit.log_command(
                command, allowed=True,
                output=output.strip() if output else None
            )

            return ToolResult(
                success=result.returncode == 0,
                output=output.strip() or "(no output)",
                tool_name='bash'
            )

        except subprocess.TimeoutExpired:
            self.audit.log_command(command, allowed=True, reason='timeout')
            return ToolResult(
                success=False,
                output="Command timed out after 120 seconds",
                tool_name='bash'
            )
        except Exception as e:
            self.audit.log_command(command, allowed=True, reason=str(e))
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
                self.audit.log_file_access(raw_path, tool_name, allowed=False, reason=error)
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
            # Check file size limit before reading
            if self.security.max_file_read_size is not None:
                try:
                    file_size = file_path.stat().st_size
                    if file_size > self.security.max_file_read_size:
                        reason = (f"File too large: {file_size} bytes "
                                  f"(limit: {self.security.max_file_read_size})")
                        self.audit.log_file_access(path, 'read', allowed=False, reason=reason)
                        return ToolResult(False, f"[Security] {reason}", 'read')
                except OSError:
                    pass  # Let the read_text() below handle missing files

            content = file_path.read_text()
            self.audit.log_file_access(
                path, 'read', allowed=True, bytes_count=len(content)
            )
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

        # Check write size limit
        if self.security.max_file_write_size is not None:
            if len(content) > self.security.max_file_write_size:
                reason = (f"Content too large: {len(content)} bytes "
                          f"(limit: {self.security.max_file_write_size})")
                self.audit.log_file_access(path, 'write', allowed=False, reason=reason)
                return ToolResult(False, f"[Security] {reason}", 'write')

        try:
            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            self.audit.log_file_access(
                path, 'write', allowed=True, bytes_count=len(content)
            )
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
            self.audit.log_file_access(
                path, 'edit', allowed=True,
                bytes_count=len(new_content)
            )

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
