"""Audit logging for agent loop security events.

Writes JSONL (one JSON object per line) to session-specific log files.
Three verbosity levels:
  - basic:    commands, tool calls, security violations
  - detailed: + file access, API calls, cost tracking
  - forensic: + command output, environment, timing
"""

import json
import os
import time
from pathlib import Path
from typing import Any


class AuditLogger:
    """JSONL audit logger for agent loop sessions."""

    def __init__(self, log_dir: str | None = None, level: str = 'basic'):
        if log_dir:
            self.log_dir = Path(log_dir)
        else:
            self.log_dir = Path(os.path.expanduser('~/.agent-loop/audit'))
        self.level = level  # basic, detailed, forensic
        self._session_id: str | None = None
        self._log_file: Path | None = None
        self._enabled = level != 'disabled'

    def start_session(self, session_id: str, user_id: str = '',
                      security_profile: str = '') -> None:
        """Begin logging for a new session."""
        if not self._enabled:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._session_id = session_id
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        self._log_file = self.log_dir / f'{timestamp}-{session_id}.jsonl'
        self._write({
            'event': 'session_start',
            'session_id': session_id,
            'user_id': user_id,
            'security_profile': security_profile,
        })

    def end_session(self) -> None:
        """End the current session."""
        if not self._enabled or not self._session_id:
            return
        self._write({'event': 'session_end', 'session_id': self._session_id})
        self._session_id = None
        self._log_file = None

    # ── Event logging methods ─────────────────────────────────────────────

    def log_command(self, command: str, allowed: bool,
                    reason: str | None = None,
                    output: str | None = None) -> None:
        """Log a bash command execution attempt."""
        if not self._enabled:
            return
        entry: dict[str, Any] = {
            'event': 'command',
            'command': command,
            'allowed': allowed,
        }
        if reason:
            entry['reason'] = reason
        if output is not None and self.level == 'forensic':
            # Truncate large outputs
            entry['output'] = output[:4096]
        self._write(entry)

    def log_file_access(self, path: str, operation: str, allowed: bool,
                        reason: str | None = None,
                        bytes_count: int = 0) -> None:
        """Log a file read/write/edit attempt."""
        if not self._enabled:
            return
        if self.level == 'basic':
            # basic level only logs blocked file access
            if allowed:
                return
        entry: dict[str, Any] = {
            'event': 'file_access',
            'path': path,
            'operation': operation,
            'allowed': allowed,
        }
        if reason:
            entry['reason'] = reason
        if bytes_count and self.level in ('detailed', 'forensic'):
            entry['bytes'] = bytes_count
        self._write(entry)

    def log_tool_call(self, tool_name: str, success: bool,
                      args_summary: str = '') -> None:
        """Log a tool call execution."""
        if not self._enabled:
            return
        entry: dict[str, Any] = {
            'event': 'tool_call',
            'tool': tool_name,
            'success': success,
        }
        if args_summary:
            entry['args'] = args_summary[:512]
        self._write(entry)

    def log_security_violation(self, rule: str, severity: str,
                               details: dict[str, Any] | None = None) -> None:
        """Log a security rule violation."""
        if not self._enabled:
            return
        entry: dict[str, Any] = {
            'event': 'security_violation',
            'rule': rule,
            'severity': severity,
        }
        if details:
            entry['details'] = details
        self._write(entry)

    def log_api_call(self, backend: str, model: str,
                     tokens: int = 0, cost: float = 0.0) -> None:
        """Log an API call with cost info."""
        if not self._enabled:
            return
        if self.level == 'basic':
            return
        self._write({
            'event': 'api_call',
            'backend': backend,
            'model': model,
            'tokens': tokens,
            'cost': cost,
        })

    def log_proxy_action(self, command: str, proxy_name: str,
                         action: str, reason: str = '') -> None:
        """Log a command proxy intercept."""
        if not self._enabled:
            return
        entry: dict[str, Any] = {
            'event': 'proxy_action',
            'command': command[:512],
            'proxy': proxy_name,
            'action': action,
        }
        if reason:
            entry['reason'] = reason
        self._write(entry)

    # ── Internal ──────────────────────────────────────────────────────────

    def _write(self, entry: dict[str, Any]) -> None:
        """Append a timestamped JSON entry to the log file."""
        if not self._log_file:
            return
        entry['timestamp'] = time.time()
        try:
            with open(self._log_file, 'a') as f:
                f.write(json.dumps(entry, default=str) + '\n')
        except OSError:
            pass  # Don't crash the agent loop over logging failures

    # ── Query helpers ─────────────────────────────────────────────────────

    def search_logs(self, query: str, days: int = 7) -> list[dict[str, Any]]:
        """Search audit logs by text match."""
        results: list[dict[str, Any]] = []
        cutoff = time.time() - (days * 86400)
        if not self.log_dir.exists():
            return results
        for log_file in sorted(self.log_dir.glob('*.jsonl')):
            try:
                with open(log_file) as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if entry.get('timestamp', 0) < cutoff:
                            continue
                        if query.lower() in line.lower():
                            results.append(entry)
                        if len(results) >= 100:
                            return results
            except OSError:
                continue
        return results
