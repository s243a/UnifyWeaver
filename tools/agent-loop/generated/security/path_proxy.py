"""PATH-based command proxy layer.

Generates wrapper scripts in ~/.agent-loop/bin/ that shadow dangerous
commands.  When enabled, these scripts are prepended to PATH so that
even if the in-process proxy misses something (e.g. inside a piped
command or shell script), the wrapper catches it at exec time.

This is Layer 3.5 in the security model — between the in-process
proxy (Layer 3) and proot isolation (Layer 4).
"""

import os
import stat
from pathlib import Path
from dataclasses import dataclass, field

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .proxy import CommandProxyManager, CommandProxy


class PathProxyManager:
    """Manages wrapper scripts in ~/.agent-loop/bin/."""

    DEFAULT_BIN_DIR = os.path.expanduser('~/.agent-loop/bin')

    def __init__(self, bin_dir: str | None = None):
        self.bin_dir = Path(bin_dir or self.DEFAULT_BIN_DIR)
        self._generated: set[str] = set()

    def generate_wrappers(self, proxy_mgr: 'CommandProxyManager') -> list[str]:
        """Auto-generate wrapper scripts from CommandProxyManager rules.

        Returns list of command names for which wrappers were created.
        """
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        generated = []

        for cmd_name, proxy in proxy_mgr.proxies.items():
            script = self._build_wrapper(cmd_name, proxy)
            path = self.bin_dir / cmd_name
            path.write_text(script)
            path.chmod(stat.S_IRWXU)  # 0o700
            generated.append(cmd_name)
            self._generated.add(cmd_name)

        return generated

    def build_env(self, base_env: dict[str, str] | None = None,
                  proxy_mode: str = 'enabled',
                  audit_log: str | None = None) -> dict[str, str]:
        """Build environment dict with our bin dir prepended to PATH.

        Args:
            base_env: Starting environment (default: os.environ copy).
            proxy_mode: 'enabled' or 'strict' — passed to wrappers.
            audit_log: Optional path to JSONL audit log for wrapper logging.

        Returns:
            New env dict suitable for subprocess.run(env=...).
        """
        env = dict(base_env or os.environ)
        current_path = env.get('PATH', '')
        bin_str = str(self.bin_dir)
        # Don't double-prepend
        if not current_path.startswith(bin_str + ':'):
            env['PATH'] = f'{bin_str}:{current_path}'
        env['AGENT_LOOP_PROXY_MODE'] = proxy_mode
        if audit_log:
            env['AGENT_LOOP_AUDIT_LOG'] = audit_log
        return env

    def cleanup(self) -> None:
        """Remove all generated wrapper scripts."""
        for cmd_name in list(self._generated):
            script = self.bin_dir / cmd_name
            if script.exists():
                script.unlink()
            self._generated.discard(cmd_name)

    def status(self) -> dict:
        """Return diagnostic info."""
        wrappers = []
        if self.bin_dir.exists():
            wrappers = sorted(
                f.name for f in self.bin_dir.iterdir()
                if f.is_file() and os.access(f, os.X_OK)
            )
        return {
            'bin_dir': str(self.bin_dir),
            'exists': self.bin_dir.exists(),
            'wrappers': wrappers,
            'generated': sorted(self._generated),
        }

    # ── Internal ──────────────────────────────────────────────────────────

    def _build_wrapper(self, cmd_name: str, proxy: 'CommandProxy') -> str:
        """Build a bash wrapper script for a single command.

        The wrapper:
        1. Finds the REAL binary by searching PATH excluding our bin dir
        2. Checks the full command against block-action rules
        3. If blocked: prints reason to stderr, exits 126
        4. If allowed: exec's the real binary with all original args
        """
        # Build rule checks (only 'block' rules — 'warn' just logs)
        checks = []
        for rule in proxy.rules:
            if rule.action != 'block':
                continue
            # Escape single quotes for embedding in bash
            pattern = rule.pattern.replace("'", "'\\''")
            message = (rule.message or f'Blocked: {rule.pattern}').replace("'", "'\\''")
            checks.append(
                f'if echo "$FULL_CMD" | grep -qiP -- \'{pattern}\'; then\n'
                f'  echo "[PATH-Proxy] {message}" >&2\n'
                f'  exit 126\n'
                f'fi'
            )

        checks_block = '\n'.join(checks) if checks else ': # no block rules'

        # Strict-mode full block
        strict_block = ''
        if proxy.blocked_in_strict:
            strict_block = (
                'if [ "$AGENT_LOOP_PROXY_MODE" = "strict" ]; then\n'
                f'  echo "[PATH-Proxy] Command \'{cmd_name}\' is blocked in strict mode" >&2\n'
                '  exit 126\n'
                'fi'
            )

        # Build script — shebang MUST be at column 0 (first byte of file)
        lines = [
            '#!/data/data/com.termux/files/usr/bin/bash',
            f'# Auto-generated PATH proxy wrapper for: {cmd_name}',
            '# DO NOT EDIT — regenerated from CommandProxyManager rules',
            'set -euo pipefail',
            '',
            'SELF_DIR="$(cd "$(dirname "$0")" && pwd)"',
            f'FULL_CMD="{cmd_name} $*"',
            '',
            '# Find the real binary by removing our dir from PATH',
            'CLEAN_PATH="$(echo "$PATH" | tr \':\' \'\\n\' | grep -v "^$SELF_DIR$" | tr \'\\n\' \':\')"',
            f'REAL_BIN="$(PATH="$CLEAN_PATH" command -v {cmd_name} 2>/dev/null || true)"',
            '',
            f'if [ -z "$REAL_BIN" ]; then',
            f'  echo "[PATH-Proxy] Real \'{cmd_name}\' not found in PATH" >&2',
            '  exit 127',
            'fi',
            '',
            '# Strict-mode full block',
            strict_block,
            '',
            '# Rule checks',
            checks_block,
            '',
            '# Audit logging (if enabled via env)',
            'if [ -n "${AGENT_LOOP_AUDIT_LOG:-}" ]; then',
            f'  printf \'{{"event":"path_proxy","command":"{cmd_name}","args":"%s","action":"allow"}}\\n\' "$*" >> "$AGENT_LOOP_AUDIT_LOG" 2>/dev/null || true',
            'fi',
            '',
            'exec "$REAL_BIN" "$@"',
        ]
        return '\n'.join(lines) + '\n'
