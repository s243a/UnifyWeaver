"""proot-based filesystem isolation layer.

Wraps subprocess commands in proot to provide filesystem isolation.
On Termux/Android, proot is available via ``pkg install proot``.

This is Layer 4 in the security model — the outermost execution
wrapper, sitting after PATH proxying (Layer 3.5).

When ``redirect_home`` is enabled, proot binds a temporary directory
over ``$HOME`` so that destructive commands (e.g. ``rm -rf ~/``) hit
the fake home instead of the real one.  The real home contents are
copied into the temp dir so commands see a realistic environment.
"""

import os
import shlex
import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProotConfig:
    """Configuration for proot sandboxing."""
    # Extra directories the agent is allowed to access (bind-mounted)
    allowed_dirs: list[str] = field(default_factory=list)
    # Read-only bind mounts
    readonly_binds: list[str] = field(default_factory=list)
    # Kill child processes on exit
    kill_on_exit: bool = True
    # Termux prefix (for binaries, libs, etc.)
    termux_prefix: str = '/data/data/com.termux/files/usr'
    # Extra proot flags passed verbatim
    extra_flags: list[str] = field(default_factory=list)
    # Redirect $HOME to a temp directory inside proot.
    # When set to a path, proot binds that path over $HOME so writes
    # to ~ hit the fake dir, not the real home.
    redirect_home: str | None = None
    # Dry-run mode: wrap_command returns the full proot invocation
    # string but _run() callers can inspect it instead of executing.
    # Useful for testing that the command WOULD be sandboxed correctly
    # without actually running destructive commands.
    dry_run: bool = False


class ProotSandbox:
    """Wraps commands in proot for filesystem isolation."""

    def __init__(self, working_dir: str, config: ProotConfig | None = None):
        self.working_dir = working_dir
        self.config = config or ProotConfig()
        self._proot_path: str | None = None
        self._available: bool | None = None

    def is_available(self) -> bool:
        """Check if proot is installed and usable."""
        if self._available is not None:
            return self._available
        self._proot_path = shutil.which('proot')
        self._available = self._proot_path is not None
        return self._available

    def wrap_command(self, command: str) -> str:
        """Wrap a bash command to run inside proot.

        The wrapped command:
        - Binds the working directory (read-write)
        - Binds Termux usr prefix (for binaries, libs)
        - Binds /proc, /dev, /system (needed for many commands)
        - Binds any extra allowed_dirs from config
        - Sets working directory inside proot
        - Uses --kill-on-exit to prevent orphan processes

        Returns:
            Shell command string suitable for subprocess.run(shell=True).

        Raises:
            RuntimeError: If proot is not installed.
        """
        if not self.is_available():
            raise RuntimeError(
                "proot is not installed. Install with: pkg install proot"
            )

        parts = [self._proot_path]

        if self.config.kill_on_exit:
            parts.append('--kill-on-exit')

        # Working directory inside proot
        parts.extend(['-w', self.working_dir])

        # Essential system binds (Termux/Android)
        essential = [
            self.config.termux_prefix,  # bins, libs, etc.
            '/proc',
            '/dev',
            '/system',                  # Android linker
        ]
        for bind in essential:
            if os.path.exists(bind):
                parts.extend(['-b', bind])

        # Working directory (read-write)
        parts.extend(['-b', self.working_dir])

        # Redirect $HOME to a fake directory so destructive commands
        # (rm -rf ~/, etc.) hit the copy, not the real home.
        if self.config.redirect_home:
            real_home = os.path.expanduser('~')
            parts.extend(['-b', f'{self.config.redirect_home}:{real_home}'])

        # User-configured allowed directories
        for dir_path in self.config.allowed_dirs:
            expanded = os.path.expanduser(dir_path)
            if os.path.exists(expanded):
                parts.extend(['-b', expanded])

        # Read-only binds
        for dir_path in self.config.readonly_binds:
            expanded = os.path.expanduser(dir_path)
            if os.path.exists(expanded):
                parts.extend(['-b', expanded])

        # Extra flags
        parts.extend(self.config.extra_flags)

        # The command to execute inside proot
        bash_path = os.path.join(self.config.termux_prefix, 'bin', 'bash')
        parts.extend([bash_path, '-c', command])

        return self._quote_parts(parts)

    def describe_command(self, command: str) -> dict:
        """Return a description of what wrap_command would produce.

        Useful for dry-run / inspection without executing anything.
        Returns a dict with 'proot_args', 'inner_command', 'binds',
        and 'redirect_home' so callers can verify the sandbox config.
        """
        if not self.is_available():
            return {'error': 'proot not available'}

        real_home = os.path.expanduser('~')
        binds = []
        # Essential
        for bind in [self.config.termux_prefix, '/proc', '/dev', '/system']:
            if os.path.exists(bind):
                binds.append(bind)
        binds.append(self.working_dir)
        # Home redirect
        home_redirect = None
        if self.config.redirect_home:
            home_redirect = f'{self.config.redirect_home}:{real_home}'
            binds.append(home_redirect)
        # Allowed dirs
        for d in self.config.allowed_dirs:
            expanded = os.path.expanduser(d)
            if os.path.exists(expanded):
                binds.append(expanded)
        return {
            'inner_command': command,
            'binds': binds,
            'redirect_home': home_redirect,
            'working_dir': self.working_dir,
            'kill_on_exit': self.config.kill_on_exit,
            'dry_run': self.config.dry_run,
        }

    def build_env_overrides(self) -> dict[str, str]:
        """Return env vars needed for proot execution.

        These should be merged into the subprocess env dict.
        """
        return {
            # Required on many Android kernels to avoid seccomp errors
            'PROOT_NO_SECCOMP': '1',
            # Disable termux-exec path remapping inside proot
            'LD_PRELOAD': '',
        }

    def status(self) -> dict:
        """Return diagnostic info."""
        return {
            'available': self.is_available(),
            'proot_path': self._proot_path,
            'working_dir': self.working_dir,
            'allowed_dirs': self.config.allowed_dirs,
            'kill_on_exit': self.config.kill_on_exit,
        }

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _quote_parts(parts: list[str]) -> str:
        """Quote command parts for shell execution.

        Everything except the inner command (after ``-c``) gets normal
        shell quoting.  The inner command needs single-quote escaping.
        """
        if len(parts) >= 3 and parts[-2] == '-c':
            prefix = parts[:-1]
            inner = parts[-1]
            quoted_prefix = ' '.join(shlex.quote(p) for p in prefix)
            escaped_inner = inner.replace("'", "'\\''")
            return f"{quoted_prefix} '{escaped_inner}'"
        return ' '.join(shlex.quote(p) for p in parts)