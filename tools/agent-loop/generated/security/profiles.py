"""Enhanced security profiles for the agent loop.

Each profile defines a complete security posture: what's blocked, what's
proxied, what's logged, and what isolation is applied.
"""

from dataclasses import dataclass, field


@dataclass
class SecurityProfile:
    """Full security profile with all layer settings."""
    name: str
    description: str = ''

    # Layer 1: Path validation
    path_validation: bool = True
    blocked_paths: list[str] = field(default_factory=list)
    allowed_paths: list[str] = field(default_factory=list)

    # Layer 2: Command blocklist / allowlist
    command_blocklist: bool = True
    blocked_commands: list[str] = field(default_factory=list)
    allowed_commands: list[str] = field(default_factory=list)
    allowed_commands_only: bool = False  # If True, only allowed_commands may run

    # Layer 3: Command proxying
    command_proxying: str = 'disabled'  # disabled, optional, enabled, strict

    # Layer 4: Filesystem isolation
    proot_isolation: bool = False

    # Layer 5: Audit logging
    audit_logging: str = 'disabled'  # disabled, basic, detailed, forensic

    # Layer 6: Network isolation (future)
    network_isolation: str = 'disabled'  # disabled, localhost_only, blocked

    # Layer 7: Anomaly detection (future)
    anomaly_detection: bool = False

    # Safe commands — subset of allowed_commands that skip confirmation
    # (read-only / harmless commands the user doesn't need to approve)
    safe_commands: list[str] = field(default_factory=list)

    # Resource limits
    max_file_read_size: int | None = None   # bytes, None = unlimited
    max_file_write_size: int | None = None


# ── Built-in profiles ─────────────────────────────────────────────────────

_GUARDED_EXTRA_BLOCKS = [
    r'^sudo\s',
    r'\bbase64\b.*\|\s*(bash|sh)',
    r'\beval\s',
    r'\bnohup\s',
    r'\bdisown\s',
    r'&\s*$',                              # backgrounding
    r'\bpython[23]?\s+-c\s.*os\.system',
    r'\bpython[23]?\s+-c\s.*subprocess',
    r'\bpython[23]?\s+-c\s.*__import__',
    r'\bnode\s+-e\s.*child_process',
]

# Commands that are inherently safe (read-only / harmless) — skip confirmation
_PARANOID_SAFE = [
    r'^ls(\s|$)',
    r'^cat\s',
    r'^head\s',
    r'^tail\s',
    r'^grep\s',
    r'^echo\s',
    r'^pwd$',
    r'^cd\s',
    r'^wc\s',
    r'^sort\s',
    r'^diff\s',
    r'^git\s+(status|log|diff|show|branch)',  # read-only git
]

# Commands that are allowed but potentially dangerous — still ask for confirmation
_PARANOID_CONFIRM = [
    # find: block -exec, -execdir, -delete, -ok; no chaining
    r'^find\s+(?!.*(-exec|-execdir|-delete|-ok)\b)[^;|&]*$',
    r'^python3\s+[^-].*\.py$',             # run .py files, no -c
    r'^node\s+[^-].*\.js$',                # run .js files, no -e
]

# Combined allowlist (safe + confirm)
_PARANOID_ALLOWED = _PARANOID_SAFE + _PARANOID_CONFIRM


def get_builtin_profiles() -> dict[str, SecurityProfile]:
    """Return all built-in security profiles."""
    return {
        'open': SecurityProfile(
            name='open',
            description='No restrictions - for trusted manual use',
            path_validation=False,
            command_blocklist=False,
            command_proxying='disabled',
            audit_logging='disabled',
        ),
        'cautious': SecurityProfile(
            name='cautious',
            description='Basic safety for well-behaved agents like Claude Code',
            path_validation=True,
            command_blocklist=True,
            command_proxying='disabled',
            audit_logging='basic',
        ),
        'guarded': SecurityProfile(
            name='guarded',
            description='Actively protected and monitored for semi-autonomous agents',
            path_validation=True,
            command_blocklist=True,
            blocked_commands=list(_GUARDED_EXTRA_BLOCKS),
            safe_commands=list(_PARANOID_SAFE),
            command_proxying='enabled',
            audit_logging='detailed',
            network_isolation='localhost_only',
        ),
        'paranoid': SecurityProfile(
            name='paranoid',
            description='Maximum security for chaotic/untrusted agents',
            path_validation=True,
            command_blocklist=True,
            allowed_commands_only=True,
            allowed_commands=list(_PARANOID_ALLOWED),
            safe_commands=list(_PARANOID_SAFE),
            command_proxying='strict',
            audit_logging='forensic',
            network_isolation='blocked',
            anomaly_detection=True,
            max_file_read_size=1_048_576,     # 1 MB
            max_file_write_size=10_485_760,   # 10 MB
        ),
    }


def get_profile(name: str) -> SecurityProfile:
    """Get a built-in profile by name, defaulting to cautious."""
    profiles = get_builtin_profiles()
    return profiles.get(name, profiles['cautious'])
