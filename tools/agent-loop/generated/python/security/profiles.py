"""Enhanced security profiles for the agent loop.

Each profile defines a complete security posture: what's blocked, what's
proxied, what's logged, and what isolation is applied.
"""

from dataclasses import dataclass, field

# Binding: security_profile/2 -> SecurityConfig.from_profile(name) [function_call]

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

    # Layer 3.5: PATH-based wrapper scripts
    path_proxying: bool = False

    # Layer 4: Filesystem isolation
    proot_isolation: bool = False
    proot_allowed_dirs: list[str] = field(default_factory=list)

    # Layer 5: Audit logging
    audit_logging: str = 'disabled'  # disabled, basic, detailed, forensic

    # Layer 6: Network isolation (future)
    network_isolation: str = 'disabled'  # disabled, localhost_only, blocked

    # Layer 7: Anomaly detection (future)
    anomaly_detection: bool = False

    # Safe commands — subset that skip confirmation
    safe_commands: list[str] = field(default_factory=list)
    max_file_read_size: int | None = None  # bytes, None = unlimited
    max_file_write_size: int | None = None


# ── Built-in profiles ─────────────────────────────────────────────────────

_GUARDED_EXTRA_BLOCKS = [
    r'^sudo\s',
    r'\bbase64\b.*\|\s*(bash|sh)',
    r'\beval\s',
    r'\bnohup\s',
    r'\bdisown\s',
    r'&\s*$',
    r'\bpython[23]?\s+-c\s.*os\.system',
    r'\bpython[23]?\s+-c\s.*subprocess',
    r'\bpython[23]?\s+-c\s.*__import__',
    r'\bnode\s+-e\s.*child_process',
]

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
    r'^git\s+(status|log|diff|show|branch)',
]

_PARANOID_CONFIRM = [
    r'^find\s+(?!.*(-exec|-execdir|-delete|-ok)\b)[^;|&]*$',
    r'^python3\s+[^-].*\.py$',
    r'^node\s+[^-].*\.js$',
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
            command_proxying='enabled',
            audit_logging='detailed',
            network_isolation='localhost_only',
            safe_commands=list(_PARANOID_SAFE),
        ),
        'paranoid': SecurityProfile(
            name='paranoid',
            description='Maximum security for chaotic/untrusted agents',
            path_validation=True,
            command_blocklist=True,
            allowed_commands=list(_PARANOID_ALLOWED),
            allowed_commands_only=True,
            command_proxying='strict',
            audit_logging='forensic',
            network_isolation='blocked',
            anomaly_detection=True,
            safe_commands=list(_PARANOID_SAFE),
            max_file_read_size=1048576,
            max_file_write_size=10485760,
        ),
    }


def get_profile(name: str) -> SecurityProfile:
    """Get a built-in profile by name, defaulting to cautious."""
    profiles = get_builtin_profiles()
    return profiles.get(name, profiles['cautious'])


# --- shared_logic: security (generated from compile_logic) ---

def is_path_safe(path):
    """Check if a path does not contain directory traversal sequences."""
    return (not path.startswith("..") and not path.startswith("/.."))

def is_visible_file(filename):
    """Check if a filename is not a hidden dotfile."""
    return not filename.startswith(".")

def is_hidden_path(path):
    """Check if a path component starts with a dot (hidden file/directory)."""
    return path.startswith(".")

def has_path_traversal(path):
    """Check if a path contains .. traversal sequences."""
    return path.startswith("..")

def is_safe_command(cmd):
    """Check if a command starts with a known safe prefix (ls, cat, grep, echo)."""
    return ((cmd.startswith("ls") or cmd.startswith("cat")) or (cmd.startswith("grep") or cmd.startswith("echo")))

def is_blocked_command(cmd):
    """Check if a command starts with a known dangerous prefix (rm -rf, dd, mkfs)."""
    return (cmd.startswith("rm -rf") or (cmd.startswith("dd ") or cmd.startswith("mkfs")))

def is_writable_path(path):
    """Check if a path is in a writable location (not starting with /etc, /usr, /bin)."""
    return not (path.startswith("/etc") or (path.startswith("/usr") or path.startswith("/bin")))

def needs_audit(profile):
    """Check if a security profile requires audit logging (guarded or paranoid)."""
    return (profile == "paranoid" or profile == "guarded")

def allows_auto(profile):
    """Check if a security profile allows auto-approval of tools (only open does)."""
    return profile == "open"

def profile_count(profiles):
    """Return the number of available security profiles."""
    return profiles

def is_home_path(path):
    """Check if a path starts with /home or /data/data (Termux home)."""
    return (path.startswith("/home") or path.startswith("/data/data"))

