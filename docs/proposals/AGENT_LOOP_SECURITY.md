# Agent Loop Security Hardening

## Problem

The agent loop's `tools.py` executes bash commands with `shell=True` and performs file operations with no path validation beyond `startswith('/')`. The confirmation prompt is the only defense, and it's bypassed by `--auto-tools` and `--approval-mode yolo`.

Meanwhile, UnifyWeaver already has a rich declarative security layer (shell_sandbox.pl, rpyc_security.pl, firewall rules, shell command constraints) — but none of it is connected to the agent loop's Python execution.

This proposal covers practical hardening that works in Termux without containers, VMs, or root.

## Current State

| Layer | Status | Notes |
|-------|--------|-------|
| Tool whitelisting | Working | `allowed_tools` list in ToolHandler |
| Confirmation prompts | Working | `confirm_destructive` flag; safe commands skip prompt in paranoid |
| Approval modes | Working | default/auto_edit/yolo/plan |
| Path validation | **Implemented** | `realpath()` resolves symlinks/`..`; blocks `/etc/shadow`, `~/.ssh/`, etc. |
| Command sanitization | **Implemented** | Blocklist + allowlist-only mode (paranoid) |
| Sensitive path blocking | **Implemented** | Blocks system files, credential stores, cloud configs |
| Command proxying | **Implemented** | In-process proxy for rm, curl, wget, python, git, ssh, scp, nc |
| Audit logging | **Implemented** | JSONL audit trail with basic/detailed/forensic levels |
| Security profiles | **Implemented** | open/cautious/guarded/paranoid with distinct behaviors |
| Sandbox integration | **Missing** | proot isolation designed but not yet implemented |
| Declarative rules | **Not connected** | Prolog specs exist but aren't enforced |

## Proposal: Layered Security

### Layer 1: Path Validation (tools.py)

Normalize all file paths and block dangerous targets. No external dependencies.

```python
import os

# Sensitive paths that should never be read or written by agent tools
BLOCKED_PATHS = {
    '/etc/shadow', '/etc/passwd', '/etc/sudoers',
    '/root/.ssh', '/root/.gnupg',
}
BLOCKED_PREFIXES = (
    '/proc/', '/sys/', '/dev/',
)
BLOCKED_HOME_PATTERNS = (
    '.ssh/', '.gnupg/', '.aws/', '.config/gcloud/',
    '.env', '.netrc', '.npmrc',
)

def validate_path(raw_path: str, working_dir: str) -> tuple[str, str | None]:
    """Resolve and validate a file path.

    Returns (resolved_path, error_message).
    error_message is None if the path is safe.
    """
    # Resolve relative paths against working_dir
    if raw_path.startswith('/'):
        resolved = os.path.realpath(raw_path)
    else:
        resolved = os.path.realpath(os.path.join(working_dir, raw_path))

    # Block system paths
    if resolved in BLOCKED_PATHS:
        return resolved, f"Blocked: {resolved} is a sensitive system file"
    for prefix in BLOCKED_PREFIXES:
        if resolved.startswith(prefix):
            return resolved, f"Blocked: {prefix} is a system directory"

    # Block sensitive home directory patterns
    home = os.path.expanduser('~')
    if resolved.startswith(home):
        rel = resolved[len(home):].lstrip('/')
        for pattern in BLOCKED_HOME_PATTERNS:
            if rel.startswith(pattern) or rel == pattern.rstrip('/'):
                return resolved, f"Blocked: ~/{pattern} contains credentials"

    return resolved, None
```

Key points:
- `os.path.realpath()` resolves `..` and symlinks, defeating path traversal
- Blocklists cover system files, credential stores, and cloud provider configs
- Returns the resolved path so the tool can log what it actually accessed

### Layer 2: Command Proxy via PATH (bash tool)

Instead of raw `shell=True`, wrap commands through a proxy script that enforces constraints. This works on Termux without containers.

**Approach A: PATH-based proxy**

Create wrapper scripts in a `~/.agent-loop/bin/` directory that shadow dangerous commands:

```bash
# ~/.agent-loop/bin/rm — proxy that blocks rm -rf /
#!/data/data/com.termux/files/usr/bin/bash
for arg in "$@"; do
    case "$arg" in
        /) echo "BLOCKED: rm on /" >&2; exit 1 ;;
        /*) ;; # absolute paths are allowed
    esac
done
exec /data/data/com.termux/files/usr/bin/rm "$@"
```

The agent loop prepends this directory to `PATH` when calling subprocess:

```python
def _execute_bash(self, args: dict) -> ToolResult:
    proxy_dir = os.path.expanduser('~/.agent-loop/bin')
    env = os.environ.copy()
    if os.path.isdir(proxy_dir):
        env['PATH'] = proxy_dir + ':' + env.get('PATH', '')

    result = subprocess.run(
        command, shell=True, env=env,
        capture_output=True, text=True,
        timeout=120, cwd=self.working_dir
    )
```

**Approach B: Command blocklist in Python**

Before executing, check the command against patterns:

```python
BLOCKED_COMMANDS = [
    r'\brm\s+-[rf]*\s+/',          # rm -rf /
    r'\bmkfs\b',                    # format filesystem
    r'\bdd\s+.*of=/dev/',          # write to device
    r'>\s*/dev/sd',                # redirect to device
    r'\bcurl\b.*\|\s*bash',        # curl | bash
    r'\bwget\b.*\|\s*bash',        # wget | bash
    r'\bchmod\s+777\b',            # world-writable
    r'\bchown\s+root\b',           # chown to root
]

def is_command_blocked(command: str) -> str | None:
    """Return reason if command is blocked, None if allowed."""
    for pattern in BLOCKED_COMMANDS:
        if re.search(pattern, command, re.IGNORECASE):
            return f"Blocked: matches dangerous pattern '{pattern}'"
    return None
```

**Recommendation:** Use both — blocklist catches obvious dangers before execution, PATH proxies catch things the blocklist misses.

### Layer 3: PRoot Sandbox (optional, Termux-compatible)

PRoot is available on Termux (`pkg install proot`) and provides filesystem isolation via ptrace — no root required. The agent loop can optionally run bash commands inside proot:

```python
def _execute_bash_sandboxed(self, args: dict) -> ToolResult:
    command = args.get('command', '')

    if self.sandbox_mode == 'proot':
        # Create isolated rootfs with bind mounts for allowed paths
        proot_cmd = [
            'proot',
            '-0',                              # fake root
            '-w', self.working_dir,            # working directory
            '-b', f'{self.working_dir}',       # bind working dir
            '-b', '/data/data/com.termux/files/usr',  # bind Termux usr
            '/bin/bash', '-c', command,
        ]
        result = subprocess.run(
            proot_cmd, capture_output=True, text=True,
            timeout=120
        )
    else:
        result = subprocess.run(
            command, shell=True, ...
        )
```

This connects the existing `shell_sandbox.pl` proot definitions to actual execution.

**Tradeoffs:**
- Requires `proot` package (`pkg install proot`)
- ~10-30% overhead from ptrace interception
- Not all programs work under proot (some syscalls aren't emulated)
- Good for file isolation, not for network isolation

### Layer 4: Network Restrictions

For API backends making outbound requests, add domain allowlisting:

```python
ALLOWED_DOMAINS = {
    'openrouter.ai',
    'api.anthropic.com',
    'api.openai.com',
    'generativelanguage.googleapis.com',
}

def validate_url(url: str) -> str | None:
    """Return error if URL domain is not in allowlist."""
    from urllib.parse import urlparse
    host = urlparse(url).hostname
    if host and host not in ALLOWED_DOMAINS:
        return f"Blocked: {host} not in allowed domains"
    return None
```

For bash commands, this is harder — use iptables/nftables rules or a transparent proxy. On Termux without root, the practical option is to use proot with a restricted resolv.conf.

### Layer 5: Tool Call Logging and Audit Trail

Log every tool invocation for post-hoc review:

```python
import json
from datetime import datetime

class ToolAuditLog:
    def __init__(self, log_dir='~/.agent-loop/audit'):
        self.log_dir = os.path.expanduser(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(
            self.log_dir,
            f"audit_{datetime.now():%Y%m%d}.jsonl"
        )

    def log(self, tool_call, result, blocked_reason=None):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'tool': tool_call.name,
            'args': tool_call.arguments,
            'success': result.success if result else False,
            'blocked': blocked_reason,
        }
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
```

### Layer 6: Security Profiles and Customization

Tie layers 1-5 together with named profiles in `uwsal.json`. Every layer is independently configurable, and blocklists can be extended or replaced.

```json
{
  "security": {
    "profile": "cautious",
    "blocked_paths": [],
    "blocked_path_prefixes": [],
    "blocked_home_patterns": [],
    "blocked_commands": [],
    "allowed_paths": [],
    "allowed_commands": [],
    "profiles": {
      "open": {
        "path_validation": false,
        "command_blocklist": false,
        "sandbox": "none",
        "confirm_tools": false
      },
      "cautious": {
        "path_validation": true,
        "command_blocklist": true,
        "sandbox": "none",
        "confirm_tools": true
      },
      "guarded": {
        "path_validation": true,
        "command_blocklist": true,
        "sandbox": "proot",
        "confirm_tools": false,
        "allowed_paths": ["./"],
        "blocked_network": true
      },
      "paranoid": {
        "path_validation": true,
        "command_blocklist": true,
        "sandbox": "proot",
        "confirm_tools": true,
        "blocked_network": true,
        "audit_log": true
      }
    }
  }
}
```

#### Customizing Blocklists

The built-in blocklists are defaults. Users can extend or override them in `uwsal.json`:

```json
{
  "security": {
    "profile": "cautious",
    "blocked_paths": ["/data/production/"],
    "blocked_commands": ["\\bsystemctl\\b"],
    "allowed_paths": ["/etc/hosts"],
    "allowed_commands": ["rm -rf ./build"]
  }
}
```

- **`blocked_*`** entries are *added* to the built-in defaults
- **`allowed_*`** entries create *exceptions* — an allowed entry overrides a blocked match
- Resolution order: allowed list checked first, then blocked list, then built-in defaults

This means you can run with the default `cautious` profile but whitelist specific paths your project needs:

```json
{
  "security": {
    "profile": "cautious",
    "allowed_paths": ["/etc/nginx/conf.d/", "/var/log/myapp/"]
  }
}
```

Or add project-specific blocks without touching the defaults:

```json
{
  "security": {
    "blocked_paths": ["/data/production/db/"],
    "blocked_commands": ["\\bdrop\\s+database\\b"]
  }
}
```

#### CLI Flags

```
--security-profile PROFILE   Set security profile (open/cautious/guarded/paranoid)
--no-security                Alias for --security-profile open
--security-blocked-path P    Add path to blocklist (repeatable)
--security-allowed-path P    Add path to allowlist (repeatable)
```

The `--no-security` flag is the explicit "I know what I'm doing" escape hatch. It disables path validation, command blocklists, and confirmation prompts entirely. This is useful for trusted environments, automated pipelines, or when the overhead of validation interferes with a specific workflow.

#### Profile + Override Interaction

CLI flags override config file settings, which override profile defaults:

```
Built-in defaults  ←  profile settings  ←  uwsal.json overrides  ←  CLI flags
```

Example: `cautious` profile enables path validation, but `--no-security` disables everything regardless of what's in the config file.

## Implementation Priority

| Priority | Layer | Effort | Impact | Status |
|----------|-------|--------|--------|--------|
| **P0** | Path validation | Small | Blocks path traversal and credential theft | **Done** |
| **P0** | Command blocklist | Small | Blocks obvious destructive commands | **Done** |
| **P1** | Audit logging | Small | Enables post-hoc review | **Done** |
| **P1** | In-process command proxy | Medium | Defense-in-depth for bash | **Done** |
| **P2** | Security profiles | Medium | Unified configuration | **Done** |
| **P2** | PRoot sandbox | Medium | Filesystem isolation | Planned |
| **P3** | Network restrictions | Large | Requires proot or root access | Planned |
| **P3** | Declarative rule bridge | Large | Connect Prolog specs to Python | Planned |

## Files modified

| File | Change | Status |
|------|--------|--------|
| `tools.py` | Path validation, command blocklist, proxy integration, audit hooks, safe command flow | **Done** |
| `agent_loop.py` | `--security-profile` flag, AuditLogger init, session lifecycle | **Done** |
| `config.py` | Parse `security` section from uwsal.json | **Done** |
| `security/__init__.py` | Security subpackage exports | **Done** |
| `security/audit.py` | JSONL audit logger (basic/detailed/forensic) | **Done** |
| `security/profiles.py` | SecurityProfile dataclass, built-in profiles, safe/confirm command lists | **Done** |
| `security/proxy.py` | CommandProxyManager with per-command rules (rm, curl, git, ssh, etc.) | **Done** |

## Termux-Specific Notes

- **No root**: PRoot works without root (ptrace-based). Firejail, bubblewrap, nsjail need root or specific capabilities.
- **No Docker/Podman**: Termux doesn't support container runtimes.
- **No iptables**: Network isolation requires root. Only option is proot with restricted DNS or transparent proxy via Termux:API.
- **PRoot availability**: `pkg install proot` — already common in Termux for running Linux distros.
- **PATH tricks work**: Termux uses a standard PATH layout, so `~/.agent-loop/bin` prepending works naturally.

## Relationship to Existing Prolog Security

The existing declarative layer (shell_sandbox.pl, rpyc_security.pl, firewall rules) defines *what* should be allowed. This proposal implements *enforcement* in the Python execution layer. A future P3 task could auto-generate the Python blocklists and security profiles from the Prolog specs, closing the loop between declaration and enforcement.

---

Generated with [Claude Code](https://claude.com/claude-code)
