"""Security subsystem for UnifyWeaver Agent Loop.

Provides audit logging, enhanced security profiles, command proxying,
PATH-based wrapper scripts, and proot filesystem isolation.
"""

from .audit import AuditLogger
from .profiles import SecurityProfile, get_profile, get_builtin_profiles
from .proxy import CommandProxyManager
from .path_proxy import PathProxyManager
from .proot_sandbox import ProotSandbox, ProotConfig

__all__ = [
    'AuditLogger',
    'SecurityProfile',
    'get_profile',
    'get_builtin_profiles',
    'CommandProxyManager',
    'PathProxyManager',
    'ProotSandbox',
    'ProotConfig',
]
