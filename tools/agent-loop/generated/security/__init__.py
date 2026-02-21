"""Security subsystem for UnifyWeaver Agent Loop.

Provides audit logging, enhanced security profiles, and command proxying.
"""

from .audit import AuditLogger
from .profiles import SecurityProfile, get_profile, get_builtin_profiles
from .proxy import CommandProxyManager

__all__ = [
    'AuditLogger',
    'SecurityProfile',
    'get_profile',
    'get_builtin_profiles',
    'CommandProxyManager',
]
