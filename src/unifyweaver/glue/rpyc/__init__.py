"""
UnifyWeaver RPyC Transport Module

Provides network-based RPC with live object proxies through RPyC,
enabling transparent remote Python execution from Prolog.

Proxy Layers:
    - root: Layer 1 - Direct exposed_ method access
    - wrapped_root: Layer 2 - Safe attribute access
    - auto_root: Layer 3 - Automatic wrapping
    - smart_root: Layer 4 - Local-class-aware wrapping

Security Modes:
    - ssh: SSH tunnel (recommended for production)
    - ssl: SSL/TLS with certificates
    - unsecured: Plain TCP (development only)

Example:
    from unifyweaver.glue.rpyc import create_ssh_bridge

    bridge = create_ssh_bridge('server.example.com', user='deploy')
    with bridge.secure_connection() as proxy:
        result = proxy.modules.numpy.array([1, 2, 3])
        print(proxy.root.exposed_compute(result))
"""

from .remote_execution import (
    # Main classes
    ConfigurableRPyCBridge,
    SecureRPyCProxy,
    SecurityError,

    # Proxy layers
    RootProxy,
    ClientModuleWrapper,
    ClientModuleAdapter,
    AutoWrapProxy,
    SmartAutoWrapProxy,

    # Local utilities
    LocalMethods,
    DictWrapper,

    # Async support
    AsyncResult,
    async_call,

    # Factory functions
    create_ssh_bridge,
    create_ssl_bridge,
    create_unsecured_bridge,

    # Server generation
    generate_rpyc_server_script,

    # Utility functions
    is_rpyc_proxy,
    is_simple_type,
    setup_logging,
)

from .dict_wrapper import (
    DictWrapper,
    mkDictWrapper,
    wrapped_exec,
    call_function,
    call_with_args,
    call_with_kwargs,
    call_with_mixed_args,
    call_and_wrap,
    wrap,
    get_dict_value,
    set_dict_value,
    BoundCallable,
)

__all__ = [
    # Main classes
    'ConfigurableRPyCBridge',
    'SecureRPyCProxy',
    'SecurityError',

    # Proxy layers
    'RootProxy',
    'ClientModuleWrapper',
    'ClientModuleAdapter',
    'AutoWrapProxy',
    'SmartAutoWrapProxy',

    # Local utilities
    'LocalMethods',
    'DictWrapper',

    # Async support
    'AsyncResult',
    'async_call',

    # Factory functions
    'create_ssh_bridge',
    'create_ssl_bridge',
    'create_unsecured_bridge',

    # Server generation
    'generate_rpyc_server_script',

    # Utility functions
    'is_rpyc_proxy',
    'is_simple_type',
    'setup_logging',

    # Dict wrapper utilities
    'mkDictWrapper',
    'wrapped_exec',
    'call_function',
    'call_with_args',
    'call_with_kwargs',
    'call_with_mixed_args',
    'call_and_wrap',
    'wrap',
    'get_dict_value',
    'set_dict_value',
    'BoundCallable',
]

__version__ = '0.1.0'
