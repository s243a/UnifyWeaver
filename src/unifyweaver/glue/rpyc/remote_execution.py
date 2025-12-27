"""
remote_execution.py - RPyC Remote Execution for UnifyWeaver

Adapted from JanusBridge project by John William Creighton (s243a)

Provides network-based RPC with live object proxies through RPyC,
supporting SSH/SSL/unsecured connection modes and four proxy layers.
"""

import rpyc
import subprocess
import time
import getpass
import os
import stat
import pathlib
import traceback
import types
import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any, Callable

# UnifyWeaver logger
logger = logging.getLogger("unifyweaver.rpyc")


def setup_logging(level=logging.INFO, format_str=None):
    """
    Configure logging for unifyweaver.rpyc module.

    Args:
        level: Logging level (default: INFO)
        format_str: Custom format string for log messages
    """
    format_str = format_str or "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=format_str)
    logger.setLevel(level)


# Import DictWrapper from local module
try:
    from .dict_wrapper import DictWrapper, wrapped_exec, call_function
except ImportError:
    from dict_wrapper import DictWrapper, wrapped_exec, call_function


class ConfigurableRPyCBridge:
    """
    RPyC bridge with configurable security tunneling.

    Supports three connection modes:
    - ssh: SSH tunnel (recommended for production)
    - ssl: SSL/TLS with certificates
    - unsecured: Plain TCP (development only)
    """

    def __init__(self, host: str, tunnel_type: str = 'ssh',
                 acknowledge_risk: bool = False, **kwargs):
        """
        Initialize RPyC bridge.

        Args:
            host: Remote host address
            tunnel_type: Connection security mode ('ssh', 'ssl', 'unsecured')
            acknowledge_risk: Required for unsecured connections
            **kwargs: Additional configuration options
        """
        self.host = host
        self.tunnel_type = tunnel_type
        self.connection = None
        self.ssh_process = None
        self.acknowledge_risk = acknowledge_risk
        self.config = self._setup_config(**kwargs)

        # Security check for unsecured connections
        if tunnel_type == 'unsecured':
            self._validate_unsecured_connection()

    def _setup_config(self, **kwargs) -> Dict[str, Any]:
        """Setup configuration based on tunnel type."""
        config = {
            'remote_port': kwargs.get('remote_port', 18812),
            'local_port': kwargs.get('local_port', 18813),
        }

        if self.tunnel_type == 'ssh':
            config.update({
                'ssh_user': kwargs.get('ssh_user', getpass.getuser()),
                'ssh_password': kwargs.get('ssh_password'),
                'ssh_port': kwargs.get('ssh_port', 22),
                'ssh_key': kwargs.get('ssh_key'),
            })
        elif self.tunnel_type == 'ssl':
            config.update({
                'keyfile': kwargs.get('keyfile', 'client.key'),
                'certfile': kwargs.get('certfile', 'client.cert'),
                'ca_certs': kwargs.get('ca_certs', 'ca.cert')
            })

        return config

    def _validate_unsecured_connection(self):
        """Validate that unsecured connection is explicitly acknowledged."""
        if not self.acknowledge_risk:
            raise SecurityError(
                "Unsecured connections require acknowledge_risk=True. "
                "This mode should only be used for development or isolated environments."
            )

        logger.warning(
            "UNSECURED CONNECTION: Data transmitted in plain text. "
            "Only use for development or trusted networks."
        )

    @contextmanager
    def secure_connection(self):
        """Context manager for secure RPyC connections."""
        try:
            if self.tunnel_type == 'ssh':
                with self._ssh_tunnel():
                    yield self._connect_via_tunnel()
            elif self.tunnel_type == 'ssl':
                yield self._connect_ssl()
            elif self.tunnel_type == 'unsecured':
                yield self._connect_unsecured()
            else:
                raise ValueError(f"Unknown tunnel type: {self.tunnel_type}")
        finally:
            self._cleanup()

    def connect(self) -> 'SecureRPyCProxy':
        """
        Establish connection without context manager.

        Returns:
            SecureRPyCProxy: Connected proxy object

        Note: Caller is responsible for calling close() when done.
        """
        if self.tunnel_type == 'ssh':
            self._start_ssh_tunnel()
            return self._connect_via_tunnel()
        elif self.tunnel_type == 'ssl':
            return self._connect_ssl()
        elif self.tunnel_type == 'unsecured':
            return self._connect_unsecured()
        else:
            raise ValueError(f"Unknown tunnel type: {self.tunnel_type}")

    def _start_ssh_tunnel(self):
        """Start SSH tunnel without context manager."""
        ssh_user = self.config['ssh_user']
        ssh_port = self.config['ssh_port']
        local_port = self.config['local_port']
        remote_port = self.config['remote_port']

        ssh_cmd = [
            'ssh', '-L', f'{local_port}:localhost:{remote_port}',
            '-p', str(ssh_port),
            '-N',
            f'{ssh_user}@{self.host}'
        ]

        # Add SSH key if specified
        if self.config.get('ssh_key'):
            ssh_cmd.extend(['-i', self.config['ssh_key']])

        # Add password via sshpass if specified
        if self.config.get('ssh_password'):
            ssh_cmd = ['sshpass', '-p', self.config['ssh_password']] + ssh_cmd

        logger.info(f"Creating SSH tunnel: {ssh_user}@{self.host}:{ssh_port}")

        self.ssh_process = subprocess.Popen(
            ssh_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        time.sleep(3)

        if self.ssh_process.poll() is not None:
            stderr = self.ssh_process.stderr.read().decode()
            raise ConnectionError(f"SSH tunnel failed: {stderr}")

        logger.info("SSH tunnel established")

    @contextmanager
    def _ssh_tunnel(self):
        """Create SSH tunnel for secure RPyC connection."""
        try:
            self._start_ssh_tunnel()
            yield
        finally:
            if self.ssh_process:
                self.ssh_process.terminate()
                self.ssh_process.wait()
                logger.info("SSH tunnel closed")

    def _connect_via_tunnel(self) -> 'SecureRPyCProxy':
        """Connect to RPyC server via SSH tunnel."""
        local_port = self.config['local_port']
        logger.info(f"Connecting to RPyC via tunnel (localhost:{local_port})")

        conn = rpyc.connect('localhost', local_port)
        return SecureRPyCProxy(conn, self.tunnel_type, self.config)

    def _connect_ssl(self) -> 'SecureRPyCProxy':
        """Connect to RPyC server via SSL/TLS."""
        logger.info(f"Connecting to RPyC via SSL ({self.host})")

        conn = rpyc.ssl_connect(
            self.host,
            self.config['remote_port'],
            keyfile=self.config['keyfile'],
            certfile=self.config['certfile'],
            ca_certs=self.config['ca_certs']
        )
        return SecureRPyCProxy(conn, self.tunnel_type, self.config)

    def _connect_unsecured(self) -> 'SecureRPyCProxy':
        """Connect to RPyC server without security."""
        remote_port = self.config['remote_port']
        logger.warning(f"Connecting UNSECURED to {self.host}:{remote_port}")

        try:
            conn = rpyc.classic.connect(self.host, port=remote_port)
            logger.info("TCP connection established")
            return SecureRPyCProxy(conn, self.tunnel_type, self.config)
        except Exception as e:
            error_msg = (
                f"Failed to connect to RPyC server at {self.host}:{remote_port}.\n"
                "Ensure the server is running."
            )
            raise ConnectionError(error_msg) from e

    def _cleanup(self):
        """Cleanup resources."""
        if self.ssh_process:
            try:
                self.ssh_process.terminate()
                self.ssh_process.wait(timeout=3)
            except Exception as e:
                logger.warning(f"SSH cleanup warning: {e}")
            self.ssh_process = None

        if self.connection:
            try:
                self.connection.close()
            except Exception:
                pass
            self.connection = None


class SecurityError(Exception):
    """Raised when security requirements are not met."""
    pass


class LocalMethods:
    """Client-side methods that can use remote or local execution."""

    def __init__(self, proxy: 'SecureRPyCProxy'):
        self._proxy = proxy

    def wrapped_exec(self, code: str, initial_vars: Optional[Dict] = None) -> DictWrapper:
        """
        Execute Python code remotely, returning results in a DictWrapper.

        Args:
            code: Python code string to execute
            initial_vars: Optional initial variables dictionary

        Returns:
            DictWrapper containing execution namespace
        """
        try:
            remote_dict_wrapper = self._proxy.conn.modules['dict_wrapper']
            result = remote_dict_wrapper.wrapped_exec(code, initial_vars)
            return DictWrapper(result.dict if hasattr(result, 'dict') else result)
        except Exception:
            return self._basic_remote_exec(code, initial_vars)

    def _basic_remote_exec(self, code: str, initial_vars: Optional[Dict] = None) -> DictWrapper:
        """Basic remote code execution without dict_wrapper."""
        namespace = initial_vars or {}
        remote_builtins = self._proxy.conn.modules.builtins
        remote_builtins.exec(code, remote_builtins.__dict__, namespace)
        return DictWrapper(namespace)

    def call_function(self, func: Callable, *args, **kwargs):
        """Call a function (remote or local) with arguments."""
        return func(*args, **kwargs)

    def __getattr__(self, name: str):
        """Route to remote exposed methods if not found locally."""
        if hasattr(self._proxy.conn.root, 'exposed_' + name):
            def remote_method(*args, **kwargs):
                return getattr(self._proxy.conn.root, 'exposed_' + name)(*args, **kwargs)
            return remote_method
        raise AttributeError(f"No local or remote method named '{name}'")


# =============================================================================
# Proxy Layer 1: RootProxy - Direct exposed_ method access
# =============================================================================

class RootProxy:
    """
    Layer 1: Direct access to exposed_ methods on the remote server.

    Use for simple RPC calls where you know the method name.
    """

    def __init__(self, parent_proxy: 'SecureRPyCProxy'):
        self._parent_proxy = parent_proxy

    def __getattr__(self, name: str):
        exposed_name = 'exposed_' + name
        proxy = self._parent_proxy

        if hasattr(proxy.conn.root, exposed_name):
            def remote_method(*args, **kwargs):
                result = getattr(proxy.conn.root, exposed_name)(*args, **kwargs)
                logger.debug(f"{exposed_name}({args}, {kwargs}) -> {result}")
                return result
            return remote_method

        # Fallback to local_methods
        if hasattr(proxy.local_methods, name):
            return getattr(proxy.local_methods, name)

        raise AttributeError(f"No remote or local method named '{name}'")


# =============================================================================
# Proxy Layer 2: ClientModuleWrapper - Safe attribute access
# =============================================================================

class ClientModuleAdapter:
    """Client-side adapter for remote module proxies."""

    def __init__(self, remote_proxy):
        self._remote = remote_proxy

    def __getattr__(self, name: str):
        result = getattr(self._remote, name)
        return ClientModuleWrapper(result, name)


class ClientModuleWrapper:
    """
    Layer 2: Safe attribute access wrapper.

    Prevents accidental execution by wrapping all attribute access.
    Use when you need to navigate remote objects safely.
    """

    def __init__(self, remote_proxy, display_name: Optional[str] = None):
        object.__setattr__(self, '_proxy', remote_proxy)
        object.__setattr__(self, '_display_name',
                          display_name or self._get_default_name(remote_proxy))

    def _get_default_name(self, obj) -> str:
        if hasattr(obj, '__name__'):
            return obj.__name__
        if hasattr(obj, '__class__'):
            return obj.__class__.__name__
        return str(type(obj))

    def __getattr__(self, name: str):
        _proxy = object.__getattribute__(self, '_proxy')
        unwrapped = getattr(_proxy, name)
        return ClientModuleWrapper(unwrapped, getattr(unwrapped, '__name__', name))

    def __call__(self, *args, **kwargs):
        _proxy = object.__getattribute__(self, '_proxy')
        result = _proxy(*args, **kwargs)
        return ClientModuleWrapper(result)

    def __str__(self):
        _display_name = object.__getattribute__(self, '_display_name')
        _proxy = object.__getattribute__(self, '_proxy')
        return f"<ClientModuleWrapper for '{_display_name}' [{type(_proxy).__name__}]>"

    def __repr__(self):
        return self.__str__()


# =============================================================================
# Proxy Layer 3: AutoWrapProxy - Automatic wrapping
# =============================================================================

def is_rpyc_proxy(obj) -> bool:
    """Check if an object is an RPyC proxy (netref)."""
    if hasattr(obj, '__class__') and getattr(obj.__class__, '__module__', '').startswith('rpyc.core.netref'):
        return True
    try:
        if isinstance(obj, rpyc.core.netref.BaseNetref):
            return True
    except Exception:
        pass
    return False


def is_simple_type(obj) -> bool:
    """Check if object is a simple type that doesn't need wrapping."""
    return isinstance(obj, (int, float, str, bytes, bool, type(None), list, dict, tuple, set))


def is_routing_remote_method(obj) -> bool:
    """Detect if obj is a local routing function from RootProxy."""
    return (
        type(obj) is types.FunctionType and
        getattr(obj, '__name__', '') == 'remote_method' and
        'RootProxy.__getattr__' in getattr(obj, '__qualname__', '')
    )


class AutoWrapProxy:
    """
    Layer 3: Automatically wraps RPyC proxies for better usability.

    General-purpose proxy that automatically wraps remote objects
    to provide consistent access patterns.
    """

    def __init__(self, remote_proxy, name: Optional[str] = None,
                 pre_should_wrap: Optional[Callable] = None,
                 post_should_wrap: Optional[Callable] = None):
        object.__setattr__(self, '_remote', remote_proxy)
        object.__setattr__(self, '_name', name or self._get_safe_name(remote_proxy))
        object.__setattr__(self, 'pre_should_wrap', pre_should_wrap or (lambda x: False))
        object.__setattr__(self, 'post_should_wrap', post_should_wrap or (lambda x, y: y))

    def _get_safe_name(self, obj) -> str:
        """Safely get a name for the object."""
        try:
            if hasattr(obj, '__name__'):
                return obj.__name__
            elif hasattr(obj, '__class__'):
                return obj.__class__.__name__
            else:
                return str(type(obj).__name__)
        except Exception:
            return 'UnknownProxy'

    def __getattr__(self, name: str):
        if name == '__call__':
            return object.__getattribute__(self, '__call__')

        _remote = object.__getattribute__(self, '_remote')
        attr = getattr(_remote, name)

        if self.should_wrap(attr):
            _name = object.__getattribute__(self, '_name')
            return self.wrap(attr, _name)
        return attr

    def __call__(self, *args, **kwargs):
        _remote = object.__getattribute__(self, '_remote')
        result = _remote(*args, **kwargs)

        if self.should_wrap(result):
            return self.wrap(result)
        return result

    def should_wrap(self, obj) -> bool:
        """Determine if object should be wrapped."""
        pre_should_wrap = object.__getattribute__(self, 'pre_should_wrap')
        post_should_wrap = object.__getattribute__(self, 'post_should_wrap')

        wrap_value = False

        if pre_should_wrap(obj):
            wrap_value = True
        elif is_rpyc_proxy(obj) and not is_simple_type(obj):
            wrap_value = True
        elif is_routing_remote_method(obj):
            wrap_value = True

        return post_should_wrap(obj, wrap_value)

    def wrap(self, obj, name: Optional[str] = None):
        """Create a new wrapped instance."""
        if name is None:
            name = self._get_safe_name(obj)
        return self.__class__(obj, name)


# =============================================================================
# Proxy Layer 4: SmartAutoWrapProxy - Local-class-aware wrapping
# =============================================================================

class SmartAutoWrapProxy(AutoWrapProxy):
    """
    Layer 4: Smart proxy that only wraps when class isn't available locally.

    Useful for code generation scenarios where you want to use local
    classes when available but proxy remote ones.
    """

    def __init__(self, remote_proxy, name: Optional[str] = None,
                 connection=None):
        super().__init__(remote_proxy, name)
        object.__setattr__(self, '_connection', connection)
        object.__setattr__(self, '_local_class', self._check_local_class())

    def _check_local_class(self):
        """Check if the remote object's class exists locally."""
        try:
            _remote = object.__getattribute__(self, '_remote')
            remote_class = getattr(_remote, '__class__', None)
            if not remote_class:
                return None

            remote_class_name = getattr(remote_class, '__name__', None)
            remote_module_name = getattr(remote_class, '__module__', None)

            if not (remote_class_name and remote_module_name):
                return None

            import importlib
            try:
                module = importlib.import_module(remote_module_name)
                return getattr(module, remote_class_name, None)
            except (ImportError, AttributeError):
                return None
        except Exception:
            return None

    def should_wrap(self, obj) -> bool:
        """Override to consider local class availability."""
        _local_class = object.__getattribute__(self, '_local_class')
        if _local_class is not None:
            try:
                if isinstance(obj, _local_class):
                    return False
            except Exception:
                pass
        return super().should_wrap(obj)

    def get_wrapped_smartly(self, module_name: str):
        """Get a module with smart wrapping based on local availability."""
        _connection = object.__getattribute__(self, '_connection')
        if _connection:
            checker = self._create_class_checker()
            module = _connection.root.exposed_import_with_smart_wrap(
                module_name,
                checker
            )
            if self.should_wrap(module):
                return self.wrap(module, module_name)
            return module
        return None

    def _create_class_checker(self) -> Callable:
        """Create a function for server callback to check local class availability."""
        def check_class_locally(class_info):
            import importlib
            try:
                module = importlib.import_module(class_info['module'])
                return hasattr(module, class_info['name'])
            except ImportError:
                return False
        return check_class_locally

    def __getattr__(self, name: str):
        if name == "import_with_smart_wrap":
            return self.get_wrapped_smartly
        return super().__getattr__(name)


# =============================================================================
# SecureRPyCProxy - Main proxy class with all layers
# =============================================================================

class SecureRPyCProxy:
    """
    Main proxy for RPyC connections with four access layers.

    Properties:
        root: Layer 1 - Direct exposed_ method access
        wrapped_root: Layer 2 - Safe attribute access
        auto_root: Layer 3 - Automatic wrapping
        smart_root: Layer 4 - Local-class-aware wrapping
        modules: Direct access to remote Python modules
    """

    def __init__(self, rpyc_connection, tunnel_type: str, config: Optional[Dict] = None):
        self.conn = rpyc_connection
        self.tunnel_type = tunnel_type
        self.config = config or {}
        self.local_methods = LocalMethods(self)
        self._validate_connection()

    def _validate_connection(self):
        """Validate the RPyC connection is working."""
        try:
            if not hasattr(self.conn, 'modules'):
                raise AttributeError("Not a valid RPyC connection")

            # Test basic connection
            _ = self.conn.modules.sys
            logger.info("RPyC connection validated")

            # Check for dict_wrapper availability
            try:
                _ = self.conn.modules['dict_wrapper']
                logger.debug("dict_wrapper available remotely")
            except Exception:
                logger.debug("dict_wrapper not available remotely")

        except Exception as e:
            logger.error(f"RPyC connection validation failed: {e}")
            raise

    @property
    def root(self) -> RootProxy:
        """Layer 1: Direct exposed_ method access."""
        return RootProxy(self)

    @property
    def wrapped_root(self) -> ClientModuleAdapter:
        """Layer 2: Safe attribute access wrapper."""
        return ClientModuleAdapter(self.root)

    @property
    def auto_root(self) -> AutoWrapProxy:
        """Layer 3: Automatic wrapping proxy."""
        return AutoWrapProxy(self.root)

    @property
    def smart_root(self) -> SmartAutoWrapProxy:
        """Layer 4: Smart local-class-aware proxy."""
        return SmartAutoWrapProxy(self.root, connection=self.conn)

    @property
    def modules(self):
        """Direct access to remote Python modules."""
        if not hasattr(self.conn, 'modules'):
            raise AttributeError("No RPyC server connection")
        return self.conn.modules

    def get_module(self, full_name: str):
        """Import a module on the remote server."""
        remote_importlib = self.conn.modules.importlib
        return remote_importlib.import_module(full_name)

    def close(self):
        """Close the RPyC connection."""
        if self.conn:
            self.conn.close()


# =============================================================================
# Async Support
# =============================================================================

class AsyncResult:
    """
    Wrapper for RPyC async results.

    Provides a consistent interface for async operations.
    """

    def __init__(self, async_result):
        self._result = async_result

    @property
    def ready(self) -> bool:
        """Check if the result is ready."""
        return self._result.ready

    def wait(self, timeout: Optional[float] = None):
        """Wait for the result to be ready."""
        self._result.wait(timeout)

    @property
    def value(self):
        """Get the result value (blocks if not ready)."""
        return self._result.value

    @property
    def error(self):
        """Get any error that occurred."""
        return self._result.error


def async_call(proxy: SecureRPyCProxy, module_name: str,
               func_name: str, args: tuple = (), kwargs: dict = None) -> AsyncResult:
    """
    Make an asynchronous call to a remote function.

    Args:
        proxy: SecureRPyCProxy connection
        module_name: Name of the remote module
        func_name: Name of the function to call
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        AsyncResult that can be awaited
    """
    kwargs = kwargs or {}
    module = proxy.get_module(module_name)
    func = getattr(module, func_name)

    # RPyC async wrapper
    async_func = rpyc.async_(func)
    result = async_func(*args, **kwargs)

    return AsyncResult(result)


# =============================================================================
# Factory Functions
# =============================================================================

def create_ssh_bridge(host: str, user: Optional[str] = None,
                      password: Optional[str] = None, **kwargs) -> ConfigurableRPyCBridge:
    """
    Create SSH-tunneled RPyC bridge (recommended for production).

    Args:
        host: Remote host address
        user: SSH username (default: current user)
        password: SSH password (optional, prefer key-based auth)
        **kwargs: Additional options (ssh_port, ssh_key, remote_port, etc.)
    """
    config = {'ssh_user': user or getpass.getuser()}
    if password:
        config['ssh_password'] = password
    config.update(kwargs)
    return ConfigurableRPyCBridge(host, tunnel_type='ssh', **config)


def create_ssl_bridge(host: str, keyfile: Optional[str] = None,
                      certfile: Optional[str] = None,
                      ca_certs: Optional[str] = None, **kwargs) -> ConfigurableRPyCBridge:
    """
    Create SSL/TLS RPyC bridge.

    Args:
        host: Remote host address
        keyfile: Path to client key file
        certfile: Path to client certificate file
        ca_certs: Path to CA certificates file
    """
    config = {}
    if keyfile:
        config['keyfile'] = keyfile
    if certfile:
        config['certfile'] = certfile
    if ca_certs:
        config['ca_certs'] = ca_certs
    config.update(kwargs)
    return ConfigurableRPyCBridge(host, tunnel_type='ssl', **config)


def create_unsecured_bridge(host: str, acknowledge_risk: bool = False,
                           **kwargs) -> ConfigurableRPyCBridge:
    """
    Create unsecured RPyC bridge (development/isolated environments only).

    Args:
        host: Remote host address
        acknowledge_risk: Must be True to use unsecured connections

    Warning:
        Only use for development, testing, or isolated virtual machines.
        Data is transmitted in plain text with no authentication.
    """
    return ConfigurableRPyCBridge(
        host,
        tunnel_type='unsecured',
        acknowledge_risk=acknowledge_risk,
        **kwargs
    )


# =============================================================================
# Server Generation
# =============================================================================

def generate_rpyc_server_script(output_file: str = 'rpyc_server.py',
                                service_name: str = 'UnifyWeaverService') -> str:
    """
    Generate an RPyC server setup script.

    Args:
        output_file: Output file path
        service_name: Name for the generated service class

    Returns:
        Path to generated file
    """
    server_script = f'''#!/usr/bin/env python3
"""
RPyC Server for UnifyWeaver Remote Integration
Generated by UnifyWeaver RPyC Transport
"""

import rpyc
from rpyc.utils.server import ThreadedServer
from rpyc.core.service import ClassicService
import sys
import os


class {service_name}(ClassicService):
    """RPyC service with UnifyWeaver integration."""

    def on_connect(self, conn):
        print(f"Client connected")
        self._setup_environment()

    def on_disconnect(self, conn):
        print("Client disconnected")

    def _setup_environment(self):
        """Setup environment on the server."""
        # Add paths for UnifyWeaver modules
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Try to import dict_wrapper
        try:
            import dict_wrapper
            print("+ dict_wrapper loaded")
        except ImportError as e:
            print(f"- dict_wrapper not found: {{e}}")


def start_server(port: int = 18812, mode: str = 'unsecured'):
    """Start RPyC server."""
    print("=" * 50)
    print(f"Starting {{mode.upper()}} RPyC server on port {{port}}")
    print("=" * 50)

    if mode == 'unsecured':
        print("WARNING: This server provides UNSECURED access!")
        print("Only use for development or isolated environments.")

    server = ThreadedServer(
        {service_name},
        port=port,
        protocol_config={{"allow_all_attrs": True}}
    )
    print(f"RPyC server listening on port {{port}}")
    print("Press Ctrl+C to stop")

    try:
        server.start()
    except KeyboardInterrupt:
        print("\\nStopping server...")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='UnifyWeaver RPyC Server')
    parser.add_argument('--mode', choices=['unsecured', 'ssl'], default='unsecured',
                       help='Server security mode')
    parser.add_argument('--port', type=int, default=18812,
                       help='Server port (default: 18812)')

    args = parser.parse_args()
    start_server(args.port, args.mode)
'''

    pathlib.Path(output_file).write_text(server_script, encoding='utf-8')

    # Make executable on Unix-like systems
    try:
        pathlib.Path(output_file).chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
    except Exception:
        pass

    logger.info(f"RPyC server script generated: {output_file}")
    return output_file
