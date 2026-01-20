# Skill: RPyC (Remote Python Call)

Network-based Python RPC with live object proxies, bidirectional calls, and multiple security modes.

## When to Use

- User asks "how do I call Python from Prolog over the network?"
- User needs remote Python service calls
- User wants process isolation between Prolog and Python
- User needs async Python execution

## Quick Start

```prolog
:- use_module('src/unifyweaver/glue/rpyc_glue').

% Connect to local RPyC server
rpyc_connect('localhost', [
    security(unsecured),
    acknowledge_risk(true)
], Proxy).

% Call a function
rpyc_call(Proxy, numpy, mean, [[1, 2, 3, 4, 5]], Result).
% Result = 3.0

% Disconnect
rpyc_disconnect(Proxy).
```

## Connection Management

### Connect

```prolog
rpyc_connect(Host, Options, Proxy).
```

**Security Modes:**
- `security(ssh)` - SSH tunnel (secure, requires SSH access)
- `security(ssl)` - SSL/TLS (secure, requires certificates)
- `security(unsecured)` - No encryption (requires `acknowledge_risk(true)`)

**Common Options:**
- `remote_port(Port)` - Server port (default: 18812)
- `acknowledge_risk(true)` - Required for unsecured mode

**SSH Options:**
- `ssh_user(User)` - SSH username
- `ssh_port(Port)` - SSH port (default: 22)
- `ssh_key(Path)` - Path to SSH key file

**SSL Options:**
- `keyfile(Path)` - Client key file
- `certfile(Path)` - Client certificate file
- `ca_certs(Path)` - CA certificates file

### Connection Examples

```prolog
% Secure SSH connection
rpyc_connect('server.example.com', [
    security(ssh),
    ssh_user(admin),
    ssh_key('/home/user/.ssh/id_rsa'),
    remote_port(18812)
], Proxy).

% SSL connection
rpyc_connect('api.example.com', [
    security(ssl),
    certfile('/certs/client.pem'),
    keyfile('/certs/client.key'),
    ca_certs('/certs/ca.pem')
], Proxy).

% Local development (unsecured)
rpyc_connect('localhost', [
    security(unsecured),
    acknowledge_risk(true)
], Proxy).
```

### Scoped Connection

```prolog
rpyc_with_connection(Host, Options, Goal).
```

Executes Goal with connection, ensuring cleanup:
```prolog
rpyc_with_connection('localhost',
    [security(unsecured), acknowledge_risk(true)],
    (rpyc_call(Proxy, math, sqrt, [16], R), writeln(R))
).
```

### Disconnect

```prolog
rpyc_disconnect(Proxy).
```

## Remote Execution

### Synchronous Call

```prolog
rpyc_call(Proxy, Module, Function, Args, Result).
```

**Examples:**
```prolog
% Math operations
rpyc_call(Proxy, math, sqrt, [16], R).    % R = 4.0
rpyc_call(Proxy, math, pow, [2, 10], R).  % R = 1024.0

% NumPy operations
rpyc_call(Proxy, numpy, mean, [[1,2,3,4,5]], R).  % R = 3.0
rpyc_call(Proxy, numpy, std, [[1,2,3,4,5]], R).   % R = 1.414...

% Custom module
rpyc_call(Proxy, mymodule, process, [Data], Result).
```

### Asynchronous Call

```prolog
rpyc_async_call(Proxy, Module, Function, Args, AsyncResult).
rpyc_await(AsyncResult, Result).
rpyc_ready(AsyncResult).  % Succeeds if result is ready
```

**Example:**
```prolog
% Start long-running computation
rpyc_async_call(Proxy, scipy, optimize, [func, x0], AsyncResult).

% Do other work...

% Wait for result
rpyc_await(AsyncResult, OptimizationResult).
```

### Execute Python Code

```prolog
rpyc_exec(Proxy, Code, Namespace).
```

**Example:**
```prolog
rpyc_exec(Proxy, "
import numpy as np
result = np.array([1,2,3]) * 2
", Namespace).
```

## Module Access

### Import Module

```prolog
rpyc_import(Proxy, ModuleName, ModuleRef).
```

### Get Nested Module

```prolog
rpyc_get_module(Proxy, 'scipy.optimize', ModuleRef).
```

## Proxy Layers

RPyC provides four proxy layers for different use cases:

| Layer | Purpose | Use Case |
|-------|---------|----------|
| `root` | Direct `exposed_` method access | Simple RPC |
| `wrapped_root` | Safe attribute access | Prevent accidental execution |
| `auto_root` | Automatic wrapping | General use |
| `smart_root` | Local-class-aware wrapping | Code generation |

```prolog
rpyc_root(Proxy, RootProxy).
rpyc_wrapped_root(Proxy, WrappedRoot).
rpyc_auto_root(Proxy, AutoRoot).
rpyc_smart_root(Proxy, SmartRoot).
```

## Code Generation

### Generate Client Code

```prolog
generate_rpyc_client(Predicates, Options, Code).
```

Generates Python client code for calling predicates over RPyC.

### Generate Service Code

```prolog
generate_rpyc_service(Predicates, Options, ServiceCode).
```

Generates Python RPyC service that exposes predicates.

### Generate Server Script

```prolog
generate_rpyc_server(Options, ServerScript).
```

Generates standalone RPyC server startup script.

## Error Handling

```prolog
% Catch connection errors
catch(
    rpyc_connect('unavailable.server', Opts, Proxy),
    Error,
    (format('Connection failed: ~w~n', [Error]), fail)
).

% Catch call errors
catch(
    rpyc_call(Proxy, module, nonexistent_func, [], R),
    Error,
    (format('Call failed: ~w~n', [Error]), R = error)
).
```

## Requirements

- Python 3.8+ with `rpyc` package
- Janus for Prolog-Python bridge
- For SSH: `ssh` command (and optionally `sshpass`)
- For SSL: Valid certificates

## Testing

```prolog
test_rpyc_glue.
```

## Related

**Parent Skill:**
- `skill_ipc.md` - IPC sub-master

**Sibling Skills:**
- `skill_pipe_communication.md` - Unix pipes
- `skill_python_bridges.md` - Cross-runtime embedding

**Code:**
- `src/unifyweaver/glue/rpyc_glue.pl`
- `src/unifyweaver/glue/rpyc_security.pl` - Security whitelisting
