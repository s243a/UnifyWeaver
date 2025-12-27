# Proposal: RPyC Transport for Cross-Target Glue

**Status:** Draft
**Author:** John William Creighton (@s243a)
**Co-Author:** Claude Code (Opus 4.5)
**Date:** 2025-12-26
**Related:** `janus_glue.pl`, `cross-target-glue/`, JanusBridge project

## Summary

Add RPyC (Remote Python Call) as a transport type in the cross-target glue system. Unlike pipe-based IPC which serializes data between processes, RPyC provides network-based RPC with live object proxies, enabling transparent remote Python execution with full module access.

## Motivation

### Current Limitations

The existing cross-target glue supports three transports:

| Transport | Location | Objects | Security |
|-----------|----------|---------|----------|
| `pipe` | Same machine | Serialized (TSV/JSON) | Process isolation |
| `http` | Network | Serialized (JSON) | TLS optional |
| `janus` | In-process | Live (shared memory) | None (same process) |

**Gaps:**
1. No live object access over network (http serializes everything)
2. No bidirectional calls (pipe is parent→child)
3. No transparent module access on remote machines
4. Janus is SWI-Prolog specific and in-process only

### What RPyC Provides

RPyC fills these gaps:

| Feature | Pipe | HTTP | Janus | RPyC |
|---------|------|------|-------|------|
| Remote machine | No | Yes | No | Yes |
| Live object proxies | No | No | Yes | Yes |
| Bidirectional calls | No | No | Yes | Yes |
| Full module access | No | No | Yes | Yes |
| Security options | Process | TLS | N/A | SSH/SSL/None |

### JanusBridge Precedent

The JanusBridge project (in `context/projects/JanusBridge`) already implements sophisticated RPyC integration with:
- `ConfigurableRPyCBridge` - Connection factory with security options
- `SecureRPyCProxy` - Four-layer proxy wrapping system
- SSH/SSL/unsecured tunnel modes
- `dict_wrapper.py` - Reliable dictionary exchange
- Prolog tutorials (`appendix_c_remote.pl`)

This proposal leverages JanusBridge's proven patterns.

## Use Cases

### 1. Remote Computation Offloading

Run compute-heavy Python (NumPy, ML inference) on a dedicated server:

```prolog
% Connect to GPU server
setup_rpyc_connection('gpu-server.local', [security(ssh), user(admin)], Proxy),

% Execute NumPy on remote
py_call(Proxy:modules:numpy, NP),
py_call(NP:linalg:solve(A, B), Solution).
```

### 2. Distributed Pipeline Stages

Different pipeline stages on different machines:

```prolog
compile_pipeline([
    step(ingest, python, read_data/1, [host(localhost), transport(pipe)]),
    step(process, python, transform/1, [host('worker-1'), transport(rpyc)]),
    step(analyze, python, ml_predict/1, [host('gpu-server'), transport(rpyc)]),
    step(store, python, write_db/1, [host(localhost), transport(pipe)])
], Code).
```

### 3. Development with Remote Environments

Access a remote Python environment (specific versions, packages) from local Prolog:

```prolog
% Connect to Python 3.11 environment on dev server
setup_rpyc_connection('dev.internal', [
    security(ssh),
    python_path('/opt/python311/bin/python')
], Proxy).
```

### 4. Multi-Tenant Python Services

Share a Python service across multiple Prolog clients:

```prolog
% Multiple clients connect to shared ML service
setup_rpyc_connection('ml-service.prod', [
    security(ssl),
    certfile('client.pem')
], Proxy).
```

## Proposed Design

### Transport Registration

```prolog
% In glue/target_registry.pl
transport(rpyc).
transport_properties(rpyc, [
    location(network),
    serialization(none),      % Live proxies, not serialized
    bidirectional(true),
    security([ssh, ssl, unsecured])
]).

valid_transport(python, python, rpyc).
valid_transport(prolog, python, rpyc).
```

### Core Module: `rpyc_glue.pl`

```prolog
:- module(rpyc_glue, [
    % Connection management
    rpyc_connect/3,           % +Host, +Options, -Proxy
    rpyc_disconnect/1,        % +Proxy
    rpyc_async_connect/3,     % +Host, +Options, -AsyncProxy

    % Remote execution (sync)
    rpyc_call/4,              % +Proxy, +Module, +Function, -Result
    rpyc_exec/3,              % +Proxy, +Code, -Namespace

    % Remote execution (async)
    rpyc_async_call/4,        % +Proxy, +Module, +Function, -AsyncResult
    rpyc_await/2,             % +AsyncResult, -Result
    rpyc_ready/1,             % +AsyncResult (succeeds if ready)

    % Module access
    rpyc_import/3,            % +Proxy, +ModuleName, -ModuleRef

    % Proxy layers
    rpyc_root/2,              % +Proxy, -RootProxy
    rpyc_wrapped_root/2,      % +Proxy, -WrappedRoot
    rpyc_auto_root/2,         % +Proxy, -AutoRoot
    rpyc_smart_root/2,        % +Proxy, -SmartRoot

    % Code generation
    generate_rpyc_client/3,   % +Predicates, +Options, -Code
    generate_rpyc_service/3,  % +Predicates, +Options, -ServiceCode
    generate_rpyc_server/3    % +ServiceName, +Options, -ServerScript
]).
```

### Service Generation Example

```prolog
% Generate RPyC service from predicate declarations
?- generate_rpyc_service([
       exposed(transform_data/2, [input(list), output(list)]),
       exposed(predict/2, [input(dict), output(float)]),
       exposed(train_model/3, [input(list), input(dict), output(model)])
   ], [
       service_name('MLService'),
       imports([numpy, sklearn]),
       whitelist(true)
   ], ServiceCode).
```

Generated Python:
```python
import rpyc
from rpyc.utils.server import ThreadedServer
import numpy
import sklearn

class MLService(rpyc.Service):
    ALIASES = ['MLService']

    def exposed_transform_data(self, input_list):
        # Generated from transform_data/2
        ...

    def exposed_predict(self, input_dict):
        # Generated from predict/2
        ...

    def exposed_train_model(self, data, config):
        # Generated from train_model/3
        ...

if __name__ == '__main__':
    server = ThreadedServer(MLService, port=18812)
    server.start()
```

### Security Modes

Following JanusBridge's model:

```prolog
% SSH tunnel (recommended for production)
rpyc_connect('server.com', [
    security(ssh),
    user('deploy'),
    ssh_key('/path/to/key'),
    remote_port(18812)
], Proxy).

% SSL/TLS (certificate-based)
rpyc_connect('server.com', [
    security(ssl),
    certfile('client.pem'),
    keyfile('client.key'),
    ca_certs('ca.pem')
], Proxy).

% Unsecured (development only - with warnings)
rpyc_connect('localhost', [
    security(unsecured),
    acknowledge_risk(true)  % Must explicitly acknowledge
], Proxy).
```

### Proxy Layers

Adopt JanusBridge's layered proxy system:

| Layer | Purpose | When to Use |
|-------|---------|-------------|
| `root` | Direct `exposed_` method access | Simple RPC calls |
| `wrapped_root` | Safe attribute access | Prevent accidental execution |
| `auto_root` | Automatic wrapping | General use |
| `smart_root` | Local-class-aware wrapping | Code generation |

### Integration with Existing Glue

```prolog
% Compile step with RPyC transport
compile_step(step(Name, python, Pred/Arity), Options, Code) :-
    member(transport(rpyc), Options),
    member(host(Host), Options),
    generate_rpyc_step(Name, Pred/Arity, Host, Options, Code).
```

## Comparison with Existing Transports

### When to Use Each

```
Decision Tree:
    │
    ├─ Same machine, same process?
    │   └─ Yes → janus (if SWI-Prolog) or pipe
    │
    ├─ Same machine, process isolation needed?
    │   └─ Yes → pipe
    │
    ├─ Remote machine, simple request/response?
    │   └─ Yes → http
    │
    └─ Remote machine, live objects or bidirectional?
        └─ Yes → rpyc
```

### Performance Characteristics

| Transport | Latency | Throughput | Setup Cost |
|-----------|---------|------------|------------|
| janus | ~1μs | Highest | None |
| pipe | ~1ms | High | Process spawn |
| rpyc | ~10ms | Medium | TCP + auth |
| http | ~50ms | Medium | TCP + TLS + parse |

### Object Handling

| Transport | Python Dict | NumPy Array | Custom Class |
|-----------|-------------|-------------|--------------|
| pipe | Serialize (JSON) | Serialize (slow) | Not supported |
| http | Serialize (JSON) | Serialize (slow) | Not supported |
| janus | Zero-copy | Zero-copy | Live reference |
| rpyc | Live proxy | Live proxy | Live proxy |

## Implementation Plan

### Phase 1: Foundation (Low Risk)

1. Create `src/unifyweaver/glue/rpyc_glue.pl`
2. Add basic connection management (unsecured mode first)
3. Implement `rpyc_call/4` for simple function calls
4. Add tests with local RPyC server

**Dependency:** JanusBridge's `remote_execution.py`

### Phase 2: Security (Medium Risk)

1. Add SSH tunnel support
2. Add SSL/TLS support
3. Implement security warnings for unsecured mode
4. Add connection validation

### Phase 3: Proxy System (Medium Risk)

1. Port JanusBridge's proxy layers
2. Implement `rpyc_import/3` for module access
3. Add `wrapped_exec` integration
4. Handle dict_wrapper for reliable data exchange

### Phase 4: Code Generation (Low Risk)

1. `generate_rpyc_client/3` - Generate Prolog client code
2. `generate_rpyc_service/3` - Generate Python service code
3. Integration with `compile_pipeline/3`

### Phase 5: Documentation & Examples

1. Add to cross-target glue book (Chapter 22?)
2. Create example project
3. Add to API reference

## Dependencies

### Required
- JanusBridge project (`context/projects/JanusBridge`)
  - `remote_execution.py`
  - `dict_wrapper.py`
- RPyC Python package (`pip install rpyc`)

### Optional
- Paramiko (for SSH tunneling without system ssh)
- SSL certificates (for SSL mode)

## Security Considerations

### Threat Model

| Threat | Mitigation |
|--------|------------|
| Eavesdropping | SSH/SSL encryption |
| Man-in-the-middle | Certificate verification |
| Unauthorized access | SSH key auth or SSL client certs |
| Code injection | Server-side whitelisting (optional) |

### Explicit Security Acknowledgment

For unsecured connections, require explicit acknowledgment:

```prolog
rpyc_connect('localhost', [
    security(unsecured),
    acknowledge_risk(true),    % Required
    allowed_hosts([localhost]) % Restrict to localhost
], Proxy).
```

### Production Recommendations

1. **Always use SSH or SSL** for non-localhost connections
2. **Use SSH keys** instead of passwords
3. **Enable server whitelisting** for exposed methods
4. **Log all connections** for audit

## Design Decisions (Resolved)

1. **Dependency management**: Adapt JanusBridge code directly
   - JanusBridge is maintained by the same author (@s243a)
   - Copy and adapt `remote_execution.py` and `dict_wrapper.py` into UnifyWeaver
   - Modify as needed for UnifyWeaver's specific requirements

2. **Proxy layer complexity**: Implement all four layers
   - RootProxy → ClientModuleWrapper → AutoWrapProxy → SmartAutoWrapProxy
   - Full flexibility for different use cases
   - Matches JanusBridge's proven architecture

3. **Server generation**: Yes, generate from predicates
   - `generate_rpyc_service/3` will create Python RPyC services
   - Input: list of predicates to expose
   - Output: complete Python service code with `exposed_` methods

4. **Async support**: Yes, implement async mode
   - RPyC supports async via `rpyc.async_()` wrapper
   - Add `async(true)` option to connection
   - Enables **event-based programming** patterns
   - Useful for high-concurrency and non-blocking pipelines

### Async and Event-Based Programming

Async support enables reactive/event-driven architectures:

```prolog
% Register callback for remote events
rpyc_on_event(Proxy, 'data_ready', handle_data/1),
rpyc_on_event(Proxy, 'model_trained', handle_model/1),

% Start async operations
rpyc_async_call(Proxy, ml, train_model, [Data], TrainFuture),
rpyc_async_call(Proxy, etl, process_batch, [Batch], ProcessFuture),

% Non-blocking check
(   rpyc_ready(TrainFuture)
->  rpyc_await(TrainFuture, Model),
    save_model(Model)
;   continue_other_work
).

% Or use callbacks (event-based)
handle_data(Data) :-
    process_incoming(Data),
    update_dashboard(Data).

handle_model(Model) :-
    deploy_model(Model),
    notify_stakeholders(Model).
```

This enables:
- **Reactive pipelines**: Respond to events as they occur
- **Non-blocking I/O**: Don't wait for slow operations
- **Parallel execution**: Multiple async operations simultaneously
- **Event streaming**: Subscribe to continuous data streams
- **Callbacks from Python**: Python can notify Prolog of events

## Alternatives Considered

### gRPC
- **Pro:** Industry standard, many languages
- **Con:** Requires proto definitions, no live objects

### ZeroMQ
- **Pro:** High performance, flexible patterns
- **Con:** Low-level, requires serialization

### Pyro5
- **Pro:** Similar to RPyC, simpler
- **Con:** Less mature security, smaller community

### Direct sockets
- **Pro:** Maximum control
- **Con:** Significant implementation effort

**Decision:** RPyC chosen because JanusBridge already has working implementation and it provides live object proxies.

## References

### Internal
- `context/projects/JanusBridge/src/core/remote_execution.py`
- `context/projects/JanusBridge/src/core/dict_wrapper.py`
- `src/unifyweaver/glue/janus_glue.pl`
- `education/book-07-cross-target-glue/`

### External
- [RPyC Documentation](https://rpyc.readthedocs.io/)
- [JanusBridge Appendix C Tutorial](context/Obsidian/JanusBridge/Chapters/Appendix_C/)

## Acknowledgements

This proposal builds on the extensive work in the JanusBridge project, which provides the foundation for RPyC integration with Prolog via Janus.
