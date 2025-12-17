# Client-Server Architecture: Implementation Plan

## Overview

This document outlines the phased implementation approach for adding client-server capabilities to UnifyWeaver.

## Implementation Principles

1. **Incremental**: Each phase delivers working functionality
2. **Backward Compatible**: Existing pipelines continue to work
3. **Test-Driven**: Each feature has integration tests
4. **Multi-Target**: Features work across Python, Go, Rust

## Phase Overview

| Phase | Focus | Deliverables |
|-------|-------|--------------|
| **Phase 1** | In-Process Services | Service definition, basic call_service |
| **Phase 2** | Stateful Services | State operations, service lifecycle |
| **Phase 3** | Cross-Process | Unix socket transport |
| **Phase 4** | Error Handling | Timeouts, retries, fallbacks |
| **Phase 5** | Network Transport | HTTP/TCP services |
| **Phase 6** | Advanced Features | Streaming, async, discovery |

---

## Phase 1: In-Process Services

### Goal
Enable defining and calling services within a single process.

### Deliverables

#### 1.1 Service Definition Syntax

```prolog
% New directive: service/2
service(echo, [
    receive(Message),
    respond(Message)
]).

service(double, [
    receive(N),
    transform(N, Result, Result is N * 2),
    respond(Result)
]).
```

#### 1.2 Service Validation

**File: `src/unifyweaver/core/service_validation.pl`** (new)

```prolog
:- module(service_validation, [
    validate_service/2,
    is_valid_service/1,
    is_valid_handler_spec/1
]).

validate_service(service(Name, HandlerSpec), Errors) :-
    ( atom(Name) -> E1 = [] ; E1 = [error(invalid_service_name, Name)] ),
    ( is_valid_handler_spec(HandlerSpec) -> E2 = [] ; E2 = [error(invalid_handler)] ),
    append(E1, E2, Errors).
```

#### 1.3 Pipeline Integration: call_service Stage

**File: `src/unifyweaver/core/pipeline_validation.pl`** (modify)

```prolog
% Add validation for call_service stage
is_valid_stage(call_service(ServiceName, _RequestExpr, _ResponseVar)) :-
    atom(ServiceName).
```

#### 1.4 Python Target: Service Compilation

**File: `src/unifyweaver/targets/python_target.pl`** (modify)

```prolog
% Generate service class
compile_service_to_python(service(Name, HandlerSpec), PythonCode) :-
    generate_service_handler(HandlerSpec, HandlerCode),
    format(string(PythonCode), "
class ~wService:
    def __init__(self):
        self.state = {}

    def call(self, request):
~w
        return response

_services['~w'] = ~wService()
", [Name, HandlerCode, Name, Name]).

% Generate call_service stage
generate_stage_flow(call_service(ServiceName, RequestExpr, ResponseVar), InVar, OutVar, Code) :-
    format(string(Code), "
    # Call service: ~w
    def _call_~w(record):
        request = record.get('~w')
        response = _services['~w'].call(request)
        record['~w'] = response
        return record
    ~w = map(_call_~w, ~w)
", [ServiceName, ServiceName, RequestExpr, ServiceName, ResponseVar, OutVar, ServiceName, InVar]).
```

#### 1.5 Go Target: Service Compilation

**File: `src/unifyweaver/targets/go_target.pl`** (modify)

```go
// Generated service structure
type EchoService struct {
    state map[string]interface{}
}

func (s *EchoService) Call(request Record) Record {
    // Handler implementation
    return request
}

var services = map[string]Service{
    "echo": &EchoService{state: make(map[string]interface{})},
}
```

#### 1.6 Integration Tests

**File: `tests/integration/test_in_process_services.sh`** (new)

```bash
# Test 1: Service definition validation
# Test 2: Service compilation to Python
# Test 3: Service compilation to Go
# Test 4: call_service in pipeline
# Test 5: Service with transformation
# Test 6: Multiple services in pipeline
```

### Files Changed (Phase 1)

| File | Action | Description |
|------|--------|-------------|
| `src/unifyweaver/core/service_validation.pl` | Create | Service validation module |
| `src/unifyweaver/core/pipeline_validation.pl` | Modify | Add call_service validation |
| `src/unifyweaver/targets/python_target.pl` | Modify | Service compilation, call_service stage |
| `src/unifyweaver/targets/go_target.pl` | Modify | Service compilation, call_service stage |
| `src/unifyweaver/targets/rust_target.pl` | Modify | Service compilation, call_service stage |
| `tests/integration/test_in_process_services.sh` | Create | Integration tests |
| `CHANGELOG.md` | Modify | Document new feature |

---

## Phase 2: Stateful Services

### Goal
Enable services that maintain state between requests.

### Deliverables

#### 2.1 State Operations

```prolog
service(counter, [stateful(true)], [
    receive(Op),
    route_by(Op, [
        (increment, [
            state_modify(count, succ),
            state_get(count, Value),
            respond(Value)
        ]),
        (get, [
            state_get(count, Value),
            respond(Value)
        ])
    ])
]).
```

#### 2.2 State Management

**File: `src/unifyweaver/core/service_state.pl`** (new)

```prolog
:- module(service_state, [
    init_service_state/2,
    state_get/3,
    state_put/3,
    state_modify/3
]).
```

#### 2.3 Python State Implementation

```python
class StatefulService:
    def __init__(self, initial_state=None):
        self.state = initial_state or {}

    def state_get(self, key, default=None):
        return self.state.get(key, default)

    def state_put(self, key, value):
        self.state[key] = value

    def state_modify(self, key, func):
        self.state[key] = func(self.state.get(key))
```

### Files Changed (Phase 2)

| File | Action | Description |
|------|--------|-------------|
| `src/unifyweaver/core/service_state.pl` | Create | State management |
| `src/unifyweaver/core/service_validation.pl` | Modify | Validate state operations |
| `src/unifyweaver/targets/python_target.pl` | Modify | Stateful service generation |
| `tests/integration/test_stateful_services.sh` | Create | State tests |

---

## Phase 3: Cross-Process Communication

### Goal
Enable services to run in separate processes, communicating via Unix sockets.

### Deliverables

#### 3.1 Transport Configuration

```prolog
service(user_lookup, [
    transport(unix_socket('/tmp/user_service.sock'))
], [
    receive(UserId),
    lookup_user/1,
    respond(UserRecord)
]).
```

#### 3.2 Socket Server Generation

```python
# Generated server code
import socket
import json

def run_user_lookup_server(socket_path):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(socket_path)
    sock.listen(5)

    service = UserLookupService()

    while True:
        conn, _ = sock.accept()
        try:
            for line in conn.makefile():
                request = json.loads(line)
                response = service.call(request['payload'])
                conn.sendall(json.dumps({
                    '__type': 'response',
                    '__id': request['__id'],
                    '__status': 'ok',
                    'payload': response
                }).encode() + b'\n')
        finally:
            conn.close()
```

#### 3.3 Socket Client Generation

```python
# Generated client code
import socket
import json
import uuid

class UnixSocketClient:
    def __init__(self, socket_path):
        self.socket_path = socket_path

    def call(self, request):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(self.socket_path)
        try:
            request_id = str(uuid.uuid4())
            sock.sendall(json.dumps({
                '__type': 'request',
                '__id': request_id,
                'payload': request
            }).encode() + b'\n')

            response = json.loads(sock.makefile().readline())
            if response['__status'] == 'error':
                raise ServiceError(response['error'])
            return response['payload']
        finally:
            sock.close()
```

### Files Changed (Phase 3)

| File | Action | Description |
|------|--------|-------------|
| `src/unifyweaver/core/transport.pl` | Create | Transport abstraction |
| `src/unifyweaver/transports/unix_socket.pl` | Create | Unix socket implementation |
| `src/unifyweaver/targets/python_target.pl` | Modify | Socket client/server generation |
| `tests/integration/test_socket_services.sh` | Create | Cross-process tests |

---

## Phase 4: Error Handling

### Goal
Add robust error handling: timeouts, retries, fallbacks, circuit breakers.

### Deliverables

#### 4.1 Client Options

```prolog
pipeline([
    parse/1,
    call_service(slow_service, request, response, [
        timeout(5000),           % 5 second timeout
        retry(3),                % Retry up to 3 times
        retry_delay(100),        % 100ms between retries
        fallback(default_value), % Use fallback on failure
        circuit_breaker(5, 60)   % Open after 5 failures, reset after 60s
    ]),
    output/1
]).
```

#### 4.2 Error Types

```python
class ServiceError(Exception):
    pass

class TimeoutError(ServiceError):
    pass

class ConnectionError(ServiceError):
    pass

class CircuitOpenError(ServiceError):
    pass
```

#### 4.3 Retry Logic

```python
def call_with_retry(client, request, options):
    max_retries = options.get('retry', 0)
    timeout = options.get('timeout')
    fallback = options.get('fallback')

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return client.call(request, timeout=timeout)
        except ServiceError as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(options.get('retry_delay', 0) / 1000)

    if fallback is not None:
        return fallback
    raise last_error
```

### Files Changed (Phase 4)

| File | Action | Description |
|------|--------|-------------|
| `src/unifyweaver/core/service_validation.pl` | Modify | Validate client options |
| `src/unifyweaver/core/error_handling.pl` | Create | Error types and handling |
| `src/unifyweaver/targets/python_target.pl` | Modify | Retry/timeout generation |
| `tests/integration/test_service_errors.sh` | Create | Error handling tests |

---

## Phase 5: Network Transport

### Goal
Enable services over HTTP and TCP.

### Deliverables

#### 5.1 HTTP Transport

```prolog
service(api_service, [
    transport(http('/api/service', [port(8080)]))
], [
    receive(Request),
    process/1,
    respond(Response)
]).
```

#### 5.2 HTTP Server (Flask/Gin/Actix)

```python
# Python: Flask
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/<service_name>', methods=['POST'])
def handle_request(service_name):
    service = _services.get(service_name)
    req = request.json
    response = service.call(req['payload'])
    return jsonify({
        '__type': 'response',
        '__id': req['__id'],
        '__status': 'ok',
        'payload': response
    })
```

```go
// Go: Standard library
func handleService(w http.ResponseWriter, r *http.Request) {
    serviceName := mux.Vars(r)["service"]
    service := services[serviceName]

    var req Request
    json.NewDecoder(r.Body).Decode(&req)

    response := service.Call(req.Payload)

    json.NewEncoder(w).Encode(Response{
        Type:    "response",
        ID:      req.ID,
        Status:  "ok",
        Payload: response,
    })
}
```

### Files Changed (Phase 5)

| File | Action | Description |
|------|--------|-------------|
| `src/unifyweaver/transports/http.pl` | Create | HTTP transport |
| `src/unifyweaver/transports/tcp.pl` | Create | TCP transport |
| `src/unifyweaver/targets/python_target.pl` | Modify | HTTP client/server |
| `src/unifyweaver/targets/go_target.pl` | Modify | HTTP client/server |
| `tests/integration/test_http_services.sh` | Create | Network tests |

---

## Phase 6: Advanced Features

### Goal
Add streaming responses, async calls, service discovery.

### Deliverables

#### 6.1 Streaming Responses

```prolog
service(data_stream, [streaming(true)], [
    receive(Query),
    stream_results/1,  % Yields multiple responses
    end_stream
]).
```

#### 6.2 Async Calls

```prolog
pipeline([
    parse/1,
    call_service_async(slow_service, request, future),
    do_other_work/1,
    await_service(future, response),
    output/1
]).
```

#### 6.3 Service Discovery

```prolog
% Register with discovery service
service(user_service, [
    register(consul, [
        name('user-service'),
        tags(['v1', 'production']),
        health_check('/health')
    ])
], [...]).

% Discover service at runtime
pipeline([
    call_service(discover(consul, 'user-service'), request, response),
    output/1
]).
```

---

## Testing Strategy

### Unit Tests

Each module has unit tests for:
- Validation logic
- Code generation
- State management

### Integration Tests

Each phase includes integration tests:

```bash
tests/integration/
├── test_in_process_services.sh     # Phase 1
├── test_stateful_services.sh       # Phase 2
├── test_socket_services.sh         # Phase 3
├── test_service_errors.sh          # Phase 4
├── test_http_services.sh           # Phase 5
└── test_advanced_services.sh       # Phase 6
```

### End-to-End Tests

Real-world scenarios:
- Pipeline calling multiple services
- Service calling other services
- Error recovery scenarios
- Performance benchmarks

---

## Migration Guide

### For Existing Pipelines

Existing pipelines work unchanged. The `call_service` stage is additive.

### Adopting Services

```prolog
% Before: Inline lookup
pipeline([
    parse/1,
    lookup_user/1,  % Inline predicate
    output/1
]).

% After: Service-based lookup
service(user_lookup, [
    receive(UserId),
    query_database/1,
    respond(UserRecord)
]).

pipeline([
    parse/1,
    call_service(user_lookup, user_id, user_data),
    output/1
]).
```

Benefits of migration:
- Reusable across pipelines
- Can be deployed separately
- Easier to test in isolation
- Can add caching, retries, etc.

---

## Timeline Estimates

| Phase | Complexity | Estimate |
|-------|------------|----------|
| Phase 1 | Medium | Foundation work |
| Phase 2 | Medium | Builds on Phase 1 |
| Phase 3 | High | New transport layer |
| Phase 4 | Medium | Applies existing patterns |
| Phase 5 | High | Network complexity |
| Phase 6 | High | Advanced patterns |

**Recommended approach**: Complete Phase 1-2 first, which provides immediate value. Phases 3-6 can be implemented based on user needs.

---

## Success Metrics

1. **Phase 1 Complete**: Can define and call in-process services
2. **Phase 2 Complete**: Services can maintain state
3. **Phase 3 Complete**: Services can run in separate processes
4. **Phase 4 Complete**: Robust error handling
5. **Phase 5 Complete**: Network-accessible services
6. **Phase 6 Complete**: Production-ready service infrastructure

---

*This implementation plan is a living document. Adjust phases based on user feedback and priorities.*
