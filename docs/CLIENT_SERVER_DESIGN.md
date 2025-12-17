# Client-Server Architecture Design

UnifyWeaver's client-server architecture enables service-oriented pipeline composition with transport independence.

## Overview

The core insight: **client-server communication is fundamentally two pipelines going in opposite directions**:
- **Request Pipeline**: Client → Server (carries request data)
- **Response Pipeline**: Server → Client (carries response data)

This allows the same Prolog DSL to define services, and the same compilation infrastructure to generate service implementations across all target languages.

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  service(Name, [receive(X), transform(X, Y), respond(Y)])  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Service Layer                            │
│  - Service Registry (in-process lookup)                     │
│  - Request/Response handling                                │
│  - State management (for stateful services)                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Transport Layer                           │
│  - in_process: Direct function calls                        │
│  - unix_socket: Unix domain sockets                         │
│  - tcp: TCP sockets                                         │
│  - http: HTTP/REST endpoints                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Protocol Layer                            │
│  - jsonl: JSON Lines (streaming)                            │
│  - json: Standard JSON                                      │
│  - messagepack: Binary MessagePack                          │
│  - protobuf: Protocol Buffers                               │
└─────────────────────────────────────────────────────────────┘
```

## Phase 1: In-Process Services (Current)

Phase 1 establishes the foundation with in-process service calls.

### Service Definition

```prolog
% Simple stateless service
service(echo, [
    receive(X),
    respond(X)
]).

% Service with options
service(counter, [stateful(true), timeout(5000)], [
    receive(Cmd),
    state_get(count, Current),
    ( Cmd = increment ->
        NewCount is Current + 1,
        state_put(count, NewCount),
        respond(NewCount)
    ;
        respond(Current)
    )
]).

% Service with transformation
service(user_enricher, [
    receive(UserId),
    lookup_user(UserId, UserData),
    enrich_user(UserData, Enriched),
    respond(Enriched)
]).
```

### Service Operations

| Operation | Description |
|-----------|-------------|
| `receive(Var)` | Bind incoming request to variable |
| `respond(Value)` | Send response to caller |
| `respond_error(Error)` | Send error response |
| `state_get(Key, Value)` | Get state value (stateful services) |
| `state_put(Key, Value)` | Set state value (stateful services) |
| `state_modify(Key, Func)` | Modify state with function |
| `state_delete(Key)` | Delete state key |
| `call_service(Name, Req, Resp)` | Call another service |
| `transform(In, Out, Goal)` | Transform data with predicate |
| `branch(Cond, True, False)` | Conditional execution |
| `route_by(Field, Routes)` | Route by field value |

### Service Options

| Option | Description | Default |
|--------|-------------|---------|
| `stateful(Bool)` | Enable state management | `false` |
| `transport(Type)` | Transport mechanism | `in_process` |
| `protocol(Format)` | Wire format | `jsonl` |
| `timeout(Ms)` | Request timeout | None |
| `max_concurrent(N)` | Max concurrent requests | Unlimited |
| `on_error(Handler)` | Error handler predicate | None |

### Pipeline Integration

Services can be called from pipelines using the `call_service` stage:

```prolog
% Simple service call
pipeline([
    read_json,
    call_service(enricher, record, enriched_record),
    write_json
]).

% With options
pipeline([
    read_json,
    call_service(enricher, record, enriched_record, [
        timeout(5000),
        retry(3),
        retry_delay(100),
        fallback(default_value)
    ]),
    write_json
]).
```

### Generated Code

#### Python

```python
class EchoService(Service):
    def __init__(self):
        self.name = "echo"

    def call(self, request):
        return request

_services["echo"] = EchoService()
```

#### Go

```go
type EchoService struct {
    name string
}

func NewEchoService() *EchoService {
    return &EchoService{name: "echo"}
}

func (s *EchoService) Name() string {
    return s.name
}

func (s *EchoService) Call(request interface{}) (interface{}, error) {
    return request, nil
}

func init() {
    RegisterService("echo", NewEchoService())
}
```

#### Rust

```rust
pub struct EchoService;

impl EchoService {
    pub fn new() -> Self {
        EchoService
    }
}

impl Service for EchoService {
    fn name(&self) -> &str {
        "echo"
    }

    fn call(&self, request: serde_json::Value) -> Result<serde_json::Value, ServiceError> {
        Ok(request)
    }
}

lazy_static::lazy_static! {
    static ref ECHO_SERVICE: std::sync::Arc<EchoService> = {
        let service = std::sync::Arc::new(EchoService::new());
        register_service("echo", service.clone());
        service
    };
}
```

## Future Phases

### Phase 2: Cross-Process Services (Planned)

Unix domain sockets for inter-process communication:

```prolog
service(worker, [transport(unix_socket('/tmp/worker.sock'))], [
    receive(Task),
    process_task(Task, Result),
    respond(Result)
]).
```

### Phase 3: Network Services (Planned)

TCP and HTTP transports:

```prolog
service(api, [transport(tcp('0.0.0.0', 8080)), protocol(jsonl)], [
    receive(Request),
    handle_request(Request, Response),
    respond(Response)
]).

service(rest, [transport(http('/api/v1')), protocol(json)], [
    receive(Request),
    route_by(method, [
        (get, handle_get),
        (post, handle_post)
    ])
]).
```

### Phase 4: Service Mesh (Planned)

Load balancing, circuit breakers, service discovery:

```prolog
service(gateway, [
    load_balance(round_robin),
    circuit_breaker(threshold(5), timeout(30000)),
    retry(3, exponential)
], [
    receive(Request),
    route_to_backend(Request, Response),
    respond(Response)
]).
```

### Phase 5: Multi-Language Polyglot Services (Planned)

Seamless calls between services written in different languages:

```prolog
% Prolog service definition
service(coordinator, [
    receive(Job),
    call_service(python_ml_model, Job, Prediction),    % Python ML service
    call_service(go_validator, Prediction, Validated), % Go validation service
    call_service(rust_db, Validated, Stored),          % Rust database service
    respond(Stored)
]).
```

### Phase 6: Distributed Services (Planned)

Cluster-aware services with automatic routing:

```prolog
service(distributed_cache, [
    distributed(true),
    replication(3),
    consistency(eventual)
], [
    receive(Op),
    route_to_shard(Op, Node),
    execute_on_node(Node, Op, Result),
    respond(Result)
]).
```

## Validation

The system validates service definitions at compile time:

### Service Validation (`service_validation.pl`)

- Service name must be an atom
- Handler spec must be a list
- All operations must be valid
- Options must be valid

### Pipeline Validation (`pipeline_validation.pl`)

- `call_service` stage requires atom service name
- Options must be valid (timeout, retry, retry_delay, fallback)

## Testing

Integration tests verify the implementation:

```bash
./tests/integration/test_in_process_services.sh
```

Tests cover:
1. Service validation module loads
2. Valid service definitions accepted
3. Service with options validates
4. Service operations validated
5. call_service stage validates in pipeline
6. call_service options validated
7. Python service compilation
8. Go service compilation
9. Rust service compilation
10. Python infrastructure included
11. Go infrastructure included
12. Rust infrastructure included
13. Invalid services rejected

## Files

| File | Description |
|------|-------------|
| `src/unifyweaver/core/service_validation.pl` | Service definition validation |
| `src/unifyweaver/core/pipeline_validation.pl` | Extended with call_service validation |
| `src/unifyweaver/targets/python_target.pl` | Python service compilation |
| `src/unifyweaver/targets/go_target.pl` | Go service compilation |
| `src/unifyweaver/targets/rust_target.pl` | Rust service compilation |
| `tests/integration/test_in_process_services.sh` | Integration tests |
| `docs/CLIENT_SERVER_DESIGN.md` | This document |
