# PR: Cross-Target Glue Phase 5 - Network Layer

## Title
feat: Add cross-target glue Phase 5 - network layer

## Summary

Phase 5 of the cross-target glue system implements remote target communication via HTTP and sockets. This enables distributed processing across networked services, completing the cross-target communication story from in-process to local pipes to remote calls.

## Changes

### New Module: `src/unifyweaver/glue/network_glue.pl`

**Service Registry:**
- `register_service/3` - Register services with URL and options (timeout, retries, auth)
- `service/2`, `service_options/2` - Query registered services
- `unregister_service/1` - Remove from registry
- `endpoint_url/3` - Construct full URLs for endpoints

**HTTP Server Generation:**

| Target | Framework | Features |
|--------|-----------|----------|
| Go | net/http | CORS middleware, JSON Request/Response |
| Python | Flask | flask-cors, error handlers |
| Rust | Actix-web | actix-cors, serde integration |

All servers share consistent API:
```json
Request:  {"data": <any>}
Response: {"success": bool, "data": <any>, "error": <string?>}
```

**HTTP Client Generation:**

| Target | Library | Features |
|--------|---------|----------|
| Go | net/http | Configurable timeout, json.RawMessage |
| Python | requests | ServiceError exception, type hints |
| Bash | curl/jq | Shell functions for each endpoint |

**Socket Communication:**
- `generate_socket_server/4` - TCP servers (Go/Python)
- `generate_socket_client/4` - TCP clients
- Configurable buffer sizes (default 64KB)
- Go: goroutines per connection
- Python: threading with daemon threads

**Network Pipeline:**
- `generate_network_pipeline/3` - Mix local and remote steps
- Supports Python, Go, and Bash orchestrators
- Step types: `local` (inline code), `remote` (HTTP call)

### Integration Tests: `tests/integration/glue/test_network_glue.pl`

90 test assertions covering:
- Service registry (5 tests)
- Go HTTP server (10 tests)
- Python HTTP server (9 tests)
- Rust HTTP server (9 tests)
- Go HTTP client (8 tests)
- Python HTTP client (9 tests)
- Bash HTTP client (9 tests)
- Socket server (10 tests)
- Socket client (8 tests)
- Network pipeline (15 tests)

### Example: `examples/network-glue/`

Distributed microservices pipeline:

```
┌─────────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐
│   Gateway   │───▶│Transform │───▶│    ML    │───▶│Aggregator │
│    (Go)     │    │ (Python) │    │   (Go)   │    │  (Rust)   │
│   :8080     │    │  :8081   │    │  :8082   │    │   :8083   │
└─────────────┘    └──────────┘    └──────────┘    └───────────┘
```

**Generated Files:**
- `gateway.go` - API routing
- `transform_service.py` - Data normalization
- `ml_service.go` - Prediction/classification
- `aggregator_service.rs` - High-performance stats
- `client.py` - Python client library
- `docker-compose.yml` - Container orchestration

## Test Results

```
=== Network Glue Integration Tests ===

Test: Service registry
  ✓ Service registered
  ✓ Service options stored
  ✓ Endpoint URL constructed
  ✓ Second service registered
  ✓ Service unregistered

Test: Go HTTP server generation
  ✓ Go has package main
  ✓ Go imports net/http
  ✓ Go has Request/Response types
  ✓ Go has CORS middleware
  ✓ Go registers handlers
  ... (10 tests)

Test: Python HTTP server generation
  ✓ Python imports Flask
  ✓ Python imports CORS
  ✓ Python has route decorator
  ... (9 tests)

Test: Rust HTTP server generation
  ✓ Rust uses actix_web
  ✓ Rust uses serde
  ✓ Rust has handler function
  ... (9 tests)

Test: Go HTTP client generation
  ✓ Go has callService
  ✓ Go uses POST
  ✓ Go has full URL
  ... (8 tests)

Test: Python HTTP client generation
  ✓ Python imports requests
  ✓ Python has ServiceError class
  ... (9 tests)

Test: Bash HTTP client generation
  ✓ Bash uses curl
  ✓ Bash uses jq
  ... (9 tests)

Test: Socket server generation
  ✓ Go listens on TCP
  ✓ Go uses goroutines
  ✓ Python uses threading
  ... (10 tests)

Test: Socket client generation
  ✓ Go has client struct
  ✓ Python supports context manager
  ... (8 tests)

Test: Network pipeline generation
  ✓ Python has fetch step
  ✓ Python documents remote/local
  ✓ Bash uses curl
  ... (15 tests)

All tests passed!
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    network_glue.pl                       │
├─────────────────────────────────────────────────────────┤
│  Service Registry                                        │
│  ├── register_service/3                                 │
│  ├── service/2, service_options/2                       │
│  ├── unregister_service/1                               │
│  └── endpoint_url/3                                     │
├─────────────────────────────────────────────────────────┤
│  HTTP Server Generation                                  │
│  ├── generate_go_http_server/3                          │
│  ├── generate_python_http_server/3                      │
│  └── generate_rust_http_server/3                        │
├─────────────────────────────────────────────────────────┤
│  HTTP Client Generation                                  │
│  ├── generate_go_http_client/3                          │
│  ├── generate_python_http_client/3                      │
│  └── generate_bash_http_client/3                        │
├─────────────────────────────────────────────────────────┤
│  Socket Communication                                    │
│  ├── generate_socket_server/4                           │
│  └── generate_socket_client/4                           │
├─────────────────────────────────────────────────────────┤
│  Network Pipeline                                        │
│  └── generate_network_pipeline/3                        │
└─────────────────────────────────────────────────────────┘
```

## Relationship to Previous Phases

| Phase | Communication | Transport |
|-------|---------------|-----------|
| 1 | Foundation | Registry, mapping |
| 2 | Shell scripts | Unix pipes (TSV) |
| 3 | .NET ecosystem | In-process |
| 4 | Native binaries | Unix pipes (TSV/JSON) |
| **5** | **Remote services** | **HTTP, TCP sockets** |

## Key Design Decisions

1. **Consistent API format**: All HTTP services use same Request/Response JSON schema
2. **Multiple target support**: Generate servers/clients for Go, Python, Rust, Bash
3. **Socket option**: Low-latency alternative to HTTP for streaming
4. **Service registry**: Central tracking of service URLs and options
5. **Pipeline abstraction**: Mix local and remote steps seamlessly

## Usage Example

```prolog
% Register services
register_service(ml_api, 'http://ml:8080', [timeout(60)]).

% Generate server
generate_go_http_server(
    [endpoint('/predict', predict, [])],
    [port(8080)],
    ServerCode
).

% Generate client
generate_python_http_client(
    [service_def(ml, 'http://ml:8080', ['/predict'])],
    [],
    ClientCode
).

% Generate network pipeline
generate_network_pipeline(
    [
        step(fetch, remote, 'http://api/data', []),
        step(process, local, 'result = transform(data)', [])
    ],
    [language(python)],
    PipelineCode
).
```

## Next Steps (Phase 6)

- Error handling and retry mechanisms
- Circuit breaker pattern
- Metrics collection (Prometheus)
- Performance profiling hooks
- Production deployment guide
