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

## Phase 2: Cross-Process Services (Current)

Phase 2 implements Unix socket transport for inter-process communication.

### Definition

```prolog
% Unix socket service definition
service(worker, [transport(unix_socket('/tmp/worker.sock'))], [
    receive(Task),
    process_task(Task, Result),
    respond(Result)
]).

% Stateful Unix socket service
service(session, [transport(unix_socket('/tmp/session.sock')), stateful(true), timeout(60000)], [
    receive(Cmd),
    state_get(data, Current),
    handle_cmd(Cmd, Current, New),
    state_put(data, New),
    respond(New)
]).
```

### JSONL Protocol

Request/response uses JSONL (JSON Lines) format:

```json
// Request
{"_id": "uuid-123", "_payload": {"action": "process", "data": [1,2,3]}}

// Success Response
{"_id": "uuid-123", "_status": "ok", "_payload": {"result": 6}}

// Error Response
{"_status": "error", "_error_type": "service_error", "_message": "Processing failed"}
```

### Generated Code Examples

#### Python Server

```python
class WorkerService(Service):
    def __init__(self):
        super().__init__('worker', stateful=False)
        self.socket_path = '/tmp/worker.sock'
        self.timeout = 30.0

    def start_server(self):
        # Create Unix socket and listen
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(5)
        # Handle connections in threads...

    def _process_request(self, conn, line):
        request = json.loads(line.decode('utf-8'))
        request_id = request.get('_id')
        payload = request.get('_payload', request)
        response = self.call(payload)
        self._send_response(conn, request_id, response)
```

#### Python Client

```python
class WorkerClient:
    def connect(self):
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.connect(self.socket_path)

    def call(self, request):
        request_id = str(uuid.uuid4())
        msg = {'_id': request_id, '_payload': request}
        self._socket.sendall((json.dumps(msg) + '\n').encode('utf-8'))
        # Read and return response...
```

### Helper Predicates

```prolog
% Extract transport configuration
get_service_transport(service(_, [transport(unix_socket('/tmp/test.sock'))], _), T).
% T = unix_socket('/tmp/test.sock')

% Check if service is cross-process
is_cross_process_service(service(test, [transport(unix_socket(_))], [])).
% true

% Get protocol (defaults to jsonl)
get_service_protocol(service(_, [], _), P).
% P = jsonl
```

## Phase 3: Network Services (Current)

Phase 3 implements TCP and HTTP transports for network service communication.

### TCP Services

TCP services use JSONL protocol for streaming communication over network sockets.

```prolog
% TCP service definition
service(api, [transport(tcp('0.0.0.0', 8080)), protocol(jsonl)], [
    receive(Request),
    handle_request(Request, Response),
    respond(Response)
]).

% Stateful TCP service
service(counter, [transport(tcp('0.0.0.0', 8081)), stateful(true)], [
    receive(Cmd),
    state_get(count, Current),
    handle_cmd(Cmd, Current, New),
    state_put(count, New),
    respond(New)
]).
```

### HTTP/REST Services

HTTP services provide REST API endpoints with method routing.

```prolog
% HTTP REST service
service(rest, [transport(http('/api/v1')), protocol(json)], [
    receive(Request),
    route_by(method, [
        (get, handle_get),
        (post, handle_post),
        (put, handle_put),
        (delete, handle_delete)
    ])
]).

% HTTP service with custom options
service(webapp, [transport(http('/app', [host('0.0.0.0'), port(3000)])), stateful(true)], [
    receive(Request),
    handle_web_request(Request, Response),
    respond(Response)
]).
```

### Generated Code Examples

#### Python TCP Server

```python
class ApiService(Service):
    def __init__(self):
        super().__init__('api', stateful=False)
        self.host = '0.0.0.0'
        self.port = 8080
        self.timeout = 30000

    def start_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        # Handle connections with JSONL protocol...
```

#### Go HTTP Server

```go
type RestService struct {
    name     string
    host     string
    port     int
    endpoint string
    server   *http.Server
}

func (s *RestService) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    // Handle GET, POST, PUT, DELETE, PATCH methods
    request := map[string]interface{}{
        "method":  r.Method,
        "path":    r.URL.Path,
        "query":   r.URL.Query(),
        "headers": r.Header,
    }
    response, err := s.Call(request)
    // Return JSON response...
}
```

#### Rust TCP Client

```rust
pub struct ApiClient {
    host: String,
    port: u16,
    timeout: Duration,
}

impl ApiClient {
    pub fn call(&self, request: Value) -> Result<Value, ServiceError> {
        let addr = format!("{}:{}", self.host, self.port);
        let mut stream = TcpStream::connect(&addr)?;
        // Send JSONL request, receive response...
    }
}
```

### Transport Categorization

| Transport | Category | Protocol |
|-----------|----------|----------|
| `in_process` | in_process | Direct calls |
| `unix_socket(Path)` | cross_process | JSONL |
| `tcp(Host, Port)` | network | JSONL |
| `http(Endpoint)` | network | JSON |

### Helper Predicates

```prolog
% Check if service is network-based
is_network_service(service(test, [transport(tcp('0.0.0.0', 8080))], [])).
% true

is_network_service(service(test, [transport(http('/api'))], [])).
% true

% TCP service is NOT cross-process (it's network)
is_cross_process_service(service(test, [transport(tcp('0.0.0.0', 8080))], [])).
% false
```

## Phase 4: Service Mesh (Current)

Phase 4 implements service mesh capabilities: load balancing, circuit breakers, and retry with backoff.

### Definition

```prolog
% Service mesh with all features
service(gateway, [
    load_balance(round_robin),
    circuit_breaker(threshold(5), timeout(30000)),
    retry(3, exponential)
], [
    receive(Request),
    handle_request(Request, Response),
    respond(Response)
]).

% Service mesh with custom retry delays
service(resilient_api, [
    load_balance(least_connections),
    circuit_breaker(threshold(3), timeout(10000), half_open_requests(2)),
    retry(5, exponential, [delay(100), max_delay(5000)])
], [
    receive(X),
    process(X, Y),
    respond(Y)
]).
```

### Load Balancing Strategies

| Strategy | Description |
|----------|-------------|
| `round_robin` | Distribute requests evenly across backends |
| `random` | Random backend selection |
| `least_connections` | Select backend with fewest active connections |
| `weighted` | Weight-based distribution |
| `ip_hash` | Consistent hashing based on client IP |

### Circuit Breaker Options

| Option | Description |
|--------|-------------|
| `threshold(N)` | Open circuit after N failures |
| `timeout(Ms)` | Time before attempting half-open state |
| `half_open_requests(N)` | Requests to allow in half-open state |
| `success_threshold(N)` | Successes needed to close circuit |

### Retry Strategies

| Strategy | Description |
|----------|-------------|
| `fixed` | Fixed delay between retries |
| `linear` | Linearly increasing delay |
| `exponential` | Exponentially increasing delay |

### Retry Options

| Option | Description |
|--------|-------------|
| `delay(Ms)` | Base delay between retries (default: 100ms) |
| `max_delay(Ms)` | Maximum delay cap (default: 30000ms) |
| `jitter(true/false)` | Add randomization to delay |

### Generated Code Examples

#### Python Service Mesh

```python
class CircuitState(Enum):
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()

class GatewayService(Service):
    def __init__(self):
        super().__init__('gateway', stateful=False)
        self.backends = []
        self.lb_strategy = 'round_robin'
        self.cb_config = CircuitBreakerConfig(5, 30000)
        self.retry_config = RetryConfig(3, 'exponential', 100, 30000)
        self._circuit_state = CircuitState.CLOSED
        self._failure_count = 0
        self._rr_index = 0

    def _select_backend(self):
        if not self.backends:
            return None
        if self.lb_strategy == 'round_robin':
            idx = self._rr_index % len(self.backends)
            self._rr_index += 1
            return self.backends[idx]
        elif self.lb_strategy == 'random':
            return random.choice(self.backends)
        # ...

    def _check_circuit(self):
        if self._circuit_state == CircuitState.OPEN:
            if time.time() - self._last_failure_time > self.cb_config.timeout / 1000:
                self._circuit_state = CircuitState.HALF_OPEN
                return True
            return False
        return True

    def _calculate_delay(self, attempt):
        if self.retry_config.strategy == 'fixed':
            return self.retry_config.delay
        elif self.retry_config.strategy == 'linear':
            return self.retry_config.delay * attempt
        else:  # exponential
            return min(self.retry_config.delay * (2 ** attempt),
                      self.retry_config.max_delay)

    def call(self, request):
        if not self._check_circuit():
            raise CircuitOpenError("Circuit breaker is open")

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                result = self._handle_request(request)
                self._record_success()
                return result
            except Exception as e:
                self._record_failure()
                if attempt < self.retry_config.max_retries:
                    time.sleep(self._calculate_delay(attempt) / 1000)
                else:
                    raise
```

#### Go Service Mesh

```go
type CircuitState int

const (
    CircuitClosed CircuitState = iota
    CircuitOpen
    CircuitHalfOpen
)

type GatewayService struct {
    name            string
    backends        []Backend
    lbStrategy      string
    cbThreshold     int
    cbTimeout       time.Duration
    retryMax        int
    retryStrategy   string
    retryDelay      time.Duration
    circuitState    CircuitState
    failureCount    int32
    lastFailureTime time.Time
    rrIndex         uint32
}

func (s *GatewayService) selectBackend() *Backend {
    if len(s.backends) == 0 {
        return nil
    }
    switch s.lbStrategy {
    case "round_robin":
        idx := atomic.AddUint32(&s.rrIndex, 1) - 1
        return &s.backends[idx%uint32(len(s.backends))]
    case "random":
        return &s.backends[rand.Intn(len(s.backends))]
    default:
        return &s.backends[0]
    }
}

func (s *GatewayService) checkCircuit() bool {
    if s.circuitState == CircuitOpen {
        if time.Since(s.lastFailureTime) > s.cbTimeout {
            s.circuitState = CircuitHalfOpen
            return true
        }
        return false
    }
    return true
}
```

#### Rust Service Mesh

```rust
#[derive(Clone, Copy, PartialEq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

pub struct GatewayService {
    name: String,
    backends: Vec<Backend>,
    lb_strategy: String,
    cb_threshold: i32,
    cb_timeout: Duration,
    retry_max: i32,
    retry_strategy: String,
    circuit_state: RwLock<CircuitState>,
    failure_count: AtomicI32,
    last_failure_time: RwLock<Option<Instant>>,
    rr_index: AtomicU32,
}

impl GatewayService {
    fn select_backend(&self) -> Option<&Backend> {
        if self.backends.is_empty() {
            return None;
        }
        match self.lb_strategy.as_str() {
            "round_robin" => {
                let idx = self.rr_index.fetch_add(1, Ordering::SeqCst);
                Some(&self.backends[idx as usize % self.backends.len()])
            }
            "random" => {
                let mut rng = rand::thread_rng();
                Some(&self.backends[rng.gen_range(0..self.backends.len())])
            }
            _ => Some(&self.backends[0]),
        }
    }

    fn check_circuit(&self) -> bool {
        let state = *self.circuit_state.read().unwrap();
        if state == CircuitState::Open {
            if let Some(last) = *self.last_failure_time.read().unwrap() {
                if last.elapsed() > self.cb_timeout {
                    *self.circuit_state.write().unwrap() = CircuitState::HalfOpen;
                    return true;
                }
            }
            return false;
        }
        true
    }
}
```

### Helper Predicates

```prolog
% Extract load balance strategy
get_load_balance_strategy(service(_, [load_balance(round_robin)], _), S).
% S = round_robin

% Extract circuit breaker config
get_circuit_breaker_config(service(_, [circuit_breaker(threshold(5), timeout(30000))], _), C).
% C = config(5, 30000)

% Extract retry config
get_retry_config(service(_, [retry(3, exponential)], _), R).
% R = config(3, exponential, 100, 30000, false)

% Check if service has service mesh features
is_service_mesh_service(service(test, [load_balance(round_robin)], [])).
% true

has_load_balancing(service(test, [load_balance(random)], [])).
% true

has_circuit_breaker(service(test, [circuit_breaker(threshold(5), timeout(30000))], [])).
% true

has_retry(service(test, [retry(3, exponential)], [])).
% true
```

### Phase 4 Tests

All 61 tests pass:

- **Validation Tests (20)**: Load balance strategies, circuit breaker options, retry options, discovery methods, backends validation
- **Helper Predicate Tests (10)**: Strategy extraction, config extraction, feature detection
- **Python Compilation Tests (7)**: Service mesh generation with all features
- **Go Compilation Tests (8)**: Service mesh with atomic operations
- **Rust Compilation Tests (8)**: Service mesh with RwLock and lazy_static
- **Cross-Target Consistency Tests (3)**: CircuitState enum, backends, select_backend across all targets
- **Edge Case Tests (5)**: Single feature services, default configs

## Phase 5: Multi-Language Polyglot Services (Current)

Phase 5 implements seamless calls between services written in different languages.

### Polyglot Service Definition

```prolog
% Service that calls services in other languages
service(coordinator, [
    polyglot(true),
    target_language(python),
    depends_on([
        dep(ml_model, python, tcp('localhost', 8001)),
        dep(validator, go, tcp('localhost', 8002)),
        dep(database, rust, tcp('localhost', 8003))
    ])
], [
    receive(Job),
    call_service(ml_model, Job, Prediction),
    call_service(validator, Prediction, Validated),
    call_service(database, Validated, Stored),
    respond(Stored)
]).
```

### Polyglot Options

| Option | Description |
|--------|-------------|
| `polyglot(true)` | Enable polyglot service mode |
| `target_language(Lang)` | Target language (python, go, rust, javascript, csharp) |
| `depends_on([...])` | List of service dependencies |
| `endpoint(Endpoint)` | Service endpoint URL |

### Dependency Specification

```prolog
dep(ServiceName, Language, Transport)
```

- `ServiceName`: Name of the service to call
- `Language`: Implementation language (python, go, rust, etc.)
- `Transport`: How to connect (tcp(Host, Port), http(Path), unix_socket(Path))

### Generated Code Features

**Python**: `ServiceClient` class with `requests.post`, `ServiceRegistry` with remote/local lookup
**Go**: `ServiceClient` struct with `net/http`, `ServiceRegistry` with `sync.RWMutex`
**Rust**: `ServiceClient` with `reqwest`, `ServiceRegistry` with `RwLock<HashMap<>>`

### Phase 5 Tests

```bash
./tests/integration/test_polyglot_services.sh
# 22 tests: validation, Python/Go/Rust compilation
```

## Phase 6: Distributed Services (Current)

Phase 6 implements cluster-aware services with sharding, replication, and routing.

### Distributed Service Definition

```prolog
service(user_store, [
    distributed(true),
    sharding(consistent_hash),
    partition_key(user_id),
    replication(3),
    consistency(quorum),
    cluster([
        node(node1, 'localhost', 8001),
        node(node2, 'localhost', 8002),
        node(node3, 'localhost', 8003)
    ])
], [
    receive(Request),
    handle_request(Request, Response),
    respond(Response)
]).
```

### Distributed Service Options

| Option | Description |
|--------|-------------|
| `distributed(true)` | Enable distributed mode |
| `sharding(Strategy)` | Sharding strategy |
| `partition_key(Key)` | Field to use for partitioning |
| `replication(N)` | Number of replicas |
| `consistency(Level)` | Consistency level |
| `cluster([...])` | List of cluster nodes |

### Sharding Strategies

| Strategy | Description |
|----------|-------------|
| `hash` | Hash-based partitioning (default) |
| `range` | Range-based partitioning |
| `consistent_hash` | Consistent hashing with virtual nodes |
| `geographic` | Geographic/region-based partitioning |

### Consistency Levels

| Level | Description |
|-------|-------------|
| `eventual` | Eventually consistent (default) |
| `strong` | Strong consistency (all replicas) |
| `quorum` | Quorum-based ((N/2)+1 nodes) |
| `read_your_writes` | Read-your-writes consistency |
| `causal` | Causal consistency |

### Generated Code Features

**All targets generate:**
- `ConsistentHashRing`: Virtual node ring for consistent hashing
- `ShardRouter`: Routes requests to appropriate shards
- `ReplicationManager`: Manages write/read quorums
- `ClusterNode`: Node representation with health status

**Thread Safety:**
- Python: `threading.Lock`, `RwLock`
- Go: `sync.RWMutex`, `atomic`
- Rust: `RwLock`, `AtomicU64`

### Phase 6 Tests

```bash
./tests/integration/test_distributed_services.sh
# 24 tests: validation, Python/Go/Rust compilation, cross-target consistency
```

## Phase 7: Service Discovery

Phase 7 adds automatic service discovery with health checks and multiple backend support.

### Service Discovery Options

| Option | Description | Default |
|--------|-------------|---------|
| `discovery_enabled(Bool)` | Enable service discovery | false |
| `discovery_backend(Backend)` | Discovery backend | consul |
| `discovery_ttl(Seconds)` | Service TTL for heartbeat | 60 |
| `discovery_tags(List)` | Tags for filtering | [] |
| `health_check(Config)` | Health check configuration | http('/health', 30000) |

### Supported Discovery Backends

- `consul` - HashiCorp Consul
- `etcd` - CoreOS etcd
- `dns` - DNS-based discovery
- `kubernetes` - Kubernetes Service Discovery
- `zookeeper` - Apache ZooKeeper
- `eureka` - Netflix Eureka

### Health Check Types

```prolog
% HTTP health check
health_check(http('/health', IntervalMs))

% TCP health check
health_check(tcp(Port, IntervalMs))
```

### Example: Discoverable Service

```prolog
service(api_gateway, [
    discovery_enabled(true),
    discovery_backend(consul),
    health_check(http('/health', 30000)),
    discovery_tags([production, v2])
], [
    receive(Request),
    route_by(path, Routes),
    respond(Response)
]).
```

### Generated Code Components

**Service Registry Interface:**
- `ServiceRegistry`: Abstract registry interface
- `ConsulRegistry`: Consul-based implementation
- `LocalRegistry`: In-memory implementation for testing

**Health Checking:**
- `HealthChecker`: Performs HTTP/TCP health checks
- `HealthStatus`: Healthy/Unhealthy/Unknown
- `ServiceInstance`: Service instance with metadata

**Heartbeat Mechanism:**
- Automatic TTL-based heartbeat
- Self-healing registration
- Graceful deregistration on shutdown

### Phase 7 Tests

```bash
./tests/integration/test_service_discovery.sh
# 20 tests: validation, Python/Go/Rust discovery compilation
```

## Phase 8: Service Tracing

Phase 8 adds OpenTelemetry-compatible distributed tracing.

### Tracing Options

| Option | Description | Default |
|--------|-------------|---------|
| `tracing(Bool)` | Enable distributed tracing | false |
| `trace_exporter(Exporter)` | Trace exporter backend | otlp |
| `trace_sampling(Rate)` | Sampling rate (0.0-1.0) | 1.0 |
| `trace_service_name(Name)` | Service name in traces | service name |
| `trace_propagation(Format)` | Context propagation format | w3c |
| `trace_attributes(List)` | Default span attributes | [] |

### Supported Exporters

- `otlp` / `otlp(Endpoint)` - OpenTelemetry Protocol (default)
- `jaeger` / `jaeger(Endpoint)` - Jaeger
- `zipkin` / `zipkin(Endpoint)` - Zipkin
- `datadog` / `datadog(AgentHost)` - Datadog APM
- `console` - Console output (for debugging)
- `none` - Disabled

### Propagation Formats

- `w3c` - W3C Trace Context (default)
- `b3` - B3 Single Header
- `b3_multi` - B3 Multi Header
- `jaeger` - Jaeger native format
- `xray` - AWS X-Ray format
- `datadog` - Datadog format

### Example: Traced Service

```prolog
service(payment_processor, [
    tracing(true),
    trace_exporter(otlp('http://collector:4318')),
    trace_sampling(0.1),
    trace_propagation(w3c),
    trace_attributes([environment-production, team-payments])
], [
    receive(PaymentRequest),
    validate_payment(PaymentRequest, Validated),
    process_payment(Validated, Result),
    respond(Result)
]).
```

### Generated Code Components

**Span Context:**
- `SpanContext`: Trace ID, Span ID, Trace Flags
- W3C traceparent header generation/parsing
- B3 header support

**Span Management:**
- `Span`: Individual trace span with attributes
- `SpanKind`: Server, Client, Producer, Consumer, Internal
- `SpanStatus`: Unset, OK, Error
- `SpanEvent`: Span events with timestamps

**Tracer:**
- `Tracer`: Central tracing manager
- Sampling decision logic
- Context extraction/injection for propagation
- Batch export with flush

**Exporters:**
- `SpanExporter`: Abstract exporter interface
- `OTLPSpanExporter`: OTLP HTTP export
- `JaegerSpanExporter`: Jaeger HTTP export
- `ZipkinSpanExporter`: Zipkin HTTP export
- `ConsoleSpanExporter`: Console output

### Phase 8 Tests

```bash
./tests/integration/test_service_tracing.sh
# 20 tests: validation, Python/Go/Rust tracing compilation
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

Integration tests verify the implementation for each phase:

```bash
# Phase 1: In-process services
./tests/integration/test_in_process_services.sh

# Phase 2: Unix socket services
./tests/integration/test_unix_socket_services.sh

# Phase 3: Network services (TCP/HTTP)
./tests/integration/test_network_services.sh
```

### Phase 3 Tests Cover:

1. is_network_service identifies TCP services
2. is_network_service identifies HTTP services
3. TCP service is network, not cross-process
4. Python TCP service compilation
5. Python TCP service has JSONL protocol
6. Python HTTP service compilation
7. Python HTTP service handles REST methods
8. Go TCP service compilation
9. Go TCP service has JSONL protocol
10. Go HTTP service compilation
11. Go HTTP service handles REST methods
12. Rust TCP service compilation
13. Rust TCP service has JSONL protocol
14. Rust HTTP service compilation
15. Rust HTTP service handles REST methods
16. Python TCP client compilation
17. Python HTTP client compilation
18. Go TCP client compilation
19. Go HTTP client compilation
20. Rust TCP client compilation
21. Rust HTTP client compilation
22. Stateful TCP service (Python)
23. Stateful HTTP service (Go)
24. Service dispatch through compile_service_to_python (TCP)
25. Service dispatch through compile_service_to_go (HTTP)
26. In-process service still works (regression test)

## Files

| File | Description |
|------|-------------|
| `src/unifyweaver/core/service_validation.pl` | Service definition validation (Phases 1-8) |
| `src/unifyweaver/core/pipeline_validation.pl` | Extended with call_service validation |
| `src/unifyweaver/targets/python_target.pl` | Python service compilation (all phases) |
| `src/unifyweaver/targets/go_target.pl` | Go service compilation (all phases) |
| `src/unifyweaver/targets/rust_target.pl` | Rust service compilation (all phases) |
| `tests/integration/test_in_process_services.sh` | Phase 1 integration tests |
| `tests/integration/test_unix_socket_services.sh` | Phase 2 integration tests |
| `tests/integration/test_network_services.sh` | Phase 3 integration tests |
| `tests/integration/test_service_mesh.sh` | Phase 4 integration tests |
| `tests/integration/test_polyglot_services.sh` | Phase 5 integration tests |
| `tests/integration/test_distributed_services.sh` | Phase 6 integration tests |
| `tests/integration/test_service_discovery.sh` | Phase 7 integration tests |
| `tests/integration/test_service_tracing.sh` | Phase 8 integration tests |
| `docs/CLIENT_SERVER_DESIGN.md` | This document |
