# Cross-Target Glue User Guide

UnifyWeaver's cross-target glue system enables seamless communication between different programming languages and runtimes. This guide covers practical usage patterns.

## Quick Start

### Shell Pipeline (AWK + Python)

```prolog
:- use_module('src/unifyweaver/glue/shell_glue').

% Generate a pipeline: AWK filter → Python transform → AWK summarize
generate_my_pipeline :-
    generate_pipeline(
        [
            step(filter, awk, 'filter.awk', []),
            step(transform, python, 'transform.py', []),
            step(summarize, awk, 'summarize.awk', [])
        ],
        [],
        Script
    ),
    open('run_pipeline.sh', write, S),
    write(S, Script),
    close(S).
```

### .NET Pipeline (C# + PowerShell + Python)

```prolog
:- use_module('src/unifyweaver/glue/dotnet_glue').

% Generate bridges for .NET ecosystem
generate_dotnet_bridges :-
    generate_powershell_bridge([], PSBridge),
    generate_ironpython_bridge([], PyBridge),
    % Write to files...
```

### Native Pipeline (Go + Rust)

```prolog
:- use_module('src/unifyweaver/glue/native_glue').

% Generate Go with parallel processing
generate_go_processor :-
    generate_go_pipe_main(
        'return fields',  % Your logic here
        [parallel(8)],    % 8 worker goroutines
        Code
    ).
```

### Distributed Pipeline (HTTP Services)

```prolog
:- use_module('src/unifyweaver/glue/network_glue').

% Register services
:- register_service(ml_api, 'http://ml:8080', [timeout(60)]).

% Generate HTTP server
generate_server :-
    generate_go_http_server(
        [endpoint('/predict', predict, [])],
        [port(8080), cors(true)],
        Code
    ).
```

---

## Module Reference

### Shell Glue (`shell_glue.pl`)

Generate complete scripts for shell-based targets.

#### Script Generation

```prolog
% AWK script with TSV I/O
generate_awk_script(Logic, Fields, Options, Script).

% Python script with format handling
generate_python_script(Logic, Fields, Options, Script).

% Bash script with field parsing
generate_bash_script(Logic, Fields, Options, Script).

% Pipeline orchestrator
generate_pipeline(Steps, Options, Script).
```

#### Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `format(F)` | tsv, csv, json | tsv | I/O format |
| `header(B)` | true, false | false | Skip header line |
| `input_format(F)` | tsv, csv, json | tsv | Input format |
| `output_format(F)` | tsv, csv, json | tsv | Output format |

#### Example: Log Analysis

```prolog
generate_awk_script(
    '
    if (status >= 400) {
        print ip, timestamp, status
    }
    ',
    [ip, timestamp, method, path, status],
    [format(tsv)],
    Script
).
```

---

### .NET Glue (`dotnet_glue.pl`)

Generate bridges for .NET ecosystem communication.

#### Runtime Detection

```prolog
detect_dotnet_runtime(Runtime).  % dotnet_modern | dotnet_core | mono | none
detect_ironpython(Available).    % true | false
detect_powershell(Version).      % core(V) | windows(V) | none
```

#### Bridge Generation

```prolog
% PowerShell hosting in C#
generate_powershell_bridge(Options, Code).

% IronPython hosting in C#
generate_ironpython_bridge(Options, Code).

% CPython via subprocess (fallback)
generate_cpython_bridge(Options, Code).
```

#### Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `namespace(N)` | atom | 'UnifyWeaver.Glue' | C# namespace |
| `class(C)` | atom | varies | Class name |
| `async(B)` | true, false | false | Generate async methods |
| `python_path(P)` | path | 'python3' | CPython executable |

#### IronPython Compatibility

Check if modules are compatible:

```prolog
?- ironpython_compatible(json).
true.

?- ironpython_compatible(numpy).
false.

?- can_use_ironpython([sys, json, re]).
true.

?- python_runtime_choice([numpy, json], Runtime).
Runtime = cpython_pipe.
```

Compatible modules include: sys, os, json, re, collections, csv, xml, datetime, math, itertools, functools, clr (IronPython special), and 30+ more.

---

### Native Glue (`native_glue.pl`)

Generate and manage Go/Rust binaries.

#### Toolchain Detection

```prolog
detect_go(Version).
detect_rust(Version).
detect_cargo(Version).
```

#### Code Generation

```prolog
% Go main with TSV I/O
generate_go_pipe_main(Logic, Options, Code).

% Rust main with TSV I/O
generate_rust_pipe_main(Logic, Options, Code).

% Build scripts
generate_go_build_script(SourcePath, Options, Script).
generate_rust_build_script(SourcePath, Options, Script).
```

#### Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `format(F)` | tsv, json | tsv | I/O format |
| `parallel(N)` | integer | 1 | Worker count (Go) |
| `fields(L)` | list | [] | Field names (JSON mode) |
| `optimize(B)` | true, false | true | Build optimization |

#### Go Parallel Processing

```prolog
generate_go_pipe_main(
    '
    // Process fields
    result := transform(fields)
    return result
    ',
    [parallel(8)],  % 8 goroutines
    Code
).
```

Generated code uses channels and WaitGroup for safe parallel processing.

#### Cross-Compilation

```prolog
cross_compile_targets(Targets).
% Returns: linux-amd64, linux-arm64, darwin-amd64, darwin-arm64, windows-amd64

generate_cross_compile(go, 'main.go', [linux-amd64, darwin-arm64], Script).
```

---

### Network Glue (`network_glue.pl`)

Generate HTTP servers, clients, and socket communication.

#### Service Registry

```prolog
register_service(Name, URL, Options).
service(Name, URL).
service_options(Name, Options).
unregister_service(Name).
endpoint_url(Service, Endpoint, URL).
```

#### HTTP Server Generation

```prolog
% Go server (net/http)
generate_go_http_server(Endpoints, Options, Code).

% Python server (Flask)
generate_python_http_server(Endpoints, Options, Code).

% Rust server (Actix-web)
generate_rust_http_server(Endpoints, Options, Code).
```

#### Endpoint Definition

```prolog
endpoint(Path, Handler, EndpointOptions)

% Example
[
    endpoint('/api/process', process_handler, []),
    endpoint('/api/batch', batch_handler, [methods(['POST'])]),
    endpoint('/health', health_check, [methods(['GET'])])
]
```

#### HTTP Client Generation

```prolog
% Go client
generate_go_http_client(Services, Options, Code).

% Python client (requests)
generate_python_http_client(Services, Options, Code).

% Bash client (curl + jq)
generate_bash_http_client(Services, Options, Code).
```

#### Service Definition

```prolog
service_def(Name, BaseURL, Endpoints)

% Example
service_def(ml_api, 'http://ml:8080', ['/predict', '/classify'])
```

#### Socket Communication

```prolog
% TCP server
generate_socket_server(Target, Port, Options, Code).

% TCP client
generate_socket_client(Target, Host, Options, Code).
```

#### Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `port(P)` | integer | 8080 | Listen port |
| `cors(B)` | true, false | true | Enable CORS |
| `timeout(S)` | integer | 30 | Request timeout |
| `buffer_size(N)` | integer | 65536 | Socket buffer |

---

## Common Patterns

### Pattern 1: ETL Pipeline

```prolog
% Extract (AWK) → Transform (Python) → Load (Go)
generate_etl_pipeline :-
    generate_pipeline(
        [
            step(extract, awk, 'extract.awk', []),
            step(transform, python, 'transform.py', []),
            step(load, go, './loader', [])
        ],
        [input('data.csv'), output('result.tsv')],
        Script
    ).
```

### Pattern 2: Microservices

```prolog
% Generate coordinated services
generate_microservices :-
    % API Gateway
    generate_go_http_server(
        [endpoint('/api/v1', route, [])],
        [port(8080)],
        Gateway
    ),

    % Worker service
    generate_python_http_server(
        [endpoint('/process', process, [])],
        [port(8081)],
        Worker
    ).
```

### Pattern 3: Hybrid .NET + Python

```prolog
% Use IronPython when possible, CPython for ML
generate_hybrid_pipeline :-
    % Check compatibility
    Imports = [numpy, pandas],
    python_runtime_choice(Imports, Runtime),
    % Runtime = cpython_pipe (fallback)

    (Runtime == ironpython ->
        generate_ironpython_bridge([], Bridge)
    ;
        generate_cpython_bridge([], Bridge)
    ).
```

### Pattern 4: High-Performance Data Processing

```prolog
% Go parallel → Rust aggregation
generate_data_pipeline :-
    generate_go_pipe_main(
        'return transform(fields)',
        [parallel(8), format(json)],
        GoCode
    ),

    generate_rust_pipe_main(
        'aggregate(fields)',
        [format(json)],
        RustCode
    ),

    generate_native_pipeline(
        [
            step(transform, go, './transform', []),
            step(aggregate, rust, './aggregate', [])
        ],
        [],
        Pipeline
    ).
```

---

## Examples

Complete examples are available in:

| Directory | Description |
|-----------|-------------|
| `examples/cross-target-glue/` | AWK ↔ Python log analysis |
| `examples/dotnet-glue/` | C# ↔ PowerShell ↔ Python |
| `examples/native-glue/` | Go ↔ Rust high-performance |
| `examples/network-glue/` | Distributed microservices |

Run any example:
```bash
cd examples/cross-target-glue
swipl log_pipeline.pl
./run_pipeline.sh < sample_access.log
```

---

## Troubleshooting

### Runtime Not Detected

```prolog
?- detect_go(Version).
Version = none.
```

Ensure the runtime is installed and in PATH.

### IronPython Fallback

If you see `cpython_pipe` when expecting `ironpython`, check your imports:

```prolog
?- can_use_ironpython([numpy]).
false.
```

numpy, pandas, scipy require CPython due to C extensions.

### Pipeline Fails

Check each step individually:
```bash
cat input.tsv | awk -f step1.awk  # Test step 1
cat input.tsv | awk -f step1.awk | python3 step2.py  # Test steps 1+2
```

### HTTP Service Not Responding

1. Check service is running: `curl http://localhost:8080/health`
2. Check CORS if browser-based: Enable `cors(true)` option
3. Check timeout: Increase with `timeout(60)` option

---

## Deployment Glue (`deployment_glue.pl`)

Production-ready deployment, error handling, and monitoring for remote services.

### Service Declaration

```prolog
:- use_module('src/unifyweaver/glue/deployment_glue').

% Declare a remote service
:- declare_service(ml_predictor, [
    host('ml.example.com'),
    port(8080),
    target(python),
    entry_point('server.py'),
    transport(https)
]).

% Configure SSH deployment
:- declare_deploy_method(ml_predictor, ssh, [
    user('deploy'),
    remote_dir('/opt/services')
]).

% Track source files for change detection
:- declare_service_sources(ml_predictor, [
    'src/**/*.py',
    'requirements.txt'
]).
```

### Deployment Operations

```prolog
% Generate deployment script
generate_deploy_script(ml_predictor, [], Script).

% Deploy service (checks for changes first)
deploy_service(ml_predictor, Result).

% Lifecycle operations
start_service(ml_predictor, Result).
stop_service(ml_predictor, Result).
restart_service(ml_predictor, Result).

% Check service status
service_status(ml_predictor, Status).
```

### Multi-Host Deployment

```prolog
% Configure multiple hosts
:- declare_service_hosts(api_service, [
    host_config('api1.example.com', [user('deploy')]),
    host_config('api2.example.com', [user('deploy')]),
    host_config('api3.example.com', [user('deploy')])
]).

% Deploy to all hosts
deploy_to_all_hosts(api_service, Results).
```

### Rollback Support

```prolog
% Deploy with automatic rollback on health check failure
deploy_with_rollback(ml_service, Result).

% Manual rollback
rollback_service(ml_service, Result).
```

### Graceful Shutdown

```prolog
graceful_stop(api_service, [
    drain_timeout(30),   % Wait 30s for connections
    force_after(60)      % Force kill after 60s
], Result).
```

---

## Error Handling

### Retry Policies

```prolog
:- declare_retry_policy(ml_service, [
    max_retries(5),
    initial_delay(1000),       % 1 second
    max_delay(30000),          % Max 30 seconds
    backoff(exponential),      % exponential | linear | fixed
    multiplier(2),
    retry_on([timeout, connection_refused]),
    fail_on([400, 401, 403])
]).

% Execute with retry
call_with_retry(ml_service, predict, [Input], Result).
```

### Fallback Mechanisms

```prolog
% Use backup service
:- declare_fallback(primary_service, backup_service(secondary_service)).

% Return default value
:- declare_fallback(ml_service, default_value(fallback_prediction)).

% Use cache
:- declare_fallback(data_service, cache([key(user_data), ttl(3600)])).

% Custom fallback predicate
:- declare_fallback(api_service, custom(my_fallback_handler)).

% Execute with fallback
call_with_fallback(ml_service, predict, [Input], Result).
```

### Circuit Breaker

```prolog
:- declare_circuit_breaker(ml_service, [
    failure_threshold(5),      % Open after 5 failures
    success_threshold(3),      % Close after 3 successes
    half_open_timeout(30000)   % Try half-open after 30s
]).

% Execute with circuit protection
call_with_circuit_breaker(ml_service, predict, [Input], Result).

% Query state: closed | open | half_open
circuit_state(ml_service, State).

% Reset circuit
reset_circuit_breaker(ml_service).
```

### Timeout Configuration

```prolog
:- declare_timeouts(ml_service, [
    connect_timeout(5000),     % 5s connection timeout
    read_timeout(30000),       % 30s response timeout
    total_timeout(60000)       % 60s total timeout
]).

call_with_timeout(ml_service, predict, [Input], Result).
```

### Combined Protection

```prolog
% Apply all error handling strategies
protected_call(ml_service, predict, [Input], Result).
% Order: circuit breaker → timeout → retry → fallback
```

---

## Monitoring

### Health Check Monitoring

```prolog
:- declare_health_check(api_service, [
    endpoint('/health'),
    interval(30),              % Check every 30s
    timeout(5),
    unhealthy_threshold(3),    % Unhealthy after 3 failures
    healthy_threshold(2)       % Healthy after 2 successes
]).

% Query health status
health_status(api_service, Status).  % healthy | unhealthy | unknown

% Start monitoring
start_health_monitor(api_service, Result).
```

### Metrics Collection

```prolog
:- declare_metrics(api_service, [
    collect([request_count, latency, error_count]),
    labels([service-api_service, env-production]),
    export(prometheus),
    retention(3600)            % Keep for 1 hour
]).

% Record metrics
record_metric(api_service, request_count, 1).
record_metric(api_service, latency, 150).

% Get metrics
get_metrics(api_service, Metrics).

% Export as Prometheus format
generate_prometheus_metrics(api_service, Output).
```

### Structured Logging

```prolog
:- declare_logging(api_service, [
    level(info),               % debug | info | warn | error
    format(json),              % json | text
    output(stdout),            % stdout | file(Path)
    max_entries(1000)
]).

% Log events
log_event(api_service, info, 'Request received', [method-'GET', path-'/api']).
log_event(api_service, error, 'Database failed', [retry-3]).

% Query logs
get_log_entries(api_service, [level(warn), limit(100)], Entries).
```

**JSON Output:**
```json
{"timestamp":"2025-01-15T10:30:00Z","service":"api_service","level":"info","message":"Request received","method":"GET","path":"/api"}
```

### Alerting

```prolog
:- declare_alert(api_service, high_error_rate, [
    condition('error_rate > 0.05'),
    severity(critical),        % critical | warning | info
    cooldown(300),             % 5 min between alerts
    notify([
        slack('#alerts'),
        email('oncall@example.com'),
        pagerduty
    ])
]).

% Trigger alert
trigger_alert(api_service, high_error_rate, [rate-0.1]).

% Check triggered alerts
check_alerts(api_service, TriggeredAlerts).

% Get alert history
alert_history(api_service, [limit(100)], History).
```

---

## Complete Production Example

```prolog
:- use_module('src/unifyweaver/glue/deployment_glue').

%% Service Configuration
:- declare_service(ml_predictor, [
    host('ml.example.com'),
    port(8080),
    target(python),
    entry_point('server.py'),
    transport(https)
]).

:- declare_deploy_method(ml_predictor, ssh, [
    user('deploy'),
    remote_dir('/opt/services/ml')
]).

:- declare_service_sources(ml_predictor, [
    'src/ml/**/*.py',
    'models/*.pkl',
    'requirements.txt'
]).

%% Error Handling
:- declare_retry_policy(ml_predictor, [
    max_retries(3),
    initial_delay(1000),
    backoff(exponential)
]).

:- declare_fallback(ml_predictor, default_value({status: unavailable})).

:- declare_circuit_breaker(ml_predictor, [
    failure_threshold(5),
    half_open_timeout(30000)
]).

:- declare_timeouts(ml_predictor, [
    total_timeout(60000)
]).

%% Monitoring
:- declare_health_check(ml_predictor, [
    endpoint('/health'),
    interval(30)
]).

:- declare_metrics(ml_predictor, [
    collect([requests, latency, errors]),
    export(prometheus)
]).

:- declare_logging(ml_predictor, [
    level(info),
    format(json)
]).

:- declare_alert(ml_predictor, service_unhealthy, [
    severity(critical),
    notify([pagerduty])
]).

%% Usage
call_service :-
    protected_call(ml_predictor, predict, [Input], Result),
    handle_result(Result).
```
