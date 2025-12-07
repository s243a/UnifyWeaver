# Cross-Target Glue: Implementation Plan

## Phase Overview

```
Phase 1: Foundation        → Target registry, basic pipe glue        ✅ COMPLETE
Phase 2: Shell Integration → Bash ↔ AWK ↔ Python pipes               ✅ COMPLETE
Phase 3: .NET Integration  → C# ↔ PowerShell ↔ IronPython in-process ✅ COMPLETE
Phase 4: Native Targets    → Go/Rust binary orchestration            ✅ COMPLETE
Phase 5: Network Layer     → Remote target communication             ✅ COMPLETE
Phase 6: Production Ready  → Deployment, error handling, monitoring  ✅ COMPLETE
Phase 7: Cloud & Enterprise → Containers, secrets, multi-region      (Future)
```

---

## Phase 1: Foundation ✅ COMPLETE

**Goal:** Establish core infrastructure for target management and basic communication.

**Implemented in:**
- `src/unifyweaver/core/target_registry.pl`
- `src/unifyweaver/core/target_mapping.pl`
- `src/unifyweaver/glue/pipe_glue.pl`

### Key Features

**Target Registry:**
```prolog
:- module(target_registry, [
    register_target/3,        % register_target(Name, Family, Capabilities)
    target_family/2,          % target_family(Target, Family)
    targets_same_family/2,    % targets_same_family(T1, T2)
    target_capabilities/2,    % target_capabilities(Target, Caps)
    list_targets/1            % list_targets(Targets)
]).
```

15+ built-in targets across 6 runtime families:
- Shell: bash, awk, sed, perl
- Python: python, ironpython
- .NET: csharp, powershell, fsharp
- JVM: java, scala, clojure
- Native: go, rust, c, cpp
- Database: sql, sqlite, postgresql

**Predicate-Target Mapping:**
```prolog
:- declare_target(filter/2, awk).
:- declare_target(analyze/2, python, [location(local_process)]).
```

### Deliverables ✅
- [x] `target_registry.pl` - Target metadata management
- [x] `target_mapping.pl` - Predicate-to-target declarations
- [x] `pipe_glue.pl` - Basic TSV pipe code generation
- [x] Unit tests for registry and mapping (40+ assertions)

---

## Phase 2: Shell Integration ✅ COMPLETE

**Goal:** Enable seamless Bash ↔ AWK ↔ Python pipelines.

**Implemented in:**
- `src/unifyweaver/glue/shell_glue.pl`

### Key Features

**Complete Script Generation:**
```prolog
generate_awk_script(Logic, Fields, Options, Script).
generate_python_script(Logic, Fields, Options, Script).
generate_bash_script(Logic, Fields, Options, Script).
generate_pipeline(Steps, Options, Script).
```

**Format Support:**
- TSV (default)
- CSV
- JSON

**Features:**
- Automatic field assignment from input
- Header skip support
- Format-aware output generation

### Example Output

```bash
#!/bin/bash
# Generated pipeline: filter → analyze → summarize

cat input.tsv \
    | awk -f "filter.awk" \
    | python3 "analyze.py" \
    | awk -f "summarize.awk"
```

### Deliverables ✅
- [x] AWK glue templates (reader/writer)
- [x] Python glue templates (reader/writer)
- [x] Bash glue templates (reader/writer)
- [x] Shell pipeline orchestrator
- [x] Integration tests (30+ assertions)
- [x] Example: Log analysis pipeline (`examples/cross-target-glue/`)

---

## Phase 3: .NET Integration ✅ COMPLETE

**Goal:** Enable in-process communication between C#, PowerShell, and IronPython.

**Implemented in:**
- `src/unifyweaver/glue/dotnet_glue.pl`

### Key Features

**Runtime Detection:**
```prolog
detect_dotnet_runtime(Runtime).  % dotnet_modern | dotnet_core | mono | none
detect_ironpython(Available).    % true | false
detect_powershell(Version).      % core(V) | windows(V) | none
```

**Bridge Generation:**
```prolog
generate_powershell_bridge(Options, Code).   % C# hosting PowerShell
generate_ironpython_bridge(Options, Code).   % C# hosting IronPython
generate_cpython_bridge(Options, Code).      % C# → CPython via pipes
```

**IronPython Compatibility:**
- 40+ compatible modules (sys, os, json, re, collections, clr, etc.)
- Automatic fallback to CPython for numpy/pandas/tensorflow

### Generated C# Bridges

```csharp
// PowerShell in-process
PowerShellBridge.Invoke<TInput, TOutput>(script, input)
PowerShellBridge.InvokeStream<TInput, TOutput>(script, stream)

// IronPython in-process
IronPythonBridge.Execute(pythonCode)
IronPythonBridge.ExecuteWithInput<TInput>(script, input)

// CPython via pipes (fallback)
CPythonBridge.Execute<TInput, TOutput>(script, input)
```

### Deliverables ✅
- [x] .NET runtime detection
- [x] C# ↔ PowerShell in-process bridge
- [x] C# ↔ IronPython in-process bridge
- [x] IronPython compatibility checker (40+ modules)
- [x] CPython fallback via pipes
- [x] Integration tests (72 assertions)
- [x] Example: .NET data processing pipeline (`examples/dotnet-glue/`)

---

## Phase 4: Native Targets ✅ COMPLETE

**Goal:** Orchestrate Go and Rust compiled binaries.

**Implemented in:**
- `src/unifyweaver/glue/native_glue.pl`

### Key Features

**Binary Management:**
```prolog
register_binary(Pred/Arity, Target, Path, Options).
compiled_binary(Pred/Arity, Target, Path).
compile_if_needed(Pred/Arity, Target, SourcePath, BinaryPath).
```

**Toolchain Detection:**
```prolog
detect_go(Version).
detect_rust(Version).
detect_cargo(Version).
```

**Code Generation:**
```prolog
generate_go_pipe_main(Logic, Options, Code).      % TSV/JSON, parallel workers
generate_rust_pipe_main(Logic, Options, Code).    % TSV/JSON, serde
generate_go_build_script(SourcePath, Options, Script).
generate_rust_build_script(SourcePath, Options, Script).
```

**Cross-Compilation:**
- Linux (amd64, arm64)
- macOS (amd64, arm64)
- Windows (amd64)

### Go Parallel Processing

```go
// Generated with parallel(8) option
func main() {
    lines := make(chan string, 10000)
    results := make(chan string, 10000)

    // 8 worker goroutines
    for i := 0; i < 8; i++ {
        go func() { /* process */ }()
    }
}
```

### Deliverables ✅
- [x] Binary compilation management
- [x] Go pipe-compatible wrapper generation
- [x] Rust pipe-compatible wrapper generation
- [x] Cross-compilation support (5 platforms)
- [x] Integration tests (62 assertions)
- [x] Example: High-performance pipeline (`examples/native-glue/`)

---

## Phase 5: Network Layer ✅ COMPLETE

**Goal:** Enable remote target communication via sockets/HTTP.

**Implemented in:**
- `src/unifyweaver/glue/network_glue.pl`

### Key Features

**Service Registry:**
```prolog
register_service(Name, URL, Options).
service(Name, URL).
endpoint_url(Service, Endpoint, URL).
```

**HTTP Server Generation:**
```prolog
generate_go_http_server(Endpoints, Options, Code).     % net/http + CORS
generate_python_http_server(Endpoints, Options, Code). % Flask
generate_rust_http_server(Endpoints, Options, Code).   % Actix-web
```

**HTTP Client Generation:**
```prolog
generate_go_http_client(Services, Options, Code).
generate_python_http_client(Services, Options, Code).  % requests
generate_bash_http_client(Services, Options, Code).    % curl + jq
```

**Socket Communication:**
```prolog
generate_socket_server(Target, Port, Options, Code).
generate_socket_client(Target, Host, Options, Code).
```

**Network Pipeline:**
```prolog
generate_network_pipeline(Steps, Options, Code).
% Steps can be local or remote
```

### Consistent API Format

All HTTP services use the same JSON schema:
```json
Request:  {"data": <any>}
Response: {"success": bool, "data": <any>, "error": <string?>}
```

### Deliverables ✅
- [x] HTTP server wrapper generation (Go, Python, Rust)
- [x] HTTP client wrapper generation (Go, Python, Bash)
- [x] Socket-based communication option
- [x] Service registry
- [x] Integration tests (90 assertions)
- [x] Example: Distributed microservices (`examples/network-glue/`)

---

## Phase 6: Production Ready ✅ COMPLETE

**Goal:** Production-ready deployment, error handling, and monitoring.

**Implemented in:**
- `src/unifyweaver/glue/deployment_glue.pl`

### Phase 6a: Deployment Foundation

**SSH Deployment:**
```prolog
:- declare_service(ml_predictor, [
    host('ml.example.com'),
    port(8080),
    target(python),
    entry_point('server.py'),
    transport(https)
]).

:- declare_deploy_method(ml_predictor, ssh, [
    user('deploy'),
    remote_dir('/opt/services')
]).

generate_deploy_script(ml_predictor, [], Script).
```

**Security by Default:**
- Remote services require encryption (HTTPS/SSH)
- HTTP only allowed for localhost
- `validate_security/2` checks configuration

**Change Detection:**
```prolog
:- declare_service_sources(ml_predictor, ['src/**/*.py', 'requirements.txt']).
check_for_changes(ml_predictor, Changes).
```

### Phase 6b: Advanced Deployment

**Multi-Host Deployment:**
```prolog
:- declare_service_hosts(api_service, [
    host_config('api1.example.com', [user('deploy')]),
    host_config('api2.example.com', [user('deploy')])
]).
deploy_to_all_hosts(api_service, Results).
```

**Rollback Support:**
```prolog
deploy_with_rollback(ml_service, Result).  % Auto-rollback on health check failure
rollback_service(ml_service, Result).       % Manual rollback
```

**Graceful Shutdown:**
```prolog
graceful_stop(api_service, [drain_timeout(30), force_after(60)], Result).
```

### Phase 6c: Error Handling

**Retry Policies:**
```prolog
:- declare_retry_policy(ml_service, [
    max_retries(5),
    initial_delay(1000),
    backoff(exponential),
    multiplier(2)
]).
call_with_retry(ml_service, predict, [Input], Result).
```

**Fallback Mechanisms:**
```prolog
:- declare_fallback(ml_service, default_value(fallback_prediction)).
:- declare_fallback(primary, backup_service(secondary)).
call_with_fallback(ml_service, predict, [Input], Result).
```

**Circuit Breaker:**
```prolog
:- declare_circuit_breaker(ml_service, [
    failure_threshold(5),
    success_threshold(3),
    half_open_timeout(30000)
]).
call_with_circuit_breaker(ml_service, predict, [Input], Result).
```

**Timeout Configuration:**
```prolog
:- declare_timeouts(ml_service, [
    connect_timeout(5000),
    read_timeout(30000),
    total_timeout(60000)
]).
```

**Combined Protection:**
```prolog
protected_call(ml_service, predict, [Input], Result).
% Applies: circuit breaker → timeout → retry → fallback
```

### Phase 6d: Monitoring

**Health Check Monitoring:**
```prolog
:- declare_health_check(api_service, [
    endpoint('/health'),
    interval(30),
    unhealthy_threshold(3),
    healthy_threshold(2)
]).
health_status(api_service, Status).  % healthy | unhealthy | unknown
```

**Metrics Collection (Prometheus):**
```prolog
:- declare_metrics(api_service, [
    collect([request_count, latency]),
    labels([service-api_service]),
    export(prometheus)
]).
record_metric(api_service, request_count, 1).
generate_prometheus_metrics(api_service, Output).
```

**Structured Logging:**
```prolog
:- declare_logging(api_service, [
    level(info),
    format(json),
    output(stdout)
]).
log_event(api_service, info, 'Request received', [method-'GET']).
% Output: {"timestamp":"...","service":"api_service","level":"info",...}
```

**Alerting:**
```prolog
:- declare_alert(api_service, high_error_rate, [
    severity(critical),
    cooldown(300),
    notify([slack('#alerts'), pagerduty])
]).
trigger_alert(api_service, high_error_rate, [rate-0.1]).
```

### Deliverables ✅
- [x] SSH deployment with agent forwarding
- [x] Service lifecycle management (start/stop/restart)
- [x] Change detection and automatic redeployment
- [x] Security validation (remote requires encryption)
- [x] Multi-host deployment
- [x] Rollback support with automatic rollback
- [x] Graceful shutdown with connection draining
- [x] Retry policies with exponential backoff
- [x] Fallback mechanisms
- [x] Circuit breaker pattern
- [x] Timeout configuration
- [x] Health check monitoring
- [x] Prometheus metrics export
- [x] Structured JSON logging
- [x] Alerting with severity levels
- [x] Integration tests (62 assertions)

---

## Phase 7: Cloud & Enterprise (Future)

**Goal:** Container orchestration, secrets management, and multi-region support.

### Planned Features

**Container Deployment:**
```prolog
:- deploy_method(docker, [
    registry('registry.example.com'),
    image_prefix('unifyweaver/'),
    orchestrator(kubernetes)
]).
```

**Secret Management:**
```prolog
:- secret_source(vault, [url('https://vault.example.com')]).
:- secret_source(aws_secrets, [region('us-east-1')]).
```

**Multi-Region:**
```prolog
:- region_config(api_service, [
    primary('us-east-1'),
    failover(['us-west-2', 'eu-west-1'])
]).
```

### Planned Deliverables
- [ ] Docker image building and registry push
- [ ] Kubernetes deployment manifests
- [ ] HashiCorp Vault integration
- [ ] AWS/Azure/GCP secrets management
- [ ] Multi-region deployment
- [ ] Geo-failover
- [ ] Cloud functions (Lambda, Cloud Functions)

---

## Summary

| Phase | Description | Status | Lines of Code |
|-------|-------------|--------|---------------|
| 1 | Foundation | ✅ Complete | ~600 |
| 2 | Shell Integration | ✅ Complete | ~650 |
| 3 | .NET Integration | ✅ Complete | ~1,550 |
| 4 | Native Targets | ✅ Complete | ~1,650 |
| 5 | Network Layer | ✅ Complete | ~2,150 |
| 6 | Production Ready | ✅ Complete | ~2,460 |
| 7 | Cloud & Enterprise | Future | - |

**Total implemented: ~9,060 lines across 6 modules**

## Module Summary

| Module | Purpose | Tests |
|--------|---------|-------|
| `target_registry.pl` | Target metadata management | 40+ |
| `target_mapping.pl` | Predicate-to-target declarations | 30+ |
| `pipe_glue.pl` | Basic pipe templates | - |
| `shell_glue.pl` | AWK/Python/Bash script generation | 30+ |
| `dotnet_glue.pl` | .NET bridge generation | 72 |
| `native_glue.pl` | Go/Rust binary orchestration | 62 |
| `network_glue.pl` | HTTP/socket communication | 90 |
| `deployment_glue.pl` | Deployment, error handling, monitoring | 62 |

## Success Metrics

- [x] Can compose 3+ targets in single pipeline
- [x] In-process .NET communication working
- [x] Remote target calls functional
- [x] Error handling with retry, fallback, circuit breaker
- [x] Production monitoring with metrics, logging, alerts
- [x] Performance overhead < 5% for in-process, < 10ms for pipes
