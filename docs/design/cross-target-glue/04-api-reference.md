# Cross-Target Glue API Reference

Complete API documentation for UnifyWeaver's cross-target glue system.

## Module Index

| Module | Location | Purpose |
|--------|----------|---------|
| [`target_registry`](#target_registry) | `src/unifyweaver/core/` | Target metadata management |
| [`target_mapping`](#target_mapping) | `src/unifyweaver/core/` | Predicate-to-target declarations |
| [`pipe_glue`](#pipe_glue) | `src/unifyweaver/glue/` | Basic pipe I/O templates |
| [`shell_glue`](#shell_glue) | `src/unifyweaver/glue/` | AWK/Python/Bash script generation |
| [`dotnet_glue`](#dotnet_glue) | `src/unifyweaver/glue/` | .NET bridge generation |
| [`native_glue`](#native_glue) | `src/unifyweaver/glue/` | Go/Rust binary orchestration |
| [`network_glue`](#network_glue) | `src/unifyweaver/glue/` | HTTP/socket communication |
| [`deployment_glue`](#deployment_glue) | `src/unifyweaver/glue/` | Deployment, error handling, monitoring |

---

## target_registry

Target metadata management and runtime family classification.

### Predicates

#### `register_target(+Name, +Family, +Capabilities)`
Register a new compilation target.

```prolog
register_target(myrust, native, [compiled, typed, async]).
```

**Parameters:**
- `Name`: Atom identifying the target
- `Family`: Runtime family (shell, python, dotnet, jvm, native, database)
- `Capabilities`: List of capability atoms

#### `unregister_target(+Name)`
Remove a registered target.

#### `target_exists(+Name)`
Check if a target is registered.

#### `target_family(?Target, ?Family)`
Query or verify target's runtime family.

```prolog
?- target_family(python, Family).
Family = python.
```

#### `target_capabilities(?Target, ?Capabilities)`
Get capabilities list for a target.

#### `targets_same_family(+Target1, +Target2)`
Check if two targets share a runtime family.

```prolog
?- targets_same_family(csharp, powershell).
true.
```

#### `family_targets(+Family, -Targets)`
List all targets in a runtime family.

```prolog
?- family_targets(native, Targets).
Targets = [go, rust, c, cpp].
```

#### `list_targets(-Targets)`
List all registered targets.

#### `list_families(-Families)`
List all runtime families.

#### `default_location(+Target, -Location)`
Get default execution location for a target.

**Locations:**
- `in_process` - Same runtime
- `local_process` - Separate process, same machine
- `remote(Host)` - Different machine

#### `default_transport(+Location1, +Location2, -Transport)`
Get default transport between locations.

**Transports:**
- `direct` - Function call (in-process only)
- `pipe` - Unix pipes with TSV/JSON
- `socket` - TCP streaming
- `http` - REST API calls

---

## target_mapping

Maps predicates to compilation targets and execution locations.

### Predicates

#### `declare_target(+Pred/Arity, +Target)`
Declare which target compiles a predicate.

```prolog
:- declare_target(filter/2, awk).
:- declare_target(transform/2, python).
```

#### `declare_target(+Pred/Arity, +Target, +Options)`
Declare target with additional options.

```prolog
:- declare_target(analyze/2, python, [location(local_process)]).
```

**Options:**
- `location(Location)` - Execution location
- `transport(Transport)` - Communication method
- `format(Format)` - Data format (tsv, json)

#### `undeclare_target(+Pred/Arity)`
Remove target declaration.

#### `declare_location(+Pred/Arity, +Options)`
Declare execution location for a predicate.

```prolog
:- declare_location(remote_pred/2, [host('worker.example.com'), port(8080)]).
```

#### `declare_connection(+Pred1/Arity1, +Pred2/Arity2, +Options)`
Declare how two predicates communicate.

```prolog
:- declare_connection(filter/2, transform/2, [format(json), buffer(line)]).
```

#### `predicate_target(?Pred/Arity, ?Target)`
Query predicate-target mapping.

#### `predicate_location(?Pred/Arity, ?Location)`
Query predicate execution location.

#### `resolve_location(+Pred/Arity, -Location)`
Resolve location with defaults.

#### `resolve_transport(+Pred1, +Pred2, -Transport)`
Resolve transport between predicates.

#### `list_mappings(-Mappings)`
List all predicate-target mappings.

#### `validate_mapping(+Pred/Arity, -Errors)`
Validate a mapping configuration.

---

## pipe_glue

Generate pipe I/O code for inter-target communication.

### Predicates

#### `generate_pipe_writer(+Target, +Fields, +Options, -Code)`
Generate code to write records to stdout.

```prolog
generate_pipe_writer(python, [name, age, city], [format(tsv)], Code).
```

#### `generate_pipe_reader(+Target, +Fields, +Options, -Code)`
Generate code to read records from stdin.

```prolog
generate_pipe_reader(awk, [ip, timestamp, status], [format(tsv)], Code).
```

#### `generate_tsv_writer(+Target, +Fields, -Code)`
Generate TSV output code.

#### `generate_tsv_reader(+Target, +Fields, -Code)`
Generate TSV input parsing code.

#### `generate_json_writer(+Target, +Fields, -Code)`
Generate JSON output code.

#### `generate_json_reader(+Target, +Fields, -Code)`
Generate JSON input parsing code.

#### `generate_pipeline_script(+Steps, +Options, -Script)`
Generate shell script orchestrating pipeline steps.

---

## shell_glue

Complete script generation for shell-based targets.

### Predicates

#### `generate_awk_script(+Logic, +Fields, +Options, -Script)`
Generate complete AWK script with I/O handling.

```prolog
generate_awk_script(
    'if ($3 >= 400) { print ip, timestamp, status }',
    [ip, timestamp, method, path, status],
    [format(tsv), header(true)],
    Script
).
```

**Parameters:**
- `Logic`: AWK code for the main block
- `Fields`: Input field names (assigned to variables)
- `Options`: Configuration options
- `Script`: Generated complete AWK script

#### `generate_python_script(+Logic, +Fields, +Options, -Script)`
Generate complete Python script with I/O handling.

```prolog
generate_python_script(
    'result = transform(record)\nprint_record(result)',
    [name, age, city],
    [input_format(tsv), output_format(json)],
    Script
).
```

#### `generate_bash_script(+Logic, +Fields, +Options, -Script)`
Generate Bash script with field parsing.

#### `generate_pipeline(+Steps, +Options, -Script)`
Generate pipeline orchestration script.

```prolog
generate_pipeline(
    [
        step(filter, awk, 'filter.awk', []),
        step(transform, python, 'transform.py', []),
        step(aggregate, awk, 'aggregate.awk', [])
    ],
    [input('data.tsv'), output('result.tsv')],
    Script
).
```

**Step Format:** `step(Name, Target, File, StepOptions)`

#### `input_format(+Options, -Format)`
Extract input format from options.

#### `output_format(+Options, -Format)`
Extract output format from options.

### Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `format(F)` | tsv, csv, json | tsv | I/O format |
| `input_format(F)` | tsv, csv, json | tsv | Input format |
| `output_format(F)` | tsv, csv, json | tsv | Output format |
| `header(B)` | true, false | false | Skip/emit header line |
| `input(File)` | path | stdin | Input file for pipeline |
| `output(File)` | path | stdout | Output file for pipeline |

---

## dotnet_glue

.NET ecosystem bridge generation for C#, PowerShell, and Python.

### Runtime Detection

#### `detect_dotnet_runtime(-Runtime)`
Detect available .NET runtime.

```prolog
?- detect_dotnet_runtime(R).
R = dotnet_modern.  % .NET 5+
```

**Values:** `dotnet_modern`, `dotnet_core`, `mono`, `none`

#### `detect_ironpython(-Available)`
Check if IronPython is available.

#### `detect_powershell(-Version)`
Detect PowerShell version.

**Values:** `core(Version)`, `windows(Version)`, `none`

### Compatibility Checking

#### `ironpython_compatible(+Module)`
Check if a Python module works with IronPython.

```prolog
?- ironpython_compatible(json).
true.

?- ironpython_compatible(numpy).
false.
```

**Compatible Modules:** sys, os, json, re, collections, csv, xml, datetime, math, itertools, functools, clr, random, string, io, codecs, hashlib, base64, urllib, socket, threading, copy, operator, struct, array, bisect, heapq, contextlib, abc, typing, pathlib, shutil, tempfile, zipfile, gzip, pickle, sqlite3

**Incompatible Modules:** numpy, pandas, scipy, tensorflow, torch, sklearn, cv2, PIL, matplotlib

#### `can_use_ironpython(+Imports)`
Check if all imports are IronPython-compatible.

```prolog
?- can_use_ironpython([sys, json, re]).
true.

?- can_use_ironpython([numpy, json]).
false.
```

#### `python_runtime_choice(+Imports, -Runtime)`
Choose best Python runtime for given imports.

```prolog
?- python_runtime_choice([json, sys], Runtime).
Runtime = ironpython.

?- python_runtime_choice([numpy, pandas], Runtime).
Runtime = cpython_pipe.
```

### Bridge Generation

#### `generate_powershell_bridge(+Options, -Code)`
Generate C# code hosting PowerShell.

```prolog
generate_powershell_bridge([namespace('MyApp'), async(true)], Code).
```

**Options:**
- `namespace(N)` - C# namespace (default: 'UnifyWeaver.Glue')
- `class(C)` - Class name (default: 'PowerShellBridge')
- `async(B)` - Generate async methods

**Generated API:**
```csharp
PowerShellBridge.Invoke<TInput, TOutput>(script, input)
PowerShellBridge.InvokeStream<TInput, TOutput>(script, stream)
```

#### `generate_ironpython_bridge(+Options, -Code)`
Generate C# code hosting IronPython.

**Generated API:**
```csharp
IronPythonBridge.Execute(pythonCode)
IronPythonBridge.ExecuteWithInput<TInput>(script, input)
IronPythonBridge.ExecuteWithOutput<TOutput>(script)
```

#### `generate_cpython_bridge(+Options, -Code)`
Generate C# code calling CPython via subprocess.

**Options:**
- `python_path(P)` - Python executable (default: 'python3')
- `format(F)` - Pipe format (tsv, json)

**Generated API:**
```csharp
CPythonBridge.Execute<TInput, TOutput>(script, input)
CPythonBridge.ExecuteStream<TInput, TOutput>(script, stream)
```

#### `generate_csharp_host(+Bridges, +Options, -Code)`
Generate complete C# host with multiple bridges.

```prolog
generate_csharp_host([powershell, ironpython], [namespace('Pipeline')], Code).
```

#### `generate_dotnet_pipeline(+Steps, +Options, -Code)`
Generate .NET pipeline using bridges.

---

## native_glue

Go and Rust binary orchestration.

### Binary Management

#### `register_binary(+Pred/Arity, +Target, +Path, +Options)`
Register a compiled binary.

```prolog
register_binary(transform/2, go, './bin/transform', [optimized(true)]).
```

#### `compiled_binary(?Pred/Arity, ?Target, ?Path)`
Query registered binaries.

#### `compile_if_needed(+Pred/Arity, +Target, +SourcePath, -BinaryPath)`
Compile source if binary is stale.

### Toolchain Detection

#### `detect_go(-Version)`
Detect Go installation.

```prolog
?- detect_go(V).
V = '1.21.0'.
```

#### `detect_rust(-Version)`
Detect Rust installation.

#### `detect_cargo(-Version)`
Detect Cargo installation.

### Go Code Generation

#### `generate_go_pipe_main(+Logic, +Options, -Code)`
Generate Go main() with pipe I/O.

```prolog
generate_go_pipe_main(
    'return transform(fields)',
    [format(tsv), parallel(8)],
    Code
).
```

**Options:**
- `format(F)` - tsv or json
- `parallel(N)` - Worker goroutine count
- `fields(L)` - Field names (JSON mode)
- `buffer(N)` - Channel buffer size

#### `generate_go_wrapper(+FuncName, +Schema, +Options, -Code)`
Generate Go function wrapper.

#### `generate_go_build_script(+SourcePath, +Options, -Script)`
Generate Go build script.

**Options:**
- `optimize(B)` - Enable optimizations
- `output(Path)` - Binary output path
- `ldflags(Flags)` - Linker flags

### Rust Code Generation

#### `generate_rust_pipe_main(+Logic, +Options, -Code)`
Generate Rust main() with pipe I/O.

```prolog
generate_rust_pipe_main(
    'transform(&fields)',
    [format(json)],
    Code
).
```

#### `generate_rust_wrapper(+FuncName, +Schema, +Options, -Code)`
Generate Rust function wrapper.

#### `generate_rust_build_script(+SourcePath, +Options, -Script)`
Generate Cargo build script.

### Cross-Compilation

#### `cross_compile_targets(-Targets)`
List supported cross-compilation targets.

```prolog
?- cross_compile_targets(T).
T = [linux-amd64, linux-arm64, darwin-amd64, darwin-arm64, windows-amd64].
```

#### `generate_cross_compile(+Target, +Source, +Platforms, -Script)`
Generate cross-compilation script.

```prolog
generate_cross_compile(go, 'main.go', [linux-amd64, darwin-arm64], Script).
```

### Pipeline Orchestration

#### `generate_native_pipeline(+Steps, +Options, -Script)`
Generate script orchestrating native binaries.

```prolog
generate_native_pipeline(
    [
        step(transform, go, './transform', [parallel(4)]),
        step(aggregate, rust, './aggregate', [])
    ],
    [input('data.jsonl')],
    Script
).
```

---

## network_glue

HTTP and socket communication for distributed systems.

### Service Registry

#### `register_service(+Name, +URL, +Options)`
Register a remote service.

```prolog
register_service(ml_api, 'http://ml.example.com:8080', [timeout(60)]).
```

**Options:**
- `timeout(Seconds)` - Request timeout
- `retries(N)` - Retry count
- `format(F)` - json or tsv
- `auth(Method)` - bearer(Token) or basic(User, Pass)

#### `service(?Name, ?URL)`
Query registered services.

#### `service_options(?Name, ?Options)`
Query service options.

#### `unregister_service(+Name)`
Remove a service registration.

#### `endpoint_url(+Service, +Endpoint, -URL)`
Construct full endpoint URL.

```prolog
?- endpoint_url(ml_api, '/predict', URL).
URL = 'http://ml.example.com:8080/predict'.
```

### HTTP Server Generation

#### `generate_http_server(+Target, +Endpoints, +Options, -Code)`
Generate HTTP server for any supported target.

#### `generate_go_http_server(+Endpoints, +Options, -Code)`
Generate Go HTTP server (net/http).

```prolog
generate_go_http_server(
    [
        endpoint('/api/process', process_handler, []),
        endpoint('/api/batch', batch_handler, [methods(['POST'])]),
        endpoint('/health', health, [methods(['GET'])])
    ],
    [port(8080), cors(true)],
    Code
).
```

**Endpoint Format:** `endpoint(Path, Handler, EndpointOptions)`

**Endpoint Options:**
- `methods(List)` - HTTP methods (default: ['POST'])

#### `generate_python_http_server(+Endpoints, +Options, -Code)`
Generate Python Flask server.

#### `generate_rust_http_server(+Endpoints, +Options, -Code)`
Generate Rust Actix-web server.

### HTTP Client Generation

#### `generate_http_client(+Target, +Services, +Options, -Code)`
Generate HTTP client for any supported target.

#### `generate_go_http_client(+Services, +Options, -Code)`
Generate Go HTTP client.

```prolog
generate_go_http_client(
    [service_def(ml_api, 'http://ml:8080', ['/predict', '/classify'])],
    [timeout(30)],
    Code
).
```

**Service Definition:** `service_def(Name, BaseURL, Endpoints)`

#### `generate_python_http_client(+Services, +Options, -Code)`
Generate Python requests client.

#### `generate_bash_http_client(+Services, +Options, -Code)`
Generate Bash client using curl + jq.

### Socket Communication

#### `generate_socket_server(+Target, +Port, +Options, -Code)`
Generate TCP socket server.

```prolog
generate_socket_server(go, 9000, [buffer_size(65536)], Code).
```

#### `generate_socket_client(+Target, +Host, +Options, -Code)`
Generate TCP socket client.

```prolog
generate_socket_client(python, 'localhost:9000', [timeout(10)], Code).
```

### Network Pipeline

#### `generate_network_pipeline(+Steps, +Options, -Code)`
Generate distributed pipeline with local and remote steps.

```prolog
generate_network_pipeline(
    [
        step(preprocess, local, './preprocess', []),
        step(analyze, remote, 'http://worker:8081/analyze', []),
        step(aggregate, local, './aggregate', [])
    ],
    [],
    Code
).
```

### Server/Client Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `port(P)` | integer | 8080 | Listen port |
| `cors(B)` | true, false | true | Enable CORS headers |
| `timeout(S)` | integer | 30 | Request timeout (seconds) |
| `buffer_size(N)` | integer | 65536 | Socket buffer size |
| `format(F)` | json, tsv | json | Request/response format |

### API Response Format

All HTTP services use a consistent JSON schema:

```json
// Request
{"data": <any>}

// Success Response
{"success": true, "data": <result>}

// Error Response
{"success": false, "error": "<message>"}
```

---

## Common Option Reference

### Format Options

| Option | Description | Supported Values |
|--------|-------------|------------------|
| `format(F)` | Data format | tsv, csv, json |
| `input_format(F)` | Input format | tsv, csv, json |
| `output_format(F)` | Output format | tsv, csv, json |

### Location Options

| Option | Description | Example |
|--------|-------------|---------|
| `location(L)` | Execution location | in_process, local_process, remote('host') |
| `host(H)` | Remote hostname | 'worker.example.com' |
| `port(P)` | Port number | 8080 |

### Performance Options

| Option | Description | Example |
|--------|-------------|---------|
| `parallel(N)` | Worker count | 8 |
| `buffer(N)` | Buffer size | 65536 |
| `timeout(S)` | Timeout seconds | 30 |
| `retries(N)` | Retry attempts | 3 |

### Build Options

| Option | Description | Example |
|--------|-------------|---------|
| `optimize(B)` | Enable optimizations | true |
| `output(Path)` | Output path | './bin/app' |
| `ldflags(F)` | Linker flags | '-s -w' |

---

## Error Handling

All generation predicates fail gracefully with meaningful error messages.

### Validation Errors

```prolog
?- validate_mapping(unknown_pred/2, Errors).
Errors = [no_target_declared, predicate_not_found].
```

### Runtime Detection Failures

```prolog
?- detect_go(V).
V = none.  % Go not installed
```

### Compatibility Failures

```prolog
?- python_runtime_choice([numpy], Runtime).
Runtime = cpython_pipe.  % Fallback to CPython
```

---

## deployment_glue

Production-ready deployment, error handling, and monitoring.

### Service Declarations

#### `declare_service(+Name, +Options)`
Declare a remote service with configuration.

```prolog
declare_service(ml_predictor, [
    host('ml.example.com'),
    port(8080),
    target(python),
    entry_point('server.py'),
    transport(https)
]).
```

**Options:**
- `host(H)` - Remote hostname
- `port(P)` - Service port (default: 8080)
- `target(T)` - Compilation target
- `entry_point(F)` - Main file
- `transport(T)` - http | https (default: https for remote)

#### `service_config(?Name, ?Options)`
Query service configuration.

#### `undeclare_service(+Name)`
Remove service declaration.

### Deployment Methods

#### `declare_deploy_method(+Service, +Method, +Options)`
Configure deployment method.

```prolog
declare_deploy_method(ml_predictor, ssh, [
    user('deploy'),
    remote_dir('/opt/services')
]).
```

**Methods:** `ssh`, `local`

**SSH Options:**
- `user(U)` - SSH username
- `remote_dir(D)` - Remote directory
- `agent(B)` - Use SSH agent (default: true)

#### `deploy_method_config(?Service, ?Method, ?Options)`
Query deployment configuration.

### Source Tracking

#### `declare_service_sources(+Service, +Patterns)`
Track source files for change detection.

```prolog
declare_service_sources(ml_predictor, ['src/**/*.py', 'requirements.txt']).
```

#### `compute_source_hash(+Service, -Hash)`
Compute SHA256 hash of tracked sources.

#### `check_for_changes(+Service, -Changes)`
Check for source changes since last deployment.

**Returns:** `no_changes`, `changed(Old, New)`, or `never_deployed`

### Security

#### `validate_security(+Service, -Errors)`
Validate security configuration.

```prolog
?- validate_security(insecure_service, Errors).
Errors = [remote_requires_encryption('remote.example.com')].
```

#### `requires_encryption(+Service)`
Check if service requires encrypted transport.

#### `is_local_service(+Service)`
Check if service is on localhost.

### Deployment Operations

#### `deploy_service(+Service, -Result)`
Deploy service (checks for changes first).

#### `generate_deploy_script(+Service, +Options, -Script)`
Generate deployment shell script.

#### `generate_systemd_unit(+Service, +Options, -Unit)`
Generate systemd service unit file.

### Lifecycle

#### `start_service(+Service, -Result)`
Start a deployed service.

#### `stop_service(+Service, -Result)`
Stop a running service.

#### `restart_service(+Service, -Result)`
Restart a service.

#### `service_status(+Service, -Status)`
Query service status.

**Status:** `running(Pid, Since)`, `stopped`, `deploying`, `error(Msg)`

### Multi-Host Deployment

#### `declare_service_hosts(+Service, +Hosts)`
Configure multiple deployment hosts.

```prolog
declare_service_hosts(api, [
    host_config('api1.example.com', [user('deploy')]),
    host_config('api2.example.com', [user('deploy')])
]).
```

#### `deploy_to_all_hosts(+Service, -Results)`
Deploy to all configured hosts.

### Rollback

#### `deploy_with_rollback(+Service, -Result)`
Deploy with automatic rollback on health check failure.

#### `rollback_service(+Service, -Result)`
Manually rollback to previous deployment.

#### `store_rollback_hash(+Service, +Hash)`
Store version for potential rollback.

### Graceful Shutdown

#### `graceful_stop(+Service, +Options, -Result)`
Stop service with connection draining.

**Options:**
- `drain_timeout(S)` - Wait for connections (seconds)
- `force_after(S)` - Force kill after (seconds)

### Retry Policies

#### `declare_retry_policy(+Service, +Policy)`
Configure retry behavior.

```prolog
declare_retry_policy(ml_service, [
    max_retries(5),
    initial_delay(1000),
    max_delay(30000),
    backoff(exponential),
    multiplier(2),
    retry_on([timeout, connection_refused]),
    fail_on([400, 401, 403])
]).
```

**Backoff:** `exponential`, `linear`, `fixed`

#### `call_with_retry(+Service, +Operation, +Args, -Result)`
Execute operation with retry policy.

**Result:** `ok(Value)` | `error(max_retries_exceeded)`

### Fallback

#### `declare_fallback(+Service, +Fallback)`
Configure fallback behavior.

```prolog
declare_fallback(primary, backup_service(secondary)).
declare_fallback(ml_service, default_value(fallback)).
declare_fallback(data_service, cache([key(data), ttl(3600)])).
declare_fallback(api, custom(my_handler)).
```

#### `call_with_fallback(+Service, +Operation, +Args, -Result)`
Execute with fallback on failure.

### Circuit Breaker

#### `declare_circuit_breaker(+Service, +Config)`
Configure circuit breaker.

```prolog
declare_circuit_breaker(ml_service, [
    failure_threshold(5),
    success_threshold(3),
    half_open_timeout(30000)
]).
```

#### `circuit_state(?Service, ?State)`
Query circuit state: `closed`, `open`, `half_open`

#### `call_with_circuit_breaker(+Service, +Operation, +Args, -Result)`
Execute with circuit protection.

**Result:** `ok(Value)` | `error(circuit_open)` | `error(Reason)`

#### `reset_circuit_breaker(+Service)`
Reset circuit to closed state.

### Timeouts

#### `declare_timeouts(+Service, +Timeouts)`
Configure timeout values.

```prolog
declare_timeouts(ml_service, [
    connect_timeout(5000),
    read_timeout(30000),
    total_timeout(60000)
]).
```

#### `call_with_timeout(+Service, +Operation, +Args, -Result)`
Execute with timeout.

**Result:** `ok(Value)` | `error(timeout)`

### Combined Protection

#### `protected_call(+Service, +Operation, +Args, -Result)`
Execute with all error handling strategies.

**Order:** circuit breaker → timeout → retry → fallback

### Health Monitoring

#### `declare_health_check(+Service, +Config)`
Configure health checks.

```prolog
declare_health_check(api_service, [
    endpoint('/health'),
    interval(30),
    timeout(5),
    unhealthy_threshold(3),
    healthy_threshold(2)
]).
```

#### `health_status(?Service, ?Status)`
Query health status: `healthy`, `unhealthy`, `unknown`

#### `start_health_monitor(+Service, -Result)`
Start health monitoring.

#### `stop_health_monitor(+Service)`
Stop health monitoring.

### Metrics

#### `declare_metrics(+Service, +Config)`
Configure metrics collection.

```prolog
declare_metrics(api_service, [
    collect([request_count, latency]),
    labels([service-api]),
    export(prometheus),
    retention(3600)
]).
```

#### `record_metric(+Service, +Metric, +Value)`
Record a metric value.

#### `get_metrics(+Service, -Metrics)`
Get all metrics for a service.

#### `generate_prometheus_metrics(+Service, -Output)`
Export metrics in Prometheus format.

### Logging

#### `declare_logging(+Service, +Config)`
Configure structured logging.

```prolog
declare_logging(api_service, [
    level(info),
    format(json),
    output(stdout),
    max_entries(1000)
]).
```

**Levels:** `debug`, `info`, `warn`, `error`
**Formats:** `json`, `text`
**Output:** `stdout`, `file(Path)`

#### `log_event(+Service, +Level, +Message, +Data)`
Log an event.

```prolog
log_event(api, info, 'Request received', [method-'GET', path-'/api']).
```

#### `get_log_entries(+Service, +Options, -Entries)`
Query log entries.

**Options:**
- `level(L)` - Minimum level
- `limit(N)` - Max entries
- `since(T)` - Since timestamp

### Alerting

#### `declare_alert(+Service, +AlertName, +Config)`
Define an alert.

```prolog
declare_alert(api, high_error_rate, [
    condition('error_rate > 0.05'),
    severity(critical),
    cooldown(300),
    notify([slack('#alerts'), pagerduty])
]).
```

**Severity:** `critical`, `warning`, `info`
**Channels:** `slack(C)`, `email(A)`, `pagerduty`, `webhook(U)`

#### `trigger_alert(+Service, +AlertName, +Data)`
Trigger an alert.

#### `check_alerts(+Service, -TriggeredAlerts)`
Get triggered alerts.

#### `alert_history(+Service, +Options, -History)`
Query alert history.

---

## See Also

- [User Guide](../../guides/cross-target-glue.md) - Practical usage patterns
- [Philosophy](01-philosophy.md) - Design principles
- [Specification](02-specification.md) - Protocol details
- [Implementation Plan](03-implementation-plan.md) - Phase status
