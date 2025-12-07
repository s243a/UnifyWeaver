<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 13: API Reference

## Overview

This chapter provides complete documentation for all predicates in the cross-target glue system.

## Target Registry (`target_registry.pl`)

### `register_target/3`

Register a new target with metadata.

```prolog
register_target(+Name, +Family, +Options).
```

**Arguments:**
- `Name` - Atom identifying the target (e.g., `python3`, `go`)
- `Family` - Runtime family (`shell`, `dotnet`, `jvm`, `native`, `python`)
- `Options` - List of target options

**Options:**
| Option | Values | Description |
|--------|--------|-------------|
| `location(L)` | `in_process`, `local_process`, `remote(Host)` | Where target runs |
| `transport(T)` | `direct`, `pipe`, `socket`, `http` | Communication method |
| `version(V)` | String | Version constraint |
| `capabilities(C)` | List | Supported features |

**Example:**
```prolog
register_target(python3, shell, [
    location(local_process),
    transport(pipe),
    version('>=3.8'),
    capabilities([json, csv, async])
]).
```

### `target/2`

Query registered targets.

```prolog
target(?Name, ?Family).
```

**Example:**
```prolog
?- target(Name, shell).
Name = awk ;
Name = bash ;
Name = python3.
```

### `target_options/2`

Get options for a registered target.

```prolog
target_options(+Name, -Options).
```

### `unregister_target/1`

Remove a target from the registry.

```prolog
unregister_target(+Name).
```

---

## Target Mapping (`target_mapping.pl`)

### `compatible_targets/3`

Find targets compatible with given constraints.

```prolog
compatible_targets(+SourceFamily, +Constraints, -Targets).
```

**Constraints:**
| Constraint | Description |
|------------|-------------|
| `location(L)` | Filter by location type |
| `transport(T)` | Filter by transport type |
| `capability(C)` | Require specific capability |

**Example:**
```prolog
?- compatible_targets(shell, [capability(json)], Targets).
Targets = [python3, jq].
```

### `optimal_bridge/4`

Find the optimal bridge between two targets.

```prolog
optimal_bridge(+Source, +Dest, +Preferences, -Bridge).
```

**Preferences:**
| Preference | Description |
|------------|-------------|
| `prefer(latency)` | Minimize latency |
| `prefer(throughput)` | Maximize throughput |
| `prefer(simplicity)` | Minimize complexity |

### `bridge_cost/2`

Calculate the cost of a bridge configuration.

```prolog
bridge_cost(+Bridge, -Cost).
```

---

## Shell Glue (`shell_glue.pl`)

### `generate_awk_script/3`

Generate an AWK script for field processing.

```prolog
generate_awk_script(+Body, +Options, -Script).
```

**Options:**
| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `separator(S)` | String | `"\t"` | Field separator |
| `output_separator(S)` | String | `"\t"` | Output separator |
| `header(B)` | `true`/`false` | `false` | Skip header line |

**Example:**
```prolog
?- generate_awk_script(
    'print $1, $2 * 2',
    [separator(",")],
    Script
).
```

### `generate_python_script/3`

Generate a Python script for line processing.

```prolog
generate_python_script(+Body, +Options, -Script).
```

**Options:**
| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `fields(F)` | List of atoms | `[]` | Named field bindings |
| `separator(S)` | String | `"\t"` | Field separator |
| `imports(I)` | List | `[]` | Additional imports |

**Example:**
```prolog
?- generate_python_script(
    'output(name.upper(), str(int(age) + 1))',
    [fields([name, age, city])],
    Script
).
```

### `generate_bash_script/3`

Generate a Bash script.

```prolog
generate_bash_script(+Body, +Options, -Script).
```

**Options:**
| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `strict(B)` | `true`/`false` | `true` | Use `set -euo pipefail` |
| `variables(V)` | List | `[]` | Environment variables |

### `generate_pipeline/3`

Generate a multi-stage pipeline script.

```prolog
generate_pipeline(+Steps, +Options, -Script).
```

**Step Format:**
```prolog
step(Name, Type, Code, StepOptions)
```

**Types:** `awk`, `python`, `bash`, `jq`, `sed`, `sort`, `uniq`

**Example:**
```prolog
?- generate_pipeline(
    [
        step(parse, awk, '{print $1, $2}', []),
        step(filter, python, 'if int(v) > 10: output(k, v)', [fields([k, v])]),
        step(sort, bash, 'sort -t$\'\\t\' -k2 -n', [])
    ],
    [input('data.tsv'), output('result.tsv')],
    Script
).
```

### `generate_pipe_wrapper/4`

Generate a wrapper script for an existing command.

```prolog
generate_pipe_wrapper(+Command, +Target, +Options, -Script).
```

---

## .NET Glue (`dotnet_glue.pl`)

### `generate_powershell_bridge/2`

Generate a PowerShell bridge for C#.

```prolog
generate_powershell_bridge(+Options, -Code).
```

**Options:**
| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `input_format(F)` | `tsv`, `json` | `tsv` | Input data format |
| `output_format(F)` | `tsv`, `json` | `tsv` | Output data format |
| `namespace(N)` | String | `"Pipeline"` | C# namespace |

### `generate_ironpython_bridge/2`

Generate an IronPython bridge.

```prolog
generate_ironpython_bridge(+Options, -Code).
```

### `generate_cpython_bridge/2`

Generate a CPython subprocess bridge.

```prolog
generate_cpython_bridge(+Options, -Code).
```

### `ironpython_compatible/1`

Check if a Python module is IronPython-compatible.

```prolog
ironpython_compatible(+Module).
```

**Compatible modules:**
```
sys, os, io, re, string, json, csv, xml, collections, itertools,
functools, datetime, math, random, hashlib, pathlib, threading,
socket, sqlite3, typing, abc, contextlib, copy, pickle, struct,
base64, shutil, tempfile, zipfile, gzip, heapq, bisect, array,
operator, codecs, urllib, clr
```

### `can_use_ironpython/1`

Check if all imports are IronPython-compatible.

```prolog
can_use_ironpython(+Imports).
```

### `python_runtime_choice/2`

Automatically select Python runtime.

```prolog
python_runtime_choice(+Imports, -Runtime).
```

**Returns:** `ironpython` or `cpython_pipe`

---

## Native Glue (`native_glue.pl`)

### `detect_go/1`

Detect Go installation.

```prolog
detect_go(-Version).
```

### `detect_rust/1`

Detect Rust installation.

```prolog
detect_rust(-Version).
```

### `generate_go_pipe_main/3`

Generate a Go pipe-compatible program.

```prolog
generate_go_pipe_main(+Body, +Options, -Code).
```

**Options:**
| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `format(F)` | `tsv`, `json` | `tsv` | Data format |
| `fields(F)` | List | `[]` | Field names (for JSON) |
| `parallel(N)` | Integer | `1` | Worker count |
| `buffer_size(N)` | Integer | `1048576` | Read buffer size |

**Example:**
```prolog
?- generate_go_pipe_main(
    '
    age, _ := strconv.Atoi(fields[1])
    if age > 30 {
        return fields
    }
    return nil
    ',
    [fields([name, age, city])],
    Code
).
```

### `generate_rust_pipe_main/3`

Generate a Rust pipe-compatible program.

```prolog
generate_rust_pipe_main(+Body, +Options, -Code).
```

**Options:** Same as Go, plus:
| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `serde(B)` | `true`/`false` | `false` | Include serde derives |

### `generate_go_build_script/3`

Generate a Go build script.

```prolog
generate_go_build_script(+Source, +Options, -Script).
```

**Options:**
| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `optimize(B)` | `true`/`false` | `true` | Strip debug symbols |
| `output(O)` | String | basename | Output binary name |

### `generate_rust_build_script/3`

Generate a Rust build script.

```prolog
generate_rust_build_script(+Source, +Options, -Script).
```

### `generate_cross_compile/4`

Generate cross-compilation script.

```prolog
generate_cross_compile(+Language, +Source, +Targets, -Script).
```

**Targets:**
```
linux-amd64, linux-arm64, darwin-amd64, darwin-arm64, windows-amd64
```

---

## Network Glue (`network_glue.pl`)

### `register_service/3`

Register a remote service.

```prolog
register_service(+Name, +URL, +Options).
```

**Options:**
| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `timeout(S)` | Integer | `30` | Request timeout |
| `retries(N)` | Integer | `0` | Retry count |
| `format(F)` | `json` | `json` | Request format |

### `service/2`

Query registered services.

```prolog
service(?Name, ?URL).
```

### `endpoint_url/3`

Build endpoint URL.

```prolog
endpoint_url(+Service, +Path, -URL).
```

### `generate_http_server/4`

Generate HTTP server code.

```prolog
generate_http_server(+Target, +Endpoints, +Options, -Code).
```

**Endpoint Format:**
```prolog
endpoint(Path, Handler, EndpointOptions)
```

**EndpointOptions:**
| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `methods(M)` | List | `['POST']` | HTTP methods |

**Server Options:**
| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `port(P)` | Integer | `8080` | Listen port |
| `cors(B)` | `true`/`false` | `true` | Enable CORS |
| `host(H)` | String | `"0.0.0.0"` | Bind address |

### `generate_go_http_server/3`

Generate Go HTTP server.

```prolog
generate_go_http_server(+Endpoints, +Options, -Code).
```

### `generate_python_http_server/3`

Generate Python Flask server.

```prolog
generate_python_http_server(+Endpoints, +Options, -Code).
```

### `generate_rust_http_server/3`

Generate Rust Actix-web server.

```prolog
generate_rust_http_server(+Endpoints, +Options, -Code).
```

### `generate_http_client/4`

Generate HTTP client code.

```prolog
generate_http_client(+Target, +Services, +Options, -Code).
```

**Service Definition:**
```prolog
service_def(Name, BaseURL, Endpoints)
```

### `generate_socket_server/4`

Generate socket server.

```prolog
generate_socket_server(+Target, +Port, +Options, -Code).
```

### `generate_socket_client/4`

Generate socket client.

```prolog
generate_socket_client(+Target, +Address, +Options, -Code).
```

### `generate_network_pipeline/3`

Generate network pipeline.

```prolog
generate_network_pipeline(+Steps, +Options, -Code).
```

**Step Format:**
```prolog
step(Name, Location, Target, StepOptions)
```

**Location:** `local` or `remote`

**Pipeline Options:**
| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `target(T)` | `python`, `bash`, `go` | `python` | Output language |
| `retry(N)` | Integer | `0` | Retry count |
| `backoff(B)` | `linear`, `exponential` | `linear` | Backoff strategy |
| `circuit_breaker(C)` | `failures(N), reset_after(S)` | none | Circuit breaker |
| `logging(L)` | `none`, `verbose` | `none` | Logging level |
| `metrics(M)` | `none`, `prometheus` | `none` | Metrics system |

---

## Error Codes

| Code | Description |
|------|-------------|
| `target_not_found` | Target not in registry |
| `incompatible_targets` | No bridge between targets |
| `missing_capability` | Target lacks required capability |
| `compilation_failed` | Native code compilation error |
| `service_unavailable` | Remote service unreachable |
| `timeout` | Operation timed out |
| `invalid_format` | Data format mismatch |

### Error Handling

```prolog
% Catch and handle errors
generate_pipeline(Steps, Options, Script) :-
    catch(
        generate_pipeline_impl(Steps, Options, Script),
        Error,
        handle_generation_error(Error, Script)
    ).

handle_generation_error(target_not_found(T), _) :-
    format(user_error, "Target not found: ~w~n", [T]),
    fail.
```

---

## Performance Tuning

### Buffer Sizes

```prolog
% For high-throughput Go processing
generate_go_pipe_main(Body, [
    buffer_size(10485760),  % 10MB buffer
    parallel(16)            % 16 workers
], Code).
```

### Pipeline Optimization

```prolog
% Enable GNU parallel for massive files
generate_pipeline(Steps, [
    parallel(gnu_parallel),
    jobs(32),
    chunk(100000)
], Script).
```

### Memory Limits

```prolog
% Constrain memory usage
generate_native_pipeline(Steps, [
    memory_limit('4G'),
    cpu_cores([0,1,2,3])
], Script).
```

---

## Common Patterns

### Pattern 1: Type-Safe Pipeline

```prolog
generate_typed_pipeline(
    [
        step(parse, awk, '{print $1, $2}', []),
        step(validate, python, 'validate(fields)', []),
        step(transform, go, 'transform(fields)', [])
    ],
    [type_check(true)],
    Script
).
```

### Pattern 2: Resilient Service Call

```prolog
generate_network_pipeline(
    [step(fetch, remote, 'http://api/data', [])],
    [
        retry(3),
        backoff(exponential),
        circuit_breaker(failures(5), reset_after(60)),
        fallback(cache)
    ],
    Code
).
```

### Pattern 3: Multi-Target Build

```prolog
generate_multiplatform_build(
    [
        binary(transform, go, 'transform.go'),
        binary(aggregate, rust, 'src/main.rs')
    ],
    [linux-amd64, darwin-arm64, windows-amd64],
    Script
).
```

---

## Version Compatibility

| Module | Minimum Prolog | Dependencies |
|--------|---------------|--------------|
| target_registry | SWI 8.0 | - |
| target_mapping | SWI 8.0 | target_registry |
| shell_glue | SWI 8.0 | - |
| dotnet_glue | SWI 8.0 | shell_glue |
| native_glue | SWI 8.0 | shell_glue |
| network_glue | SWI 8.0 | shell_glue |

---

## Next Steps

In Chapter 14, we'll explore real-world case studies demonstrating these APIs in action.

---

## Navigation

**←** [Previous: Chapter 12: Distributed Pipelines](12_distributed_pipelines) | [📖 Book 7: Cross-Target Glue](README) | [Next: Chapter 14: Case Studies →](14_case_studies)
