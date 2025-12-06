# Cross-Target Glue: Philosophy

## Overview

UnifyWeaver compiles Prolog predicates to multiple target languages. Cross-target glue enables these targets to communicate and compose, creating hybrid systems where each component runs in its optimal environment.

## Core Principles

### 1. Location Transparency

Predicates should be callable regardless of where they execute. The caller shouldn't need to know if a predicate runs:
- In the same process
- In a separate process on the same machine
- On a remote machine

```prolog
% The caller doesn't care where 'analyze' runs
result(X) :-
    fetch_data(RawData),      % Maybe Bash/curl
    analyze(RawData, Result),  % Maybe Python/pandas
    store(Result).             % Maybe SQL
```

### 2. Sensible Defaults with Override

The system should "just work" with intelligent defaults, but allow explicit control when needed.

**Default behaviors:**
- Same runtime family → same process (when possible)
- Different runtimes → process pipes
- Remote targets → network sockets

**Explicit override:**
```prolog
:- target_location(analyze/2, python, [process(separate), buffer(line)]).
:- target_location(store/1, sql, [host('db.example.com'), port(5432)]).
```

### 3. Runtime Family Affinity

Languages sharing a runtime should prefer in-process communication:

| Runtime Family | Languages | Communication |
|----------------|-----------|---------------|
| .NET/CLR | C#, F#, PowerShell, IronPython | In-process calls |
| JVM | Java, Scala, Clojure, Jython | In-process calls |
| Native | Go, Rust, C | Shared memory or pipes |
| Interpreted | Bash, AWK, Perl, Python | Process pipes |

### 4. Data Format Negotiation

Targets should agree on data formats automatically:

```
Same process:     Native objects (no serialization)
Process pipes:    TSV (default), JSON, or binary
Network:          JSON (default), Protocol Buffers, or custom
```

### 5. Streaming by Default

Data should flow as streams, not batch loads:

```
Producer → Pipe → Consumer → Pipe → Next Consumer
```

This enables:
- Memory efficiency (bounded buffers)
- Pipeline parallelism
- Early termination (consumer can stop producer)

## Design Goals

### Composability

Any target should compose with any other target:

```prolog
% AWK preprocessing → Go processing → SQL storage
pipeline(Input, Output) :-
    awk_filter(Input, Filtered),
    go_transform(Filtered, Transformed),
    sql_insert(Transformed, Output).
```

### Minimal Overhead

- In-process: Zero serialization cost
- Same machine: Minimal pipe overhead
- Network: Efficient binary protocols available

### Failure Handling

- Graceful degradation when targets unavailable
- Clear error propagation across boundaries
- Timeout handling for remote calls

### Debuggability

- Trace data flow across target boundaries
- Log serialization/deserialization
- Profile communication overhead

## Non-Goals (Initially)

These are valuable but deferred:

1. **Distributed transactions** - Too complex for v1
2. **Automatic load balancing** - Requires orchestration
3. **Service discovery** - Use explicit configuration first
4. **Authentication/encryption** - Layer on top later

## Use Cases

### Use Case 1: ETL Pipeline

```
Bash (curl) → AWK (filter) → Python (transform) → SQL (load)
     ↓              ↓               ↓                ↓
   HTTP          Pipe            Pipe            DB conn
```

### Use Case 2: .NET Integration

```
PowerShell (orchestration)
     ↓ (in-process)
C# (business logic)
     ↓ (in-process)
IronPython (ML model)
```

### Use Case 3: Microservice Boundary

```
Go Service A ←──HTTP/gRPC──→ Rust Service B
      ↓                            ↓
   Local DB                   Local Cache
```

## Philosophical Foundations

### Unix Philosophy Extended

The Unix philosophy of "small tools connected by pipes" extended to:
- Multiple languages
- Multiple machines
- Multiple paradigms

### Declarative Location

Location is a property of the deployment, not the logic:

```prolog
% Logic is location-agnostic
process(X, Y) :- transform(X, T), validate(T, Y).

% Deployment specifies location
:- deploy(transform/2, [target(python), host(worker1)]).
:- deploy(validate/2, [target(rust), host(worker2)]).
```

### Gradual Typing of Communication

Start simple, add constraints as needed:

```prolog
% Level 1: Just works (TSV over pipes)
:- uses_target(analyze/2, python).

% Level 2: Specify format
:- uses_target(analyze/2, python, [format(json)]).

% Level 3: Full control
:- uses_target(analyze/2, python, [
    format(json),
    process(separate),
    timeout(30),
    retry(3)
]).
```

## Next Steps

1. **Specification** - Define the exact API and behaviors
2. **Implementation Plan** - Phased approach to building this
3. **Prototype** - Start with Bash ↔ AWK ↔ Python pipes
