# Client-Server Architecture: Philosophy & Vision

## Overview

This document describes the philosophical foundation for extending UnifyWeaver beyond unidirectional pipelines to bidirectional client-server communication patterns.

## The Journey So Far

UnifyWeaver began with a simple but powerful abstraction: **pipelines**.

```
Input → Stage1 → Stage2 → Stage3 → Output
```

Pipelines model data flowing in one direction through a series of transformations. This is powerful for:
- ETL (Extract, Transform, Load)
- Stream processing
- Data validation and enrichment
- Record-by-record transformations

But pipelines have a fundamental limitation: **data flows one way**.

## The Limitation of Unidirectional Flow

Consider these scenarios that pipelines struggle with:

1. **Lookup enrichment**: For each record, query a database and merge results
2. **Validation services**: Send record to validator, receive pass/fail with errors
3. **Stateful computation**: Accumulate state, query it later
4. **Interactive processing**: Request clarification, wait for response
5. **Caching with invalidation**: Check cache, populate on miss, invalidate on update

These patterns require **bidirectional communication** - the ability to send a request and receive a response.

## The Insight: Two Opposing Pipes

A client-server interaction can be modeled as **two pipes flowing in opposite directions**:

```
┌─────────────────────────────────────────────────────┐
│                 Client-Server Model                  │
│                                                      │
│    ┌────────┐                      ┌────────┐       │
│    │ Client │ ──── Request ────→  │ Server │       │
│    │        │ ←─── Response ────  │        │       │
│    └────────┘                      └────────┘       │
│                                                      │
│    Request Pipe:   Client → Server                  │
│    Response Pipe:  Server → Client                  │
│                                                      │
└─────────────────────────────────────────────────────┘
```

This preserves the pipeline mental model while enabling bidirectional patterns.

## Generalization: Transport Independence

Just as pipelines can operate at different levels:

| Pipeline Level | Mechanism |
|----------------|-----------|
| In-process | Generator/iterator chains |
| Cross-process | Unix pipes, stdin/stdout |
| Cross-runtime | JSONL serialization |
| Cross-machine | (not yet implemented) |

Client-server should follow the same pattern:

| Client-Server Level | Mechanism |
|---------------------|-----------|
| In-process | Function calls with return values |
| Cross-process | Unix sockets, named pipes |
| Cross-runtime | Protocol-based (JSONL, MessagePack) |
| Network | HTTP, gRPC, WebSocket, TCP |

The key insight: **the same service definition should work at any level**.

## Design Principles

### 1. Composition Over Configuration

Services should compose like pipeline stages:

```prolog
% A service is just a bidirectional pipeline
service(user_enrichment, [
    receive(user_id),
    lookup_in_database/1,
    format_response/1,
    respond(user_record)
]).

% Services can call other services
service(full_enrichment, [
    receive(record),
    call_service(user_enrichment, user_id, user_data),
    call_service(org_enrichment, org_id, org_data),
    merge_enrichments/1,
    respond(enriched_record)
]).
```

### 2. Location Transparency

The caller shouldn't need to know if the service is:
- A local function
- A separate process
- A remote server

```prolog
% Same syntax regardless of where service runs
pipeline([
    parse/1,
    call_service(validator, record, validation_result),
    handle_validation/1,
    output/1
]).
```

### 3. Protocol Consistency

Request and response should use the same serialization as pipelines:
- JSONL for simplicity and debugging
- Optional binary protocols for performance

### 4. Graceful Degradation

When a service is unavailable:
- Configurable fallback behavior
- Retry with backoff
- Circuit breaker patterns
- Timeout handling

These patterns already exist in pipeline stages (`try_catch`, `retry`, `timeout`).

### 5. Stateful by Design

Unlike stateless pipelines, servers can maintain state:
- Connection state
- Cached computations
- Accumulated data
- Session context

This is where patterns like caching naturally belong - as **server-side concerns**, not pipeline stages.

## Relationship to Existing Features

### Pipelines Are Still Primary

Client-server doesn't replace pipelines - it extends them:

```prolog
% Pipeline that uses services
pipeline([
    parse/1,
    call_service(cache, lookup, cached_result),
    branch(cache_hit,
        use_cached/1,
        call_service(compute, transform, fresh_result)
    ),
    output/1
]).
```

### Services Can Contain Pipelines

A service handler can be a full pipeline:

```prolog
service(batch_processor, [
    receive(batch),
    % Internal pipeline for processing
    pipeline([
        unbatch,
        validate/1,
        transform/1,
        filter_by(valid),
        batch(100)
    ]),
    respond(processed_batch)
]).
```

### Existing Stages Apply

Error handling, branching, and observation stages work in services:

```prolog
service(robust_lookup, [
    receive(query),
    try_catch(
        database_query/1,
        respond_error/1
    ),
    tap(log_query),
    respond(result)
]).
```

## Use Cases Enabled

### 1. Lookup Services
```prolog
service(user_service, [
    receive(user_id),
    query_user_db/1,
    respond(user_record)
]).
```

### 2. Validation Services
```prolog
service(schema_validator, [
    receive(record),
    validate_against_schema/1,
    respond(validation_result)
]).
```

### 3. Caching Services
```prolog
service(cache_service, [
    receive(cache_request),
    route_by(operation, [
        (get, cache_get/1),
        (set, cache_set/1),
        (invalidate, cache_invalidate/1)
    ]),
    respond(cache_response)
]).
```

### 4. Aggregation Services
```prolog
service(stats_collector, [
    receive(data_point),
    update_running_stats/1,  % Stateful
    respond(current_stats)
]).
```

### 5. Orchestration Services
```prolog
service(workflow_orchestrator, [
    receive(workflow_request),
    call_service(step1_service, input1, result1),
    call_service(step2_service, result1, result2),
    call_service(step3_service, result2, result3),
    respond(workflow_result)
]).
```

## What This Is NOT

### Not a Full RPC Framework
We're not building gRPC or Thrift. The focus is on:
- Simple request/response patterns
- Composition with pipelines
- Multi-target code generation

### Not a Distributed Systems Framework
We're not solving:
- Consensus
- Distributed transactions
- Service mesh
- Container orchestration

### Not Replacing Pipelines
Pipelines remain the primary abstraction for data flow. Services extend the model for bidirectional needs.

## Success Criteria

The client-server architecture is successful if:

1. **Natural Extension**: Feels like a logical extension of pipelines, not a separate system
2. **Simple Cases Are Simple**: In-process services require minimal boilerplate
3. **Complex Cases Are Possible**: Network services with retries, timeouts work
4. **Multi-Target**: Generates correct code for Python, Go, Rust
5. **Composable**: Services compose with each other and with pipelines
6. **Testable**: Services can be unit tested in isolation

## Next Steps

1. **Specification**: Define exact syntax and semantics
2. **Protocol Design**: Request/response format
3. **Implementation Plan**: Phased approach starting with in-process
4. **Proof of Concept**: Simple service in Python target

---

*This document establishes the philosophical foundation. See `CLIENT_SERVER_SPECIFICATION.md` for technical details.*
