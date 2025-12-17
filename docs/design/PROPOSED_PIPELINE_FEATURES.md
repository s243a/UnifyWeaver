# Proposed Pipeline Features

## Overview

This document captures pipeline-related features that were proposed during design discussions. Some may be implemented as pipeline stages, others may be better suited as client-server services. This serves as a reference for future implementation decisions.

## Feature Categories

| Feature | Pipeline Stage? | Service? | Notes |
|---------|----------------|----------|-------|
| Cache | Maybe | Yes | Stateful, better as service |
| Checkpoint | Yes | Maybe | Write-aside persistence |
| Audit | Yes | Maybe | Metadata enrichment |
| Coalesce | Yes | No | Field merging |
| Profiling | Yes | Maybe | Metrics collection |
| State Monad | Complex | Yes | State threading |
| Pipeline Composition | Yes | No | Named sub-pipelines |
| Lazy Evaluation | Built-in | N/A | Already implemented |

---

## 1. Cache

### Original Proposal (Pipeline Stage)
```prolog
cache(expensive_api/1, id_field)        % Cache by key field
cache(transform/1, id, 3600)            % With TTL (seconds)
```

### Recommendation: Implement as Service
```prolog
service(cache_service, [stateful(true)], [
    receive(Request),
    route_by(op, [
        (get, [state_get(Request.key, Value), respond(Value)]),
        (set, [state_put(Request.key, Request.value), respond(ok)]),
        (get_or_compute, [
            state_get(Request.key, Cached),
            branch(is_cached,
                respond(Cached),
                [compute(Request), state_put(Request.key, Result), respond(Result)]
            )
        ])
    ])
]).

% Usage in pipeline
pipeline([
    parse/1,
    call_service(cache_service, {op: get_or_compute, key: id, compute: transform/1}, result),
    output/1
]).
```

### Why Service is Better
- Cache state persists across pipeline executions
- Can be shared across multiple pipelines
- Natural fit for stateful service pattern
- TTL, eviction policies are service concerns

---

## 2. Checkpoint

### Original Proposal (Pipeline Stage)
```prolog
checkpoint(jsonl, 'output/checkpoint.jsonl')
checkpoint(sqlite, 'data/state.db', [table(processed)])
```

### Recommendation: Keep as Pipeline Stage
Similar to `tee` but with persistence. Records pass through unchanged while being written to storage.

### Implementation Approach
```prolog
% Validation
is_valid_stage(checkpoint(Format, Location)) :-
    valid_checkpoint_format(Format),
    atom(Location).

valid_checkpoint_format(jsonl).
valid_checkpoint_format(sqlite).
valid_checkpoint_format(csv).

% Python implementation
def checkpoint_stage(stream, format, location):
    '''Write records to persistent storage while passing through.'''
    writer = get_checkpoint_writer(format, location)
    try:
        for record in stream:
            writer.write(record)
            yield record
    finally:
        writer.close()
```

### Resume Capability
Could be implemented as compilation option:
```prolog
compile_pipeline(Stages, [resume_from('checkpoint.jsonl', last_id)], Code).
```

---

## 3. Audit

### Original Proposal (Pipeline Stage)
```prolog
audit                                    % Track stage journey
audit([include_timing(true)])           % With performance data
audit([checkpoint('audit.jsonl')])      % Persist audit trail
```

### Recommendation: Keep as Pipeline Stage
Metadata enrichment that adds `_audit` field to records.

### Implementation Approach
```prolog
% Validation
is_valid_stage(audit).
is_valid_stage(audit(Options)) :- is_list(Options).

% Python implementation
def audit_stage(stream, stage_name, options=None):
    '''Enrich records with audit metadata.'''
    for record in stream:
        if '_audit' not in record:
            record['_audit'] = {'journey': [], 'created_at': now()}

        entry = {'stage': stage_name, 'timestamp': now()}
        if options.get('include_timing'):
            entry['duration_ms'] = measure_duration()

        record['_audit']['journey'].append(entry)
        yield record
```

### Integration with Checkpoint
```prolog
pipeline([
    parse/1,
    audit([include_timing(true)]),
    transform/1,
    audit,
    checkpoint(jsonl, 'audit_trail.jsonl'),  % Persist audit data
    output/1
]).
```

---

## 4. Coalesce

### Original Proposal (Pipeline Stage)
```prolog
coalesce([
    email/[primary_email, backup_email],  % First non-null wins
    name/[full_name, first_name]
])
```

### Recommendation: Keep as Pipeline Stage
Pure transformation, no state needed.

### Implementation Approach
```prolog
% Validation
is_valid_stage(coalesce(Mappings)) :-
    is_list(Mappings),
    maplist(is_valid_coalesce_mapping, Mappings).

is_valid_coalesce_mapping(Target/Sources) :-
    atom(Target),
    is_list(Sources),
    maplist(atom, Sources).

% Python implementation
def coalesce_stage(stream, mappings):
    '''Merge fields, keeping first non-null value.'''
    for record in stream:
        for target, sources in mappings.items():
            for source in sources:
                if source in record and record[source] is not None:
                    record[target] = record[source]
                    break
        yield record
```

### Use Cases
- Consolidating results from `fan_out`
- Schema migration (old field names → new)
- Default value handling

---

## 5. Profiling

### Original Proposal (Pipeline Stage)
```prolog
profile([latency, throughput, memory])
profile([latency], [sample_rate(0.1), checkpoint('metrics.jsonl')])
```

### Recommendation: Hybrid (Stage + Service)

**As Stage**: Collect metrics inline
```prolog
pipeline([
    parse/1,
    profile([latency]),
    transform/1,
    profile([latency, memory]),
    output/1
]).
```

**As Service**: Aggregate and query metrics
```prolog
service(metrics_service, [stateful(true)], [
    receive(MetricEvent),
    route_by(op, [
        (record, [update_aggregates/1, respond(ok)]),
        (query, [get_aggregates/1, respond(Stats)])
    ])
]).
```

### Implementation Approach
```python
def profile_stage(stream, metrics, options=None):
    '''Collect performance metrics.'''
    sample_rate = options.get('sample_rate', 1.0)

    for record in stream:
        if random.random() > sample_rate:
            yield record
            continue

        profile = {}
        if 'latency' in metrics:
            profile['latency_ms'] = measure_latency(record)
        if 'memory' in metrics:
            profile['memory_mb'] = get_memory_usage()
        if 'throughput' in metrics:
            profile['records_per_sec'] = calculate_throughput()

        record['_profile'] = profile
        yield record
```

---

## 6. State Monad Pattern

### Original Proposal
```prolog
with_state(InitialState, [
    stage1/1,    % Can read/modify state
    stage2/1,    % Receives state from stage1
    stage3/1     % Receives state from stage2
])
```

### Recommendation: Implement as Service Pattern
State threading is complex for pipeline users. Services provide a simpler model.

### Alternative: Explicit State Service
```prolog
service(pipeline_state, [stateful(true)], [
    receive(Op),
    route_by(Op.action, [
        (get, [state_get(Op.key, Value), respond(Value)]),
        (put, [state_put(Op.key, Op.value), respond(ok)]),
        (modify, [state_modify(Op.key, Op.func), respond(ok)])
    ])
]).

% Usage in pipeline
pipeline([
    parse/1,
    call_service(pipeline_state, {action: put, key: count, value: 0}, _),
    map(process_and_count/1),  % Calls state service internally
    call_service(pipeline_state, {action: get, key: count}, final_count),
    output/1
]).
```

### Why Service is Better
- Explicit state location (not hidden in pipeline)
- Testable in isolation
- Can persist state if needed
- Avoids functional programming complexity

---

## 7. Pipeline Composition

### Original Proposal
```prolog
% Named pipelines as reusable units
define_pipeline(validate_and_enrich, [
    validate/1,
    enrich/1,
    normalize/1
]).

% Use in other pipelines
pipeline([
    parse/1,
    use_pipeline(validate_and_enrich),
    output/1
]).
```

### Recommendation: Keep as Pipeline Feature
This is purely about code organization, not state.

### Implementation Approach
```prolog
% Store named pipelines
:- dynamic defined_pipeline/2.

define_pipeline(Name, Stages) :-
    assertz(defined_pipeline(Name, Stages)).

% Expand use_pipeline during compilation
expand_pipeline([]) --> [].
expand_pipeline([use_pipeline(Name)|Rest]) -->
    { defined_pipeline(Name, Stages) },
    Stages,
    expand_pipeline(Rest).
expand_pipeline([Stage|Rest]) -->
    [Stage],
    expand_pipeline(Rest).
```

### Benefits
- DRY (Don't Repeat Yourself)
- Named, documented pipeline fragments
- Easier testing of sub-pipelines

---

## 8. Lazy Evaluation

### Status: Already Implemented

UnifyWeaver pipelines are lazy by default:
- Python: Generator-based (`yield`)
- Go: Iterator pattern
- Rust: Iterator trait

### Potential Enhancement: Explicit Eager Stages
```prolog
% Force materialization at specific points
pipeline([
    parse/1,
    transform/1,
    materialize,      % Force all records into memory here
    expensive_sort/1, % Needs all records
    output/1
]).
```

This is already implicit in stages like `order_by`, `group_by`.

---

## Implementation Priority

Based on value and complexity:

| Priority | Feature | Approach | Rationale |
|----------|---------|----------|-----------|
| 1 | Coalesce | Pipeline stage | Simple, useful for fan_out |
| 2 | Audit | Pipeline stage | Debugging, compliance |
| 3 | Checkpoint | Pipeline stage | Persistence, resume |
| 4 | Pipeline Composition | Pipeline feature | Code organization |
| 5 | Profiling | Pipeline stage | Performance analysis |
| 6 | Cache | Service | Stateful, complex |
| 7 | State Monad | Service | Complex, niche use |

---

## Relationship to Client-Server

Many of these features benefit from or require client-server:

| Feature | Client-Server Benefit |
|---------|----------------------|
| Cache | Shared state across pipelines, persistence |
| Checkpoint | Resume from external process |
| Audit | Centralized audit log service |
| Profiling | Metrics aggregation service |
| State | Explicit state service |

The client-server architecture enables these features to be:
- Shared across multiple pipelines
- Persistent across executions
- Deployed and scaled independently
- Easier to test and monitor

---

*This document serves as a reference. Implementation decisions should consider the client-server architecture as the foundation for stateful features.*
