<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 4: Prolog Service (Bash-as-a-Service Alternative)

**Status:** 🚧 Experimental (Infrastructure exists, limited testing)
**Module:** `src/unifyweaver/targets/prolog_service_target.pl`
**Examples:** `examples/demo_prolog_service_partitioning.pl` (background processes)
**Tests:** Limited

---

## Introduction

The Prolog service target generates bash scripts that invoke Prolog for processing, similar to how Python/Perl services work. This provides:
- **Fallback when pattern matching fails** - Use full Prolog for complex queries
- **Integration with partitioning** - Process data chunks in parallel via Prolog
- **Declarative processing** - Leverage Prolog's backtracking and unification

### Comparison to Other Services

| Service | Use Case | Availability | Performance |
|---------|----------|--------------|-------------|
| **Bash** | Pattern-matched code | Always | Fastest |
| **Perl** | Text processing | Near-universal | Fast |
| **Python** | Data science | Common | Medium |
| **Prolog** | Logic queries | If installed | Variable |

---

## Architecture

### Module Interface

```prolog
:- module(prolog_service_target, [
    generate_bash_with_prolog_service/3
]).
```

**Key Predicate:**
```prolog
generate_bash_with_prolog_service(
    +PredicateList,
    +Options,
    -BashCode
).
```

---

## Example: Partitioning with Prolog

### Prolog Predicates

```prolog
% examples/demo_prolog_service_partitioning.pl

% Read stdin lines
read_stdin_lines(Lines) :-
    read_string(user_input, _, Str),
    split_string(Str, "\n", "", Lines).

% Partition into N chunks
partition_stdin(N, Partitions) :-
    read_stdin_lines(Lines),
    length(Lines, Total),
    ChunkSize is ceiling(Total / N),
    partition_by_size(Lines, ChunkSize, Partitions).

% Write partitions to files
write_partitions(Partitions) :-
    % ... write logic ...
```

### Generated Bash

```bash
#!/bin/bash
# Prolog service wrapper

swipl -q -g "
    use_module(library(lists)),
    read_stdin_lines(Lines),
    partition_stdin(3, Partitions),
    write_partitions(Partitions),
    halt
"
```

---

## Integration with Partitioning

### Use Case: Parallel Processing

```prolog
% Partition data
partition_predicate(
    large_dataset/1,
    [partitions(10)],
    bash_code
) :-
    generate_prolog_partition_service(
        [large_dataset/1],
        [chunks(10), output_dir('/tmp/partitions')],
        bash_code
    ).
```

Generated bash:
1. Invokes Prolog to split data
2. Writes N partition files
3. Processes each in parallel
4. Merges results

---

## Current Limitations

### What Works

✅ Basic predicate invocation
✅ Stdin/stdout piping
✅ Integration with partitioner module
✅ File-based result collection

### What Doesn't Work Yet

❌ **No templates** - Cannot generate full bash scripts easily
❌ **No aggregation** - unique/unordered not implemented
❌ **Limited error handling** - Prolog errors don't propagate cleanly
❌ **No optimization** - Queries run as-is, no planning
❌ **Performance overhead** - Prolog startup time significant

---

## Comparison to Bash-as-a-Service (BaaS)

### BaaS (Existing)

- Calls bash functions recursively
- Used when recursion is too complex for direct compilation
- Fast (no interpreter startup)
- Limited to bash semantics

### Prolog Service (This Chapter)

- Calls Prolog interpreter
- Used when logic is too complex for pattern matching
- Slower (interpreter startup overhead)
- Full Prolog semantics available

### When to Use Each

| Scenario | Recommended | Why |
|----------|-------------|-----|
| Deep recursion (fibonacci) | BaaS | Faster, pattern matched |
| Complex backtracking | Prolog Service | Need unification |
| Large datasets | Neither | Use streaming bash |
| Firewall forbids bash | Prolog Service | Pure Prolog fallback |

---

## Future Directions

### Needed Features

1. **Template Integration**
   - Generate bash wrappers like Perl service
   - Handle I/O redirection
   - Error propagation

2. **Optimization**
   - Query planning (like C# target)
   - Index hints
   - Join reordering

3. **Aggregation**
   - Implement unique via `setof/3`
   - Implement ordered via `sort/2`
   - Custom aggregators

4. **Debugging Support**
   - Trace mode integration
   - Step-through execution
   - Query profiling

---

## Related Work

- **Chapter 1:** Perl Service - Similar architecture, different language
- **Chapter 3:** Prolog as Target - Pure query generation (no bash wrapper)
- **Book 1, Chapter 13:** Partitioning - Integration point

---

**Status:** Experimental - Use with caution, expect API changes

---

**Authors:** John William Creighton (@s243a), Claude Code (Sonnet 4.5)
**Last Updated:** 2025-11-05

---

## Navigation

**←** [Previous: Chapter 3: Prolog as Target Language](03_prolog_target) | [📖 Back to Book-Misc](README)

