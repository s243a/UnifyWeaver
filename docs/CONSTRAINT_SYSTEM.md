<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2025 John William Creighton (@s243a)
-->
# UnifyWeaver Constraint System

## Overview

UnifyWeaver uses a **constraint annotation system** to control deduplication behavior for compiled predicates. The system supports two orthogonal constraint dimensions:

1.  **`unique`** - Whether to eliminate duplicate results
2.  **`unordered`** - Whether result order matters

## Defaults

**Explicit defaults:** `unique=true, unordered=true`

**Rationale:**
- Most Prolog queries don't care about duplicate order
- Matches typical Prolog behavior where result order is implementation-dependent
- Efficient: allows use of `sort -u` instead of hash tables
- Easy to override for temporal/ordered data

## Constraint Declaration Syntax

**Pragma-style (recommended):**
```prolog
:- constraint(grandparent/2, [unique, unordered]).
:- constraint(temporal_query/2, [unique, ordered]).
:- constraint(allow_duplicates/2, [unique(false)]).
```

**Programmatic:**
```prolog
declare_constraint(grandparent/2, [unique, unordered]).
declare_constraint(temporal_query/2, [unordered(false)]).
```

**Shorthand:**
- `unique` expands to `unique(true)`
- `unordered` expands to `unordered(true)`
- `ordered` expands to `unordered(false)`

## Deduplication Strategies

The compiler selects the deduplication strategy based on constraints:

| unique | unordered | Strategy   | Bash Implementation                          |
|--------|-----------|------------|----------------------------------------------|
| true   | true      | `sort -u`  | `... | sort -u`                              |
| true   | false     | Hash dedup | `declare -A seen` with order-preserving loop |
| false  | *         | No dedup   | Direct pipeline output                       |

**Sort -u (unique + unordered):**
```bash
grandparent() {
    parent_stream | parent_join | sort -u
}
```

**Hash dedup (unique + ordered):**
```bash
temporal_query() {
    declare -A seen
    base_pipeline | while IFS= read -r line; do
        if [[ -z "${seen[$line]}" ]]; then
            seen[$line]=1
            echo "$line"
        fi
    done
}
```

**No dedup (unique=false):**
```bash
allow_duplicates() {
    base_pipeline
}
```

## Runtime Overrides

Runtime options take precedence over declared constraints:

```prolog
% Declare as unordered
declare_constraint(my_pred/2, [unique, unordered]).

% Override at runtime to be ordered
compile_predicate(my_pred/2, [unordered(false)], Code).
% Result: Uses hash-based deduplication
```

## Changing Global Defaults

```prolog
% Change defaults to preserve order
set_default_constraints([unique(true), unordered(false)]).

% All undeclared predicates now use hash dedup by default
compile_predicate(new_pred/2, [], Code).
```

## Implementation

**Module:** [`constraint_analyzer.pl`](../src/unifyweaver/core/constraint_analyzer.pl)
- Manages constraint declarations
- Provides constraint queries
- Determines deduplication strategy

**Integration:** [`stream_compiler.pl`](../src/unifyweaver/core/stream_compiler.pl)
- Queries constraints for each predicate
- Merges with runtime options
- Generates appropriate bash code

**Tests:**
- [`constraint_analyzer.pl:test_constraint_analyzer/0`](../src/unifyweaver/core/constraint_analyzer.pl) - Unit tests
- [`test_constraints.pl:test_constraints/0`](../src/unifyweaver/core/test_constraints.pl) - Integration tests

## Examples

**Default behavior (sort -u):**
```prolog
% No declaration needed - uses defaults
compile_predicate(grandparent/2, [], Code).
% Generates: ... | sort -u
```

**Temporal/ordered data:**
```prolog
:- constraint(event_sequence/2, [unique, ordered]).
compile_predicate(event_sequence/2, [], Code).
% Generates: declare -A seen with order-preserving dedup
```

**Allow duplicates:**
```prolog
:- constraint(all_paths/2, [unique(false)]).
compile_predicate(all_paths/2, [], Code).
% Generates: no deduplication
```
