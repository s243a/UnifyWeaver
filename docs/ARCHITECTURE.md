<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2025 John William Creighton (@s243a)
-->
# UnifyWeaver Architecture

## Core Concept

UnifyWeaver compiles Prolog predicates into efficient bash scripts, treating Prolog as a declarative specification language and bash as the execution target. The system analyzes predicate structure and recursion patterns to generate optimized streaming implementations.

## Module Organization

```
src/unifyweaver/core/
├── template_system.pl      # Template rendering engine
├── stream_compiler.pl      # Non-recursive predicate compiler
└── recursive_compiler.pl   # Recursive pattern analyzer & compiler
```

### template_system.pl

Provides mustache-style template rendering for bash code generation:

```prolog
render_template(Template, Dict, Result)
```

**Features:**
- Named placeholder substitution (`{{name}}`)
- Composable template units
- Pre-defined templates for common bash patterns (BFS, streams, functions)

**Example:**
```prolog
Template = 'Hello {{name}}!',
render_template(Template, [name='World'], Result).
% Result = 'Hello World!'
```

### stream_compiler.pl

Compiles non-recursive predicates into bash streaming pipelines:

**Handles:**
- Facts (converted to associative arrays)
- Single rules (converted to pipelines)
- Multiple rules (OR patterns with `sort -u`)
- Inequality constraints (special case handling)

**Pipeline Strategy:**
```prolog
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
```
Becomes:
```bash
parent_stream | parent_join | sort -u
```

**Key Functions:**
- `compile_predicate/3` - Main entry point
- `classify_predicate/2` - Determines compilation strategy
- `generate_pipeline/3` - Creates streaming bash code

### recursive_compiler.pl

Analyzes and optimizes recursive predicates:

**Recursion Patterns Detected:**

1. **Transitive Closure** (optimized to BFS)
   ```prolog
   ancestor(X, Y) :- parent(X, Y).
   ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
   ```
   
2. **Tail Recursion** (detected, falls back to memoization)
   ```prolog
   count_acc([], Acc, Acc).
   count_acc([_|T], Acc, N) :- Acc1 is Acc + 1, count_acc(T, Acc1, N).
   ```

3. **Linear Recursion** (single recursive call per clause)

**Optimization Strategy:**
- Transitive closures → BFS with visited tracking
- Work queues in `/tmp/` for iterative processing
- Cycle detection with associative arrays
- Process-safe temp file naming

## Compilation Pipeline

```
┌─────────────────┐
│ Prolog Predicate│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Classify Pattern│  (recursive_compiler)
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌─────────┐ ┌──────────────┐
│Non-Rec  │ │ Recursive    │
│(stream) │ │ (BFS/memo)   │
└────┬────┘ └──────┬───────┘
     │             │
     ▼             ▼
┌─────────────────────┐
│ Template Rendering  │  (template_system)
└──────────┬──────────┘
           │
           ▼
    ┌─────────────┐
    │ Bash Script │
    └─────────────┘
```

## Generated Code Structure

### For Facts (Binary Predicates)

```bash
declare -A parent_data=(
    ["alice:bob"]=1
    ["bob:charlie"]=1
)

parent() {
    local key="$1:$2"
    [[ -n "${parent_data[$key]}" ]] && echo "$key"
}

parent_stream() {
    for key in "${!parent_data[@]}"; do
        echo "$key"
    done
}
```

### For Simple Rules

```bash
grandparent() {
    parent_stream | parent_join | sort -u
}

parent_join() {
    while IFS= read -r input; do
        IFS=":" read -r a b <<< "$input"
        for key in "${!parent_data[@]}"; do
            IFS=":" read -r c d <<< "$key"
            [[ "$b" == "$c" ]] && echo "$a:$d"
        done
    done
}
```

### For Transitive Closures

```bash
ancestor_all() {
    local start="$1"
    declare -A visited
    local queue_file="/tmp/ancestor_queue_$$"
    
    echo "$start" > "$queue_file"
    visited["$start"]=1
    
    while [[ -s "$queue_file" ]]; do
        > "$next_queue"
        while IFS= read -r current; do
            parent_stream | grep "^$current:" | while IFS=":" read -r from to; do
                if [[ -z "${visited[$to]}" ]]; then
                    visited["$to"]=1
                    echo "$to" >> "$next_queue"
                    echo "$start:$to"
                fi
            done
        done < "$queue_file"
        mv "$next_queue" "$queue_file"
    done
}
```

## Design Principles

### 1. Stream Everything
Data flows through pipelines rather than materializing in memory.

### 2. Bash Associative Arrays
Fast O(1) lookups for facts and visited tracking.

### 3. BFS over DFS
Prevents stack overflow and enables cycle detection in transitive closures.

### 4. Template Separation
Bash generation logic separated from Prolog analysis logic.

### 5. Composition
Generated bash functions can be sourced and composed in larger scripts.

## Testing

Each core module includes built-in tests:

```prolog
?- test_template_system.
?- test_stream_compiler.
?- test_recursive_compiler.
```

Tests generate example scripts in `output/` directory that can be executed directly.

## Future Extensions

### Planned
- External template file support (currently templates are auto-generated)
- Mutual recursion via SCC detection
- Tail recursion optimization to loops
- Multiple backend support (Python, JavaScript)

### Under Consideration
- Static analysis for optimization hints
- Parallel execution strategies
- Incremental compilation
- Query planning for complex joins

## Constraint System

### Overview

UnifyWeaver uses a **constraint annotation system** to control deduplication behavior for compiled predicates. The system supports two orthogonal constraint dimensions:

1. **`unique`** - Whether to eliminate duplicate results
2. **`unordered`** - Whether result order matters

### Defaults

**Explicit defaults:** `unique=true, unordered=true`

**Rationale:**
- Most Prolog queries don't care about duplicate order
- Matches typical Prolog behavior where result order is implementation-dependent
- Efficient: allows use of `sort -u` instead of hash tables
- Easy to override for temporal/ordered data

### Constraint Declaration Syntax

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

### Deduplication Strategies

The compiler selects the deduplication strategy based on constraints:

| unique | unordered | Strategy | Bash Implementation |
|--------|-----------|----------|---------------------|
| true   | true      | `sort -u` | `... | sort -u` |
| true   | false     | Hash dedup | `declare -A seen` with order-preserving loop |
| false  | *         | No dedup | Direct pipeline output |

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

### Runtime Overrides

Runtime options take precedence over declared constraints:

```prolog
% Declare as unordered
declare_constraint(my_pred/2, [unique, unordered]).

% Override at runtime to be ordered
compile_predicate(my_pred/2, [unordered(false)], Code).
% Result: Uses hash-based deduplication
```

### Changing Global Defaults

```prolog
% Change defaults to preserve order
set_default_constraints([unique(true), unordered(false)]).

% All undeclared predicates now use hash dedup by default
compile_predicate(new_pred/2, [], Code).
```

### Implementation

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

### Examples

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

## In Progress / Known Gaps

---

## References

- **Stream Processing**: Unix pipeline philosophy
- **BFS for Graphs**: Standard graph algorithm adapted for transitive closure
- **Template Rendering**: Mustache-style interpolation
- **Prolog Semantics**: SLD resolution adapted to procedural execution