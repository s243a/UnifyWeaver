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

## In Progress / Known Gaps

### Order-Independence Analysis

**Design Intent:**
The compiler should flag operations as order-independent before applying `sort -u` for deduplication.

**Current State:**
- `sort -u` is used for operations like `grandparent` and `related` (which ARE order-independent)
- Associative arrays (`declare -A seen`) used in recursive operations for order-preserving deduplication
- **Missing:** Explicit order-independence checking/flagging in compilation pipeline

**Intended Implementation:**
```prolog
% Flag predicates as order-independent
order_independent(grandparent/2).
order_independent(related/2).

% Check before using sort -u
generate_pipeline(Predicates, Options, Pipeline) :-
    extract_result_predicate(Predicates, ResultPred),
    (   order_independent(ResultPred) ->
        format(string(Pipeline), '~s | sort -u', [BasePipeline])
    ;   % Use hash-based deduplication preserving order
        format(string(Pipeline), '~s | dedup_preserving_order', [BasePipeline])
    ).
```

**Status:** Related code may have been removed during refactoring. Needs reimplementation.

**Impact:** Currently assumes operations are order-independent. Works correctly for current examples but could produce incorrect results for temporal or ordered data.

**Priority:** Medium - current code works for existing use cases, but should be added before expanding to order-sensitive domains.

---

## References

- **Stream Processing**: Unix pipeline philosophy
- **BFS for Graphs**: Standard graph algorithm adapted for transitive closure
- **Template Rendering**: Mustache-style interpolation
- **Prolog Semantics**: SLD resolution adapted to procedural execution