<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 1: Introduction to Cross-Target Communication

## Why Cross-Target Glue?

UnifyWeaver compiles Prolog predicates to multiple target languages. Each target has strengths:

| Target | Strengths |
|--------|-----------|
| AWK | Fast text processing, simple transformations |
| Python | Rich libraries (pandas, numpy), data science |
| Bash | System orchestration, file operations |
| Go | High performance, concurrency, single binary |
| Rust | Memory safety, zero-cost abstractions |
| C# | .NET ecosystem, enterprise integration |
| PowerShell | Windows administration, .NET access |
| SQL | Database queries, set operations |

Real-world problems often require combining these strengths. A data pipeline might:

1. **Fetch data** with Bash/curl
2. **Filter records** with AWK (fast, streaming)
3. **Transform data** with Python (pandas, complex logic)
4. **Store results** in SQL

Without cross-target glue, you'd manually:
- Write serialization code in each language
- Handle process management
- Debug format mismatches
- Maintain multiple versions

Cross-target glue automates all of this.

## The Unix Philosophy Extended

The Unix philosophy teaches us:

> "Write programs that do one thing and do it well. Write programs to work together."

Cross-target glue extends this philosophy to:
- **Multiple languages** - not just shell tools
- **Multiple machines** - distributed systems
- **Multiple paradigms** - functional, imperative, declarative

```
Traditional Unix:    grep | sort | uniq | wc
Cross-target glue:   awk_filter | python_transform | go_aggregate | sql_store
```

## Location and Transport Model

Cross-target glue separates two concerns:

### Location (Where it runs)

```
in_process      → Same runtime (e.g., C# calling PowerShell)
local_process   → Separate process, same machine (pipes)
remote(Host)    → Different machine (network)
```

### Transport (How they communicate)

```
direct          → Function call (in-process only)
pipe            → Unix pipes with TSV/JSON
socket          → TCP streaming
http            → REST API calls
```

This separation is key: **location is a deployment concern, not a logic concern**.

Your predicate logic doesn't change based on where it runs:

```prolog
% This logic is location-agnostic
process_data(Input, Output) :-
    fetch(Input, Raw),
    transform(Raw, Processed),
    store(Processed, Output).

% Deployment specifies location
:- declare_target(fetch/2, bash).
:- declare_target(transform/2, python).
:- declare_target(store/2, sql).
:- declare_location(store/2, [host('db.example.com')]).
```

## Runtime Families

Targets are grouped into runtime families. Targets in the same family can communicate more efficiently:

| Family | Targets | Default Communication |
|--------|---------|----------------------|
| .NET | C#, PowerShell, IronPython | In-process (zero serialization) |
| JVM | Java, Scala, Jython | In-process |
| Shell | Bash, AWK, sed | Pipes |
| Native | Go, Rust, C | Pipes or shared memory |
| Python | CPython | Pipes |

**Key insight**: C# calling PowerShell can pass objects directly (in-process). AWK calling Python must serialize data (pipes).

## Quick Start Example

Let's build a simple log analysis pipeline:

```prolog
% log_pipeline.pl
:- use_module('src/unifyweaver/glue/shell_glue').

% Generate a 3-stage pipeline
example_pipeline(Script) :-
    generate_pipeline(
        [
            step(filter, awk, 'filter.awk', []),
            step(analyze, python, 'analyze.py', []),
            step(report, awk, 'report.awk', [])
        ],
        [input('access.log')],
        Script
    ).
```

Run it:

```bash
$ swipl -l log_pipeline.pl -g "example_pipeline(S), write(S)" -t halt
```

Generated output:

```bash
#!/bin/bash
set -euo pipefail

cat "access.log" \
    | awk -f "filter.awk" \
    | python3 "analyze.py" \
    | awk -f "report.awk"
```

The glue system:
1. Generated the orchestration script
2. Connected stages with pipes
3. Set up proper error handling (`set -euo pipefail`)

## What Cross-Target Glue Provides

### 1. Automatic Code Generation

Generate complete, working scripts:

```prolog
generate_awk_script(Logic, Fields, Options, Script).
generate_python_script(Logic, Fields, Options, Script).
generate_go_pipe_main(Logic, Options, Code).
generate_rust_pipe_main(Logic, Options, Code).
```

### 2. Format Handling

Automatic serialization/deserialization:

```prolog
% TSV (default) - fast, simple
generate_pipeline(Steps, [format(tsv)], Script).

% JSON - structured, self-describing
generate_pipeline(Steps, [format(json)], Script).
```

### 3. In-Process Communication

For .NET family, avoid serialization entirely:

```prolog
% C# hosting PowerShell in-process
generate_powershell_bridge(Options, CSharpCode).

% Result: direct object passing, no pipes
```

### 4. Network Communication

For distributed systems:

```prolog
% Generate HTTP server
generate_go_http_server(Endpoints, Options, ServerCode).

% Generate HTTP client
generate_python_http_client(Services, Options, ClientCode).
```

### 5. Binary Orchestration

For high-performance native code:

```prolog
% Generate Go with parallel workers
generate_go_pipe_main(Logic, [parallel(8)], Code).

% Cross-compile for multiple platforms
generate_cross_compile(go, Source, [linux-amd64, darwin-arm64], Script).
```

## The Communication Flow

Here's how data flows through a typical cross-target pipeline:

```
┌──────────┐    TSV     ┌──────────┐    TSV     ┌──────────┐
│   AWK    │ ────────►  │  Python  │ ────────►  │   Go     │
│ (filter) │   pipe     │(transform)│   pipe    │(aggregate)│
└──────────┘            └──────────┘            └──────────┘
     │                       │                       │
     └───────────────────────┴───────────────────────┘
                             │
                    Orchestrator (Bash)
```

Each stage:
1. Reads from stdin
2. Parses the agreed format (TSV/JSON)
3. Processes records
4. Writes to stdout in the same format

The orchestrator connects them with pipes.

## Chapter Summary

- Cross-target glue enables multi-language composition
- Location (where) and transport (how) are separate concerns
- Runtime families determine optimal communication
- The system generates complete, working code
- You focus on logic, not plumbing

## Next Steps

In Chapter 2, we'll explore the design philosophy in depth:
- Location transparency principles
- How defaults are chosen
- When to override defaults
- Design patterns for hybrid systems

## Exercises

1. **Identify the pipeline**: Think of a data processing task you do regularly. What languages would be optimal for each stage?

2. **Family grouping**: For a C# application that needs Python ML models, which runtime should IronPython use? When would CPython be better?

3. **Format choice**: When would you choose JSON over TSV? When is TSV better?

## Advanced Example: Family Tree Pipeline

This complete example demonstrates a pipeline that processes family relationships - a classic Prolog problem. The key insight is that **UnifyWeaver compiles your Prolog predicates to target languages** - you don't write Python by hand.

### Step 1: Define the Prolog Predicates

First, define the classic ancestor relation in Prolog:

```prolog
% family.pl - The logic we want to compile
:- module(family, [parent/2, ancestor/2]).

% Base facts (will be loaded from data)
:- dynamic parent/2.

% The ancestor relation - classic transitive closure
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
```

### Step 2: Compile to Target Languages

UnifyWeaver detects that `ancestor/2` is a transitive closure over `parent/2` and generates optimized code:

```prolog
% compile_family.pl
:- use_module('src/unifyweaver/targets/python_target').
:- use_module('src/unifyweaver/targets/awk_target').
:- use_module('src/unifyweaver/core/recursive_compiler').

% Compile ancestor/2 to Python
% The system detects transitive closure pattern automatically
compile_ancestors :-
    % Classify the predicate - system detects it's transitive closure
    classify_predicate(ancestor/2, Classification),
    format("Detected pattern: ~w~n", [Classification]),

    % Compile to Python with streaming I/O
    compile_predicate_to_python(ancestor/2, [
        record_format(tsv),
        mode(generator)
    ], PythonCode),

    % Write the generated Python
    open('ancestors.py', write, Stream),
    write(Stream, PythonCode),
    close(Stream),
    format("Generated: ancestors.py~n").
```

### Step 3: Generated Python Code

UnifyWeaver generates this Python (you don't write it!):

```python
#!/usr/bin/env python3
"""
Generated by UnifyWeaver from ancestor/2
Pattern: transitive_closure(parent)
"""
import sys
from functools import cache

# Build relation from TSV input
_parent = {}

def _load_parent():
    for line in sys.stdin:
        parts = line.rstrip('\n').split('\t')
        if len(parts) >= 2:
            child, parent = parts[0], parts[1]
            _parent.setdefault(child, []).append(parent)

@cache
def _ancestor_worker(x):
    """Compute all ancestors of x (transitive closure)"""
    result = set()
    # Base case: direct parents
    for y in _parent.get(x, []):
        result.add(y)
        # Recursive case: ancestors of parents
        result.update(_ancestor_worker(y))
    return frozenset(result)

def ancestor(x, y):
    """ancestor(X, Y) :- parent(X, Y). ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y)."""
    return y in _ancestor_worker(x)

def process_stream():
    _load_parent()
    # Output all ancestor pairs
    for child in _parent:
        for anc in _ancestor_worker(child):
            print(f"{child}\t{anc}")

if __name__ == "__main__":
    process_stream()
```

### Step 4: Generate the Complete Pipeline

```prolog
% family_pipeline.pl
:- use_module('src/unifyweaver/glue/shell_glue').
:- use_module('src/unifyweaver/targets/python_target').

% Generate a pipeline using compiled predicates
family_pipeline(Script) :-
    % First, compile ancestor/2 to Python
    compile_predicate_to_python(ancestor/2, [
        record_format(tsv),
        base_relation(parent/2)
    ], AncestorPython),

    % Write the generated Python to a file
    open('ancestors.py', write, S),
    write(S, AncestorPython),
    close(S),

    % Now generate the pipeline script
    generate_pipeline(
        [
            % Stage 1: Parse family data (AWK - compiled from simple transform)
            step(parse, awk, '{print $1 "\t" $2}', [
                separator(" ")
            ]),

            % Stage 2: Run the COMPILED ancestor predicate
            step(ancestors, python, 'ancestors.py', []),

            % Stage 3: Format output (AWK)
            step(format, awk, '{print $1 " is descended from " $2}', [])
        ],
        [input('family.txt')],
        Script
    ).
```

### Sample Input (`family.txt`)

```
alice bob
alice carol
bob david
bob eve
carol frank
david george
```

### Running the Pipeline

```bash
$ swipl -l family_pipeline.pl -g "family_pipeline(S), write(S)" -t halt > run.sh
$ chmod +x run.sh
$ ./run.sh
```

### Expected Output

```
alice is descended from bob
alice is descended from carol
alice is descended from david
alice is descended from eve
alice is descended from frank
alice is descended from george
bob is descended from david
bob is descended from eve
bob is descended from george
carol is descended from frank
david is descended from george
```

### How UnifyWeaver Compiles This

1. **Pattern Detection**: `classify_predicate(ancestor/2, transitive_closure(parent))` recognizes the pattern
2. **Code Generation**: `compile_predicate_to_python/3` generates optimized Python with:
   - `@cache` decorator for memoization
   - Streaming input/output
   - Proper TSV parsing
3. **Pipeline Assembly**: `generate_pipeline/3` connects the stages

The key point: **you write Prolog, UnifyWeaver generates target code**.

```prolog
% What you write:
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).

% What UnifyWeaver generates: optimized Python with memoization
```

### Modification Exercise

**Task**: Add a `descendant/2` predicate and have UnifyWeaver compile both to the same Python file.

**What to do**:
1. Add to family.pl:
   ```prolog
   descendant(X, Y) :- ancestor(Y, X).
   ```

2. Use mutual compilation:
   ```prolog
   compile_predicate_group_to_python([ancestor/2, descendant/2], Options, Code).
   ```

**Hints**:
- UnifyWeaver detects that `descendant/2` is the inverse of `ancestor/2`
- Both can share the same memoization cache
- The generated code will have both `ancestor(x, y)` and `descendant(x, y)` functions

**Expected new output capability**:
```
alice is descended from bob
bob is ancestor of alice
```

## Further Reading

- Unix Philosophy: https://en.wikipedia.org/wiki/Unix_philosophy
- Pipes and Filters Pattern: https://www.enterpriseintegrationpatterns.com/patterns/messaging/PipesAndFilters.html
- Design documentation: `docs/design/cross-target-glue/01-philosophy.md`

---

## Navigation

[📖 Book 7: Cross-Target Glue](./) | [Next: Chapter 2: Philosophy and Design Principles →](02_philosophy)
