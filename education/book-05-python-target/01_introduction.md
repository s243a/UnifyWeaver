<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 1: Introduction to the Python Target

The Python target compiles Prolog predicates into standalone, dependency-free Python scripts. It combines Prolog's declarative logic with Python's ubiquitous ecosystem, making it ideal for portable data processing, rapid prototyping, and integration with machine learning tools.

## Why Python?

Python is everywhere. It runs on servers, laptops, phones (Termux), and embedded devices. By compiling to Python, UnifyWeaver enables:

- **Portability** - Python 3.7+ runs on virtually any platform
- **Zero dependencies** - Generated scripts use only the standard library
- **Ecosystem integration** - Easy to combine with pandas, numpy, requests
- **Rapid prototyping** - No compilation step, just run
- **Readability** - Generated code is human-readable and debuggable

## Two Evaluation Modes

The Python target supports two distinct evaluation modes, each suited for different tasks:

### Procedural Mode (Default)

Compiles predicates to Python generator functions with streaming semantics.

**Best for:**
- Arithmetic computations (factorial, fibonacci)
- Tail-recursive predicates
- Deterministic transformations
- Performance-critical paths

```prolog
% Factorial compiles to a cached recursive function
factorial(0, 1).
factorial(N, F) :- N > 0, N1 is N - 1, factorial(N1, F1), F is N * F1.

?- compile_predicate_to_python(factorial/2, [], Code).
```

**Generated Python:**
```python
@functools.cache
def _factorial_worker(n):
    if n == 0:
        return 1
    return n * _factorial_worker(n - 1)
```

### Generator Mode (Semi-Naive Fixpoint)

Implements Datalog-style fixpoint evaluation for recursive graph queries.

**Best for:**
- Transitive closure (ancestor, reachability)
- Graph algorithms
- Unknown recursion depth
- Recursive joins

```prolog
% Path compiles to fixpoint iteration
path(X, Y) :- edge(X, Y).
path(X, Z) :- edge(X, Y), path(Y, Z).

?- compile_predicate_to_python(path/2, [mode(generator)], Code).
```

**Generated Python:**
```python
def process_stream_generator(records):
    total: Set[FrozenDict] = set()
    delta: Set[FrozenDict] = set()

    # Initialize with input facts
    for record in records:
        frozen = FrozenDict.from_dict(record)
        delta.add(frozen)
        total.add(frozen)
        yield record

    # Iterate until fixpoint
    while delta:
        new_delta: Set[FrozenDict] = set()
        for fact in delta:
            for new_fact in _apply_rules(fact, total):
                if new_fact not in total:
                    total.add(new_fact)
                    new_delta.add(new_fact)
                    yield new_fact.to_dict()
        delta = new_delta
```

## Comparison with Other Targets

| Feature | Python | Bash | Go | C# |
|---------|--------|------|----|----|
| **Startup time** | Medium | Fast | Slow (compile) | Slow (compile) |
| **Execution speed** | Good | Good | Excellent | Excellent |
| **Dependencies** | None* | None | None | .NET Runtime |
| **Streaming** | Yes | Yes | Yes | Yes |
| **Fixpoint eval** | Yes | No | Yes | Yes |
| **Deployment** | Copy file | Copy file | Copy binary | Build project |

\* Optional: `lxml` for XML sources, `onnxruntime` for embeddings

### When to Choose Python

**Choose Python when:**
- Rapid prototyping and experimentation
- Integration with Python libraries (pandas, sklearn)
- Cross-platform deployment without compilation
- Semantic AI features (embeddings, search)
- Debugging and inspecting generated code

**Choose Bash when:**
- Unix text processing
- System administration scripts
- Maximum portability (POSIX systems)

**Choose Go when:**
- Single binary deployment
- Maximum performance needed
- Concurrent processing

**Choose C# when:**
- .NET ecosystem integration
- Windows-first deployment
- Large-scale production systems

## API Overview

### Basic Compilation

```prolog
:- use_module('src/unifyweaver/targets/python_target').

% Compile with default options (procedural mode)
?- compile_predicate_to_python(my_pred/2, [], Code).

% Compile with generator mode
?- compile_predicate_to_python(my_pred/2, [mode(generator)], Code).
```

### Options

| Option | Values | Description |
|--------|--------|-------------|
| `mode(Mode)` | `procedural`, `generator` | Evaluation mode |
| `record_format(Fmt)` | `jsonl`, `nul_json` | I/O format |
| `input_source(Src)` | `xml(File, Tags)` | Native XML input |

### Input/Output Format

Generated scripts read from stdin and write to stdout using JSON Lines (JSONL):

**Input:**
```json
{"arg0": "john", "arg1": "mary"}
{"arg0": "mary", "arg1": "sue"}
```

**Output:**
```json
{"arg0": "john", "arg1": "mary"}
{"arg0": "john", "arg1": "sue"}
```

## Quick Start Example

Let's compile a simple predicate and run it:

### 1. Define the Predicate

```prolog
% In SWI-Prolog
:- use_module('src/unifyweaver/targets/python_target').

% Define facts
edge(a, b).
edge(b, c).
edge(c, d).

% Define transitive closure
path(X, Y) :- edge(X, Y).
path(X, Z) :- edge(X, Y), path(Y, Z).
```

### 2. Compile to Python

```prolog
?- compile_predicate_to_python(path/2, [mode(generator)], Code),
   open('path.py', write, S),
   write(S, Code),
   close(S).
```

### 3. Run the Generated Script

```bash
# Provide input facts
echo '{"arg0": "a", "arg1": "b"}
{"arg0": "b", "arg1": "c"}
{"arg0": "c", "arg1": "d"}' | python3 path.py
```

### 4. Output

```json
{"arg0": "a", "arg1": "b"}
{"arg0": "b", "arg1": "c"}
{"arg0": "c", "arg1": "d"}
{"arg0": "a", "arg1": "c"}
{"arg0": "b", "arg1": "d"}
{"arg0": "a", "arg1": "d"}
```

The script computed the transitive closure, deriving that `a` can reach `c` and `d` through intermediate nodes.

## What's Next

In the following chapters, we'll explore:

- **Chapter 2**: Procedural mode in depth - streaming, generators, arithmetic
- **Chapter 3**: Generator mode - fixpoint evaluation for graphs
- **Chapter 4**: Recursion patterns - tail, linear, and mutual recursion
- **Chapter 5**: Semantic predicates - search, crawling, embeddings

---

## Navigation

[📖 Book 5: Python Target](./) | [Next: Chapter 2: Procedural Mode →](02_procedural_mode)
