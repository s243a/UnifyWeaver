# Python Target User Guide

## Overview

The Python target compiles Prolog predicates to standalone Python scripts. It supports **two evaluation modes**:

1. **Procedural Mode** (default) - Direct, fast execution
2. **Generator Mode** - Fixpoint iteration for graphs and transitive closure

## Basic Usage

### Procedural Mode (Default)

```prolog
compile_predicate_to_python(factorial/2, [], Code).
```

**Best for:**
- Simple predicates
- Tail recursion (compiled to `while` loops)
- Linear recursion (with `@functools.cache`)
- Arithmetic computations

### Generator Mode

```prolog
compile_predicate_to_python(path/2, [mode(generator)], Code).
```

**Best for:**
- Transitive closure (`ancestor/2`, `reachable/2`)
- Graph algorithms
- Recursive joins
- Deep recursion (no stack limit)

## API Reference

### `compile_predicate_to_python(+Predicate, +Options, -PythonCode)`

**Parameters:**
- `Predicate`: Predicate indicator (`name/arity`)
- `Options`: List of options (see below)
- `PythonCode`: Generated Python code (string)

**Options:**
- `mode(Mode)`: Evaluation mode
  - `procedural` (default) - Direct execution
  - `generator` - Fixpoint iteration
- `record_format(Format)`: Input/output format
  - `jsonl` (default) - JSON Lines (newline-delimited)
  - `nul_json` - NUL-delimited JSON

**Example:**
```prolog
compile_predicate_to_python(
    ancestor/2,
    [mode(generator), record_format(jsonl)],
    Code
).
```

## Input/Output Framing

### JSONL Format (Default)

**Input** (stdin):
```json
{"arg0": "john", "arg1": "mary"}
{"arg0": "mary", "arg1": "sue"}
```

**Output** (stdout):
```json
{"arg0": "john", "arg1": "mary"}
{"arg0": "mary", "arg1": "sue"}
{"arg0": "john", "arg1": "sue"}
```

### NUL-Delimited JSON

**Input** (stdin):
```
{"arg0": "a", "arg1": "b"}\0{"arg0": "b", "arg1": "c"}\0
```

**Output** (stdout):
```
{"arg0": "a", "arg1": "b"}\0{"arg0": "b", "arg1": "c"}\0{"arg0": "a", "arg1": "c"}\0
```

## Mode Selection Guide

### When to Use Procedural Mode

✅ **Use procedural for:**
- Factorial, Fibonacci, arithmetic
- Deterministic predicates
- Tail-recursive patterns
- Performance-critical paths
- Shallow recursion depth

```prolog
% Factorial - procedural is perfect
factorial(0, 1).
factorial(N, F) :- 
    N > 0, 
    N1 is N - 1, 
    factorial(N1, F1), 
    F is N * F1.

compile_predicate_to_python(factorial/2, [], Code).
```

**Generates:**
```python
@functools.cache
def _factorial_worker(n):
    if n == 0:
        return 1
    return n * _factorial_worker(n - 1)
```

### When to Use Generator Mode

✅ **Use generator for:**
- Transitive closure
- Graph reachability
- Unknown recursion depth
- Recursive joins
- Datalog-style queries

```prolog
% Transitive closure - generator handles any depth
edge(a, b). edge(b, c). edge(c, d).

path(X, Y) :- edge(X, Y).
path(X, Z) :- edge(X, Y), path(Y, Z).

compile_predicate_to_python(path/2, [mode(generator)], Code).
```

**Generates:**
```python
def process_stream_generator(records):
    total: Set[FrozenDict] = set()
    delta: Set[FrozenDict] = set(...)
    
    while delta:  # Iterate until fixpoint
        new_delta = set()
        for fact in delta:
            # Apply rules, derive new facts
            ...
        delta = new_delta
```

## Comparison to C# Query Target

Both use **semi-naive evaluation**:

| Feature | Python Generator | C# Query |
|---------|------------------|----------|
| Evaluation | Semi-naive fixpoint | Semi-naive fixpoint |
| Data structure | FrozenDict in sets | Immutable records |
| Termination | Fixpoint detection | Fixpoint detection |
| Input | JSONL/NUL-JSON | JSONL/NUL-JSON |
| Performance | Good | Excellent (compiled) |

**Use Python when:**
- Rapid prototyping
- Simple deployment (no compilation)
- Integration with Python ecosystem

**Use C# when:**
- Maximum performance needed
- Large datasets
- Production systems

## Examples

### Example 1: Procedural Factorial

```prolog
:- use_module('src/unifyweaver/targets/python_target').

% Define predicate
:- assertz((factorial(0, 1))).
:- assertz((factorial(N, F) :- N > 0, N1 is N - 1, factorial(N1, F1), F is N * F1)).

% Compile
?- compile_predicate_to_python(factorial/2, [], Code),
   open('factorial.py', write, S),
   write(S, Code),
   close(S).
```

**Run:**
```bash
echo '{"n": 5}' | python3 factorial.py
# Output: {"n": 5, "result": 120}
```

### Example 2: Generator Transitive Closure

```prolog
% Define graph
:- assertz(edge(a, b)).
:- assertz(edge(b, c)).
:- assertz(edge(c, d)).

% Define rules
:- assertz((path(X, Y) :- edge(X, Y))).
:- assertz((path(X, Z) :- edge(X, Y), path(Y, Z))).

% Compile
?- compile_predicate_to_python(path/2, [mode(generator)], Code),
   open('path.py', write, S),
   write(S, Code),
   close(S).
```

**Run:**
```bash
echo '{"arg0": "a", "arg1": "b"}
{"arg0": "b", "arg1": "c"}
{"arg0": "c", "arg1": "d"}' | python3 path.py

# Output (includes derived facts):
# {"arg0": "a", "arg1": "b"}
# {"arg0": "b", "arg1": "c"}
# {"arg0": "c", "arg1": "d"}
# {"arg0": "a", "arg1": "c"}  ← derived
# {"arg0": "b", "arg1": "d"}  ← derived
# {"arg0": "a", "arg1": "d"}  ← derived
```

## Technical Details

### Procedural Mode

**Recursion Handling:**
- **Tail recursion** → Optimized to `while` loops
- **Linear recursion** → `@functools.cache` memoization
- **Mutual recursion** → Shared memoization across functions

**Supported Arities:**
- Tail recursion: Arity 2 & 3
- Linear recursion: Any arity
- Mutual recursion: Arity 1

### Generator Mode

**Semi-Naive Algorithm:**
1. Initialize `delta` with input facts
2. Apply rules to facts in `delta`
3. Add new facts to `total` and `new_delta`
4. Repeat until fixpoint (`delta` is empty)

**Rule Translation:**
- **Facts** → Emit constants
- **Copy rules** → Pattern match and copy variables
- **Join rules** → Nested loops with join conditions

**Supported:**
- Binary joins (2 goals in body)
- Fact emission
- Variable unification

**Not Yet Supported:**
- N-way joins (3+ goals)
- Negation (`\+`)
- Aggregation

## Troubleshooting

### Issue: "Arguments not sufficiently instantiated"

**Cause:** Generator mode expects ground facts (no variables).

**Solution:** Ensure all input facts are fully instantiated.

### Issue: "Python recursion depth exceeded"

**Cause:** Deep recursion in procedural mode.

**Solution:** Use generator mode instead.

### Issue: "No output from generator mode"

**Cause:** Input format mismatch or no derivable facts.

**Solution:** 
- Check input is valid JSONL
- Verify predicates are defined
- Test with simple facts first

## Performance Tips

1. **Choose the right mode:**
   - Procedural for computation
   - Generator for graphs

2. **Batch input:**
   - Feed all facts at once for generator mode
   - Fixpoint iteration is more efficient with full input

3. **Limit recursion depth:**
   - For procedural mode, keep recursion shallow
   - Or use generator mode for deep graphs

## API Compatibility

The Python target API is stable and compatible with:
- SWI-Prolog 8.0+
- Python 3.7+ (uses type hints)

## Related Documentation

- [Generator Mode Design](../proposals/python_generator_mode.md)
- [Python Target Implementation](../proposals/python_target_architecture.md)
- [C# Query Target](csharp_query_target.md) - Similar semi-naive approach

## Contributing

To extend the Python target:
1. Add tests to `tests/core/test_python_target.pl` or `test_python_generator.pl`
2. Update this guide
3. Run full test suite

For questions or issues, see the main UnifyWeaver documentation.
