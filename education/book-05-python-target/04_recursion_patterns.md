<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 4: Recursion Patterns

UnifyWeaver's Python target includes sophisticated recursion handling that automatically detects and optimizes different recursion patterns. This chapter covers how the compiler handles tail recursion, linear recursion, and mutual recursion.

## Overview of Recursion Handling

The Python target recognizes three main recursion patterns:

| Pattern | Optimization | Supported Arities | Use Case |
|---------|-------------|-------------------|----------|
| **Tail Recursion** | While loops | 2, 3 | Accumulators, iterative processes |
| **Linear Recursion** | Memoization | Any | Single recursive call per clause |
| **Mutual Recursion** | Shared dispatcher | 1 | Predicates calling each other |

## Tail Recursion

Tail recursion occurs when the recursive call is the last operation in the clause. The Python target converts these to `while` loops, achieving O(1) space complexity.

### Pattern Recognition

A predicate is tail-recursive when:
1. The recursive call is in tail position (last goal)
2. There's a base case that doesn't recurse
3. Accumulator variables pass results downward

### Example: Factorial with Accumulator

**Prolog:**
```prolog
factorial(N, F) :- factorial_acc(N, 1, F).

factorial_acc(0, Acc, Acc).
factorial_acc(N, Acc, F) :-
    N > 0,
    N1 is N - 1,
    Acc1 is Acc * N,
    factorial_acc(N1, Acc1, F).
```

**Generated Python:**
```python
def _factorial_acc_worker(n, acc):
    """Tail-recursive factorial converted to while loop."""
    while True:
        # Base case
        if n == 0:
            return acc
        # Recursive case (transformed to iteration)
        if n > 0:
            n1 = n - 1
            acc1 = acc * n
            n, acc = n1, acc1
            continue
        # No match
        return None
```

### Example: Sum with Accumulator

**Prolog:**
```prolog
sum_to(N, S) :- sum_acc(N, 0, S).

sum_acc(0, Acc, Acc).
sum_acc(N, Acc, S) :-
    N > 0,
    N1 is N - 1,
    Acc1 is Acc + N,
    sum_acc(N1, Acc1, S).
```

**Generated Python:**
```python
def _sum_acc_worker(n, acc):
    """Sum 1 to N using tail recursion → while loop."""
    while True:
        if n == 0:
            return acc
        if n > 0:
            n, acc = n - 1, acc + n
            continue
        return None
```

### Benefits of Tail Recursion Optimization

- **O(1) space** - No stack growth
- **Fast execution** - No function call overhead
- **No recursion limit** - While loops don't hit Python's limit

### Supported Arities for Tail Recursion

| Arity | Example | Notes |
|-------|---------|-------|
| 2 | `pred(In, Out)` | Input → Output |
| 3 | `pred(In, Acc, Out)` | Input + Accumulator → Output |

## Linear Recursion

Linear recursion has a single recursive call per clause (not in tail position). The Python target uses `@functools.cache` for memoization.

### Pattern Recognition

A predicate is linearly recursive when:
1. Exactly one recursive call per clause
2. The recursive call is not in tail position
3. Results are combined after the recursive call

### Example: Classic Factorial

**Prolog:**
```prolog
factorial(0, 1).
factorial(N, F) :-
    N > 0,
    N1 is N - 1,
    factorial(N1, F1),
    F is N * F1.
```

**Generated Python:**
```python
import functools

@functools.cache
def _factorial_worker(n):
    """Memoized factorial computation."""
    # Base case
    if n == 0:
        return 1
    # Recursive case with memoization
    if n > 0:
        f1 = _factorial_worker(n - 1)
        if f1 is not None:
            return n * f1
    return None

def _clause_0(record):
    n = record.get('n')
    if n is not None:
        result = _factorial_worker(n)
        if result is not None:
            yield {'n': n, 'result': result}
```

### Example: Fibonacci

**Prolog:**
```prolog
fib(0, 0).
fib(1, 1).
fib(N, F) :-
    N > 1,
    N1 is N - 1,
    N2 is N - 2,
    fib(N1, F1),
    fib(N2, F2),
    F is F1 + F2.
```

**Generated Python:**
```python
@functools.cache
def _fib_worker(n):
    """Memoized Fibonacci - O(n) time with caching."""
    if n == 0:
        return 0
    if n == 1:
        return 1
    if n > 1:
        f1 = _fib_worker(n - 1)
        f2 = _fib_worker(n - 2)
        if f1 is not None and f2 is not None:
            return f1 + f2
    return None
```

### Benefits of Memoization

Without memoization, naive Fibonacci is O(2^n). With `@functools.cache`:
- **O(n) time** - Each value computed once
- **O(n) space** - Cache stores computed values
- **Automatic** - No manual cache management

### When Memoization Helps Most

- Multiple recursive calls (`fib(N-1)` + `fib(N-2)`)
- Overlapping subproblems
- Same inputs called multiple times

## Mutual Recursion

Mutual recursion occurs when two or more predicates call each other. The Python target compiles these together with a shared dispatcher.

### Pattern Recognition

```prolog
% Classic even/odd mutual recursion
is_even(0).
is_even(N) :- N > 0, N1 is N - 1, is_odd(N1).

is_odd(N) :- N > 0, N1 is N - 1, is_even(N1).
```

### Generated Python (Shared Dispatcher)

```python
@functools.cache
def _mutual_dispatch(pred_name, arg):
    """Shared dispatcher for mutually recursive predicates."""
    if pred_name == 'is_even':
        return _is_even_impl(arg)
    elif pred_name == 'is_odd':
        return _is_odd_impl(arg)
    return None

def _is_even_impl(n):
    # Base case
    if n == 0:
        return True
    # Recursive case - calls is_odd
    if n > 0:
        return _mutual_dispatch('is_odd', n - 1)
    return None

def _is_odd_impl(n):
    # No base case (0 is not odd)
    if n > 0:
        return _mutual_dispatch('is_even', n - 1)
    return None

def is_even(n):
    return _mutual_dispatch('is_even', n)

def is_odd(n):
    return _mutual_dispatch('is_odd', n)
```

### Supported Arities for Mutual Recursion

Currently, mutual recursion is supported for **arity 1 predicates only**. This covers common patterns like even/odd, but not multi-argument mutual recursion.

### Compiling Mutual Recursion Groups

The compiler detects strongly connected components (SCCs) in the call graph:

```prolog
% These form an SCC and are compiled together
a(X) :- b(X).
b(X) :- c(X).
c(X) :- a(X).  % Cycle back to a
```

## Choosing the Right Mode

The compiler automatically detects patterns, but you can influence the choice:

### Procedural Mode (Default)

Best for:
- Tail recursion
- Linear recursion with memoization
- Mutual recursion (arity 1)
- Bounded recursion depth

```prolog
compile_predicate_to_python(factorial/2, [], Code).  % Uses procedural
```

### Generator Mode

Best for:
- Unbounded recursion depth
- Transitive closure
- Graph algorithms
- When memoization isn't sufficient

```prolog
compile_predicate_to_python(ancestor/2, [mode(generator)], Code).
```

## Decision Tree

```
Is the predicate recursive?
├── No → Use procedural mode (simple streaming)
└── Yes → What kind of recursion?
    ├── Tail recursive? → Procedural mode (while loop)
    ├── Linear recursive? → Procedural mode (memoization)
    ├── Mutual recursive (arity 1)? → Procedural mode (dispatcher)
    └── Deep/unbounded recursion? → Generator mode (fixpoint)
```

## Examples

### Example 1: List Length (Tail Recursive)

**Prolog:**
```prolog
length_acc([], Acc, Acc).
length_acc([_|T], Acc, Len) :-
    Acc1 is Acc + 1,
    length_acc(T, Acc1, Len).
```

**Python (while loop):**
```python
def _length_acc_worker(lst, acc):
    while True:
        if not lst:  # Empty list
            return acc
        lst, acc = lst[1:], acc + 1
```

### Example 2: Tree Height (Linear Recursive)

**Prolog:**
```prolog
height(nil, 0).
height(node(_, L, R), H) :-
    height(L, HL),
    height(R, HR),
    H is max(HL, HR) + 1.
```

**Python (memoized):**
```python
@functools.cache
def _height_worker(tree):
    if tree is None:
        return 0
    left, right = tree.get('left'), tree.get('right')
    hl = _height_worker(left) if left else 0
    hr = _height_worker(right) if right else 0
    return max(hl, hr) + 1
```

### Example 3: State Machine (Mutual Recursive)

**Prolog:**
```prolog
% Simple state machine: start → middle → end
state_start(0).
state_start(N) :- N > 0, N1 is N - 1, state_middle(N1).

state_middle(0).
state_middle(N) :- N > 0, N1 is N - 1, state_end(N1).

state_end(0).
state_end(N) :- N > 0, N1 is N - 1, state_start(N1).
```

**Python (shared dispatcher):**
```python
@functools.cache
def _state_dispatch(state, n):
    if state == 'start':
        return _state_start_impl(n)
    elif state == 'middle':
        return _state_middle_impl(n)
    elif state == 'end':
        return _state_end_impl(n)
```

## Performance Comparison

| Pattern | Time | Space | Stack Depth |
|---------|------|-------|-------------|
| Tail (while loop) | O(n) | O(1) | 1 |
| Linear (memoized) | O(n)* | O(n) | O(n) |
| Mutual (dispatcher) | O(n)* | O(n) | O(n) |
| Generator (fixpoint) | O(n²)** | O(n) | 1 |

\* With memoization; without it, could be exponential
\** For binary joins; simpler rules are O(n)

## Common Pitfalls

### 1. Recursion Depth Exceeded

**Problem:** Deep linear recursion hits Python's limit (~1000).

**Solution:** Use generator mode or increase limit:
```python
import sys
sys.setrecursionlimit(10000)  # Use with caution
```

### 2. Cache Memory Growth

**Problem:** Memoization cache grows unboundedly.

**Solution:** Clear cache periodically or use `@lru_cache(maxsize=N)`:
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def _worker(n):
    # ... bounded cache
```

### 3. Mutual Recursion with Arity > 1

**Problem:** Currently unsupported.

**Workaround:** Encode multiple arguments as a tuple:
```prolog
% Instead of: foo(X, Y) :- bar(X, Y).
% Use: foo(Args) :- bar(Args).  % where Args = (X, Y)
```

## Summary

The Python target provides sophisticated recursion handling:

- **Tail recursion** → O(1) space via while loops
- **Linear recursion** → Efficient via memoization
- **Mutual recursion** → Shared dispatcher for arity 1
- **Deep recursion** → Generator mode for unbounded depth

The compiler automatically detects patterns and applies optimizations, but understanding these patterns helps you write Prolog that compiles to efficient Python.

---

## Navigation

**←** [Previous: Chapter 3: Generator Mode - Fixpoint Evaluation](03_generator_mode) | [📖 Book 5: Python Target](./) | [Next: Chapter 5: Semantic Predicates →](05_semantic_predicates)
