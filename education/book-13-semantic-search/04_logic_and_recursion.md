# Chapter 4: Logic and Recursion in Python

While UnifyWeaver is great for AI pipelines, it is also a fully capable **Logic Compiler**. It translates Prolog's recursive logic into efficient Python code, handling patterns that would normally cause stack overflows.

## 4.1 Procedural Mode vs. Generator Mode

The Python target supports two compilation modes:

### Procedural Mode (Default)
*   **Paradigm**: Streaming Generators (`yield`).
*   **Analogy**: Unix Pipes (`|`).
*   **Use Case**: ETL pipelines, filtering, single-pass logic.
*   **Mechanism**: Predicates become Python generator functions. `p(X) :- q(X)` becomes `for x in q(): yield x`.

### Generator Mode
*   **Paradigm**: Semi-Naive Evaluation (Datalog).
*   **Analogy**: Database Fixpoint.
*   **Use Case**: Complex cyclic graphs, transitive closures where "streaming" isn't possible.
*   **Mechanism**: Materializes sets of facts (`total`, `delta`) and loops until no new facts are added.

## 4.2 Tail Recursion Optimization (TRO)

Python does not natively support Tail Recursion Optimization. If you write a deep recursive function in Python, you will hit `RecursionError`.

UnifyWeaver solves this by compiling tail-recursive Prolog rules into **Python `while` loops**.

**Prolog:**
```prolog
count_down(0).
count_down(N) :- N > 0, N1 is N - 1, count_down(N1).
```

**Generated Python (Simplified):**
```python
def count_down(n):
    current = n
    while current > 0:
        # Logic body
        current = current - 1
    return
```

This allows you to iterate millions of times without consuming stack frames.

## 4.3 General Recursion & Memoization

For non-tail recursion (like Fibonacci or Tree traversal), UnifyWeaver uses **Memoization**.

**Prolog:**
```prolog
fib(0, 0).
fib(1, 1).
fib(N, F) :- N > 1, N1 is N-1, N2 is N-2, fib(N1, F1), fib(N2, F2), F is F1+F2.
```

**Generated Python:**
```python
import functools

@functools.cache
def _fib_worker(n):
    if n == 0: return 0
    if n == 1: return 1
    return _fib_worker(n-1) + _fib_worker(n-2)
```

The compiler automatically identifies the "worker" logic and wraps it in `functools.cache` (or a custom dictionary), ensuring $O(N)$ performance instead of exponential $O(2^N)$.

---

## Navigation

**←** [Previous: Chapter 3: The Semantic Data Pipeline](03_semantic_data_pipeline) | [📖 Book 13: Semantic Search](README) | [Next: Chapter 5: The Semantic Playbook →](05_semantic_playbook)
