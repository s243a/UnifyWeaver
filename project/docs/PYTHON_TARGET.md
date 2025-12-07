# Python Compilation Target

UnifyWeaver provides a powerful Python compilation target that transforms Prolog predicates into standalone, dependency-free (mostly) Python scripts. It supports advanced recursion patterns and high-performance execution modes.

## Compilation

To compile a predicate to Python:

```prolog
:- use_module(src/unifyweaver/targets/python_target).

compile_predicate_to_python(my_pred/2, [
    mode(procedural),       % procedural (default) or generator
    record_format(jsonl)    % jsonl (default) or nul_json
], Code).
```

## Modes

### 1. Procedural Mode (Default)
Translates Prolog rules into Python generator functions (`yield`). This mode is ideal for streaming pipelines and general logic.

- **Mapping:** Prolog `p(X) :- q(X), r(X)` becomes a nested generator loop: `for x in q(): yield from r(x)`.
- **Recursion:**
    - **Tail Recursion:** Automatically optimized to `while` loops for O(1) space.
    - **General Recursion:** Uses memoization (`@functools.cache`) to prevent redundant computation.
    - **Mutual Recursion:** Compiles groups of predicates together with a shared dispatcher.

### 2. Generator Mode (Semi-Naive)
Implements **Semi-Naive Fixpoint Evaluation** (Datalog style).
- Materializes sets of facts (`total`, `delta`).
- Iterates until no new facts are discovered.
- Useful for complex recursive graph queries where termination is guaranteed by set semantics rather than depth limits.

## Integrated Data Sources

The Python target supports **Native Input Sources**, allowing the generated script to read and process data directly without external piping.

### XML Source (lxml)
Reads, flattens, and streams XML data using `lxml.etree.iterparse`. This avoids the overhead of serializing XML to JSONL in a separate process.

**Usage:**
```prolog
compile_predicate_to_python(process_products/1, [
    input_source(xml('data.xml', ['product'])),
    mode(procedural)
], Code).
```

**Generated Python Logic:**
1.  Initializes an `lxml` streaming parser.
2.  Flattens each `<product>` element into a dictionary:
    - Attributes: `@id`, `@name`
    - Text: `text`
    - Children: Mapped by tag name (simple flattening)
3.  Feeds these dictionaries directly into the predicate logic (`process_stream`).

**Requirements:**
- The target machine must have `lxml` installed (`pip install lxml`).

## Standard I/O

By default (if no `input_source` is given), the generated script reads from **stdin** and writes to **stdout**.

- **Input:** JSON Lines (JSONL) or NUL-delimited JSON.
- **Output:** JSON Lines (JSONL) or NUL-delimited JSON.

This allows Python compiled predicates to be composed in standard Unix pipes.

## Semantic Predicates

The Python target supports high-level semantic operations backed by the embedded runtime library (SQLite, ONNX, lxml).

### `semantic_search(Query, TopK, Results)`
Performs vector similarity search against stored embeddings.

```prolog
search_physics(Results) :-
    semantic_search('quantum physics', 10, Results).
```

### `crawler_run(SeedIds, MaxDepth)`
Starts a focused crawl from the given seed IDs (URLs or paths), fetching, flattening, and embedding content.

```prolog
crawl_data(Seeds) :-
    crawler_run(Seeds, 3).
```

### `upsert_object(Id, Type, Data)`
Manually inserts or updates an object in the local SQLite database.

---

## Example: Factorial

```prolog
factorial(0, 1).
factorial(N, F) :- N > 0, N1 is N - 1, factorial(N1, F1), F is N * F1.
```

Compiles to (simplified):

```python
@functools.cache
def _factorial_worker(arg):
    if arg == 0: return 1
    return arg * _factorial_worker(arg - 1)

def _clause_0(v_0):
    # Wrapper that extracts input, calls worker, yields result
    ...
```
