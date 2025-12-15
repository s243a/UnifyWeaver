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

### 1. Pipeline Mode (Streaming)

Pipeline mode enables streaming JSONL I/O with typed object output, multiple runtime support, and predicate chaining.

**Basic Usage:**
```prolog
compile_predicate_to_python(user_info/2, [
    pipeline_input(true),           % Enable streaming input
    output_format(object),          % Yield typed dicts
    arg_names(['UserId', 'Email']), % Property names for output
    runtime(cpython)                % or ironpython, pypy, jython
], Code).
```

**Pipeline Chaining:**
```prolog
% Chain multiple predicates into a single pipeline
compile_pipeline(
    [parse_user/2, filter_adult/2, format_output/3],
    [runtime(cpython), pipeline_name(user_pipeline)],
    Code
).
```

Generated Python uses efficient generator chaining:
```python
def user_pipeline(input_stream):
    """Chained pipeline: [parse_user, filter_adult, format_output]"""
    yield from format_output(filter_adult(parse_user(input_stream)))
```

**Cross-Runtime Pipelines:**

For workflows mixing Python and C#:
```prolog
compile_pipeline(
    [python:extract/1, csharp:validate/1, python:transform/1],
    [pipeline_name(data_processor), glue_protocol(jsonl)],
    Code
).
```

This generates stage-based orchestration with automatic runtime grouping.

**Enhanced Pipeline Chaining:**

For complex data flow patterns beyond linear pipelines:
```prolog
compile_enhanced_pipeline([
    extract/1,
    filter_by(is_active),           % Filter stage
    fan_out([validate/1, enrich/1]), % Broadcast to parallel stages
    merge,                           % Combine parallel results
    route_by(has_error, [            % Conditional routing
        (true, error_handler/1),
        (false, success/1)
    ]),
    output/1
], [pipeline_name(enhanced_pipeline)], Code).
```

Enhanced stages:
- `fan_out(Stages)` — Broadcast each record to multiple stages
- `merge` — Combine results from parallel fan-out stages
- `route_by(Pred, Routes)` — Route records based on predicate condition
- `filter_by(Pred)` — Filter records by predicate

**See [Enhanced Pipeline Chaining Guide](ENHANCED_PIPELINE_CHAINING.md) for complete documentation.**

**Runtime Selection:**

| Runtime | Use Case |
|---------|----------|
| `cpython` | Standard Python, default choice |
| `ironpython` | .NET integration, C# hosting |
| `pypy` | JIT-optimized for performance |
| `jython` | Java ecosystem integration |
| `auto` | Auto-select based on context |

**Pipeline Options:**

| Option | Values | Description |
|--------|--------|-------------|
| `pipeline_input(Bool)` | `true/false` | Enable streaming input |
| `output_format(Format)` | `object/text` | Yield dicts or strings |
| `arg_names(List)` | `['Name', ...]` | Property names for output |
| `runtime(R)` | `cpython/ironpython/pypy/jython/auto` | Target runtime |
| `glue_protocol(P)` | `jsonl/messagepack` | Serialization format |

### 2. Procedural Mode (Default)
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
