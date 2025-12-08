<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Book 5: Python Target

**Portable Python Code Generation**

*Part of the [UnifyWeaver Education Series](../README.md)*

This book covers compiling Prolog predicates to Python code. Python's ubiquity makes this target ideal for portable data processing, integration with Python ecosystems, and rapid prototyping.

## Prerequisites

**Required:**
- [Book 1: Foundations](../book-01-foundations/README.md)

**Recommended:**
- [Book 2: Bash Target](../book-02-bash-target/README.md) - for stream compilation concepts

**Technical:**
- Python 3.7+ installed
- Basic Python knowledge (helpful but not required)

## What You'll Learn

By completing this book, you will be able to:

- Compile Prolog predicates to Python functions
- Choose between procedural and generator modes
- Handle different recursion patterns (tail, linear, mutual)
- Use semantic predicates for AI-powered features
- Generate streaming, memory-efficient Python code

## Chapters

**1. Introduction** (`01_introduction.md`)
- Why use the Python target?
- Two evaluation modes overview
- Comparison with other targets
- Quick start example

**2. Procedural Mode** (`02_procedural_mode.md`)
- Streaming architecture
- Compiling facts and rules
- Constraints and arithmetic
- Input/output formats (JSONL, NUL-JSON, XML)

**3. Generator Mode** (`03_generator_mode.md`)
- Semi-naive fixpoint evaluation
- FrozenDict data structure
- Binary and N-way joins
- Stratified negation
- Comparison with C# Query Runtime

**4. Recursion Patterns** (`04_recursion_patterns.md`)
- Tail recursion → while loops
- Linear recursion → memoization
- Mutual recursion → shared dispatcher
- Choosing the right mode
- Performance comparison

**5. Semantic Predicates** (`05_semantic_predicates.md`)
- Vector search (`semantic_search/3`)
- Graph RAG (`graph_search/4`)
- Web crawling (`crawler_run/2`)
- Database storage (`upsert_object/3`)
- LLM integration (`llm_ask/3`)

## Key Concepts

### Two Evaluation Modes

| Mode | Best For | Mechanism |
|------|----------|-----------|
| **Procedural** | Arithmetic, shallow recursion | Generator functions |
| **Generator** | Transitive closure, graphs | Fixpoint iteration |

### Recursion Handling

| Pattern | Optimization | Space |
|---------|-------------|-------|
| Tail recursion | While loops | O(1) |
| Linear recursion | Memoization | O(n) |
| Mutual recursion | Shared dispatcher | O(n) |

### Semantic Runtime

The Python target includes an embedded runtime for AI capabilities:
- ONNX embeddings (local, offline)
- SQLite storage
- Vector similarity search
- Graph-based retrieval (RAG)

## Quick Example

```prolog
% Define transitive closure
path(X, Y) :- edge(X, Y).
path(X, Z) :- edge(X, Y), path(Y, Z).

% Compile to Python with generator mode
?- compile_predicate_to_python(path/2, [mode(generator)], Code).
```

```bash
# Run the generated script
echo '{"arg0": "a", "arg1": "b"}
{"arg0": "b", "arg1": "c"}' | python3 path.py

# Output includes derived facts:
# {"arg0": "a", "arg1": "b"}
# {"arg0": "b", "arg1": "c"}
# {"arg0": "a", "arg1": "c"}  ← derived
```

## What's Next?

After completing Book 5, continue to:
- [Book 6: Go Target](../book-06-go-target/README.md) - Native binaries with generator mode
- [Book 7: Cross-Target Glue](../book-07-cross-target-glue/README.md) - Multi-language pipelines
- [Book 13: Semantic Search](../book-13-semantic-search/README.md) - Deep dive into Graph RAG

## License

This educational content is licensed under CC BY 4.0.
Code examples are dual-licensed under MIT OR Apache-2.0.
