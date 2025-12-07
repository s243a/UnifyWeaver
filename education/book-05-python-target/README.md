<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Book 5: Python Target

**Portable Python Code Generation**

*Part of the [UnifyWeaver Education Series](../README.md)*

This book covers compiling Prolog predicates to Python code. Python's ubiquity makes this target ideal for portable data processing, integration with Python ecosystems, and rapid prototyping.

> **Note**: For semantic search, embeddings, and AI integration, see [Book 13: Semantic Search](../book-13-semantic-search/README.md).

## Prerequisites

**Required:**
- [Book 1: Foundations](../book-01-foundations/README.md)

**Recommended:**
- [Book 2: Bash Target](../book-02-bash-target/README.md) - for stream compilation concepts

**Technical:**
- Python 3.8+ installed
- Basic Python knowledge (helpful but not required)

## What You'll Learn

By completing this book, you will be able to:

- Compile Prolog predicates to Python functions
- Generate standalone Python scripts
- Integrate with Python libraries (pandas, requests, etc.)
- Handle streaming data in Python
- Cross-compile for different Python versions

## Chapter Overview (Planned)

### Part 1: Basic Compilation

**Chapter 1: Introduction**
- Why use the Python target?
- Comparison with Bash and Go targets
- When to choose Python

**Chapter 2: Facts and Rules**
- Compiling facts to Python data structures
- Translating rules to functions
- List comprehensions from Prolog patterns

**Chapter 3: Streaming**
- Generator-based streaming
- Memory-efficient processing
- Iterator patterns

### Part 2: Advanced Features

**Chapter 4: Library Integration**
- Using pandas for data frames
- HTTP requests with requests library
- JSON and CSV processing

**Chapter 5: Recursion**
- Tail-call optimization
- Memoization patterns
- Transitive closure

**Chapter 6: Type Hints**
- Generated type annotations
- mypy compatibility
- Runtime type checking

### Part 3: Deployment

**Chapter 7: Packaging**
- Creating Python packages
- Requirements management
- Virtual environments

**Chapter 8: Performance**
- Optimization techniques
- Profiling generated code
- When to use native targets instead

## Content Status

This book is planned. Chapters are not yet written.

**For Python semantic features now**, see:
- [Book 13: Semantic Search](../book-13-semantic-search/README.md) - Graph RAG and embeddings

## Quick Example

```prolog
% Define a filtering predicate
even(N) :- 0 is N mod 2.
filter_even(List, Evens) :- findall(X, (member(X, List), even(X)), Evens).

% Compile to Python
?- compile_to_python(filter_even/2, [], Code).

% Generated Python:
% def filter_even(lst):
%     return [x for x in lst if x % 2 == 0]
```

## What's Next?

After completing Book 5, continue to:
- [Book 6: Go Target](../book-06-go-target/README.md) - Native binaries
- [Book 7: Cross-Target Glue](../book-07-cross-target-glue/README.md) - Multi-language pipelines
- [Book 13: Semantic Search](../book-13-semantic-search/README.md) - AI integration

## License

This educational content is licensed under CC BY 4.0.
Code examples are dual-licensed under MIT OR Apache-2.0.
