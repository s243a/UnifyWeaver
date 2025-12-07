<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Book 1: Foundations

**Architecture, Prolog Basics, and the Preference System**

*Part of the [UnifyWeaver Education Series](../README.md)*

This book covers the fundamental concepts every UnifyWeaver user needs to understand before working with any target language. Start here if you're new to UnifyWeaver.

## Prerequisites

- Basic command-line knowledge
- Programming experience (any language)
- SWI-Prolog 8.0+ installed

## What You'll Learn

By completing this book, you will understand:

- Core Prolog concepts (facts, rules, queries, unification)
- UnifyWeaver's principal compiler architecture
- Architecture variants (fixed-point, query engine, generator)
- How targets are selected and configured
- Basic compilation workflow

## Chapter Overview

### Part 1: Core Concepts (Complete)

**[Chapter 1: Introduction](01_introduction.md)**
- What is UnifyWeaver?
- Why declarative-to-imperative compilation?
- Overview of available targets
- Installation and setup

**[Chapter 2: Prolog Fundamentals](02_prolog_fundamentals.md)**
- Facts, rules, and queries
- Unification and pattern matching
- Lists and recursion basics
- The cut operator

**[Chapter 3: UnifyWeaver Architecture](03_unifyweaver_architecture.md)**
- The principal architecture (stream-based targets)
- Compilation pipeline overview
- Pattern classification
- Constraint analysis and template rendering
- Note on architecture variants (fixed-point, query engine, generator)

### Part 2: Configuration (Planned)

**Chapter 4: The Preference System**
- Default behaviors
- User preferences
- Target-specific options
- Configuration files

**Chapter 5: Plugin Architecture**
- Data source plugins
- Target plugins
- Custom extensions
- Plugin lifecycle

## Architecture Variants

This book introduces the **principal architecture** used by stream-based targets. Other books cover variant architectures:

| Architecture | Targets | Covered In |
|--------------|---------|------------|
| **Principal (Stream)** | Bash, AWK, Go, Rust | This book + Books 2, 6, 9 |
| **Fixed-Point** | C# Query Runtime | Book 3: C# Target |
| **Query Engine** | C# with IR | Book 3: C# Target |
| **Generator** | Python | Book 5: Python Target |

The core concepts (classification, constraint analysis) apply across all variants, but execution models differ.

## What's Next?

After completing Book 1, continue to:
- [Book 2: Bash Target](../book-02-bash-target/README.md) - Stream compilation in depth
- [Book 3: C# Target](../book-03-csharp-target/README.md) - Fixed-point and query engine approaches

## License

This educational content is licensed under CC BY 4.0.
Code examples are dual-licensed under MIT OR Apache-2.0.
