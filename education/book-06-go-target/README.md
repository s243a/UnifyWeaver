<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Book 6: Go Target

**Cross-Platform Native Binaries**

*Part of the [UnifyWeaver Education Series](../README.md)*

This book covers how to use UnifyWeaver to compile Prolog predicates into standalone Go executables. The Go target enables high-performance, cross-platform record processing with no runtime dependencies.

## Prerequisites

**Required:**
- [Book 1: Foundations](../book-01-foundations/README.md)

**Recommended:**
- [Book 5: Python Target](../book-05-python-target/README.md) - first book in the Portable Targets section

**Technical:**
- Go 1.16+ installed (for compiling generated code)
- UnifyWeaver with Go target support (v0.5+)

## Learning Path

**1. Introduction** (`01_introduction.md`)
-   Why use the Go target?
-   Architecture overview
-   Comparison with other targets

**2. Basic Compilation** (`02_basic_compilation.md`)
-   Compiling facts to lookup tables
-   Compiling rules for stream processing
-   Running generated binaries

**3. Advanced Features** (`03_advanced_features.md`)
-   Regex matching with `match/2`
-   Capture groups and extraction
-   Constraints and aggregations

**4. JSON Processing** (`04_json_processing.md`)
-   JSON input/output
-   Schema validation
-   Nested field extraction

**5. Semantic Crawling** (`05_semantic_crawling.md`)
-   Semantic Search (`semantic_search/3`)
-   RDF/XML Crawling
-   Embeddings and Vector Storage

## License

This educational content is licensed under CC BY 4.0.
Code examples are dual-licensed under MIT OR Apache-2.0.
