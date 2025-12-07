<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Book 9: Rust Target

**Memory-Safe High-Performance Compilation**

*Part of the [UnifyWeaver Education Series](../README.md)*

This book covers how to use UnifyWeaver to compile Prolog predicates into safe, high-performance Rust programs. The Rust target combines the safety of Rust with the declarative power of Prolog.

## Prerequisites

**Required:**
- [Book 1: Foundations](../book-01-foundations/README.md)

**Recommended:**
- [Book 6: Go Target](../book-06-go-target/README.md) - similar native compilation concepts

**Technical:**
- Rust and Cargo installed (`rustc`, `cargo`)
- UnifyWeaver with Rust target support

## Learning Path

**1. Introduction** (`01_introduction.md`)
-   Why use the Rust target?
-   Safety and performance benefits
-   Comparison with Go and C# targets

**2. Basic Compilation** (`02_basic_compilation.md`)
-   Compiling facts and rules
-   Single-file compilation (`rustc`)
-   Stream processing

**3. Project Generation** (`03_project_generation.md`)
-   Generating full Cargo projects
-   Dependency management
-   Building release binaries

**4. Advanced Features** (`04_advanced_features.md`)
-   Regex matching
-   JSON I/O with Serde
-   Constraints and aggregations

## License

This educational content is licensed under CC BY 4.0.
Code examples are dual-licensed under MIT OR Apache-2.0.
