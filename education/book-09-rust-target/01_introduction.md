<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 1: Introduction to the Rust Target

## What is the Rust Target?

The Rust target compiles Prolog logic into Rust code. It offers a unique blend of high-level declarative programming (Prolog) and low-level system performance and safety (Rust).

## Why Rust?

### 1. Memory Safety
Rust's ownership model ensures memory safety without a garbage collector. This makes the generated code suitable for high-performance and resource-constrained environments.

### 2. Zero-Cost Abstractions
UnifyWeaver maps Prolog constructs to efficient Rust equivalents (e.g., `HashSet` for facts, iterators for streams), ensuring that the abstraction cost is minimal.

### 3. Ecosystem
The generated code leverages the rich Rust ecosystem, including `serde` for JSON processing and `regex` for pattern matching.

## Comparison with Other Targets

| Feature | Go Target | Rust Target |
| :--- | :--- | :--- |
| **Compilation** | Fast | Slower (Optimizing) |
| **Runtime** | Garbage Collected | Manual/Ownership |
| **Safety** | Memory Safe | Memory & Thread Safe |
| **Binaries** | Single Binary | Single Binary |
| **Use Case** | General Purpose, ETL | High Performance, System Tools |

## Getting Started

To use the Rust target, you need:
1.  **UnifyWeaver** installed.
2.  **Rust Toolchain** installed (install via [rustup.rs](https://rustup.rs/)).

Check your installation:
```bash
rustc --version
cargo --version
```

---

## Navigation

[📖 Book 9: Rust Target](./) | [Next: Chapter 2: Basic Compilation →](02_basic_compilation)
