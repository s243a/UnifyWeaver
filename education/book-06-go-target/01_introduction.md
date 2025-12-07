<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 1: Introduction to the Go Target

## What is the Go Target?

The Go target is a powerful feature of UnifyWeaver that compiles your Prolog logic into standalone Go programs. Unlike the Bash target, which generates shell scripts that rely on system tools, the Go target produces compiled binaries that are fast, self-contained, and cross-platform.

## Why Go?

### 1. Single Binary Deployment
The generated code compiles into a single executable file. This means you can compile your logic on your development machine and deploy it to any server without needing to install SWI-Prolog, UnifyWeaver, or any other dependencies on the target machine.

### 2. Cross-Platform Compatibility
Go's cross-compilation capabilities allow you to build binaries for Linux, Windows, and macOS from a single source.

```bash
# Build for Windows on Linux
GOOS=windows GOARCH=amd64 go build -o mytool.exe mytool.go
```

### 3. Performance
Go provides excellent performance for string processing and I/O operations. The generated code uses efficient buffering and map-based lookups, making it suitable for processing large datasets.

## Comparison with Other Targets

| Feature | Bash Target | Go Target |
| :--- | :--- | :--- |
| **Runtime** | Requires Bash 4.0+ | None (Standalone Binary) |
| **Performance** | Good (Stream-based) | Excellent (Compiled) |
| **Portability** | Unix-like systems | Linux, Windows, macOS |
| **Dependencies** | System tools (grep, awk) | None |
| **Use Case** | Shell scripting, Pipelines | ETL, High-performance tools |

## Getting Started

To use the Go target, you need:
1.  **UnifyWeaver** installed.
2.  **Go Compiler** installed (check with `go version`).

In the next chapter, we will write our first Prolog predicate and compile it to Go.

---

## Navigation

[📖 Book 6: Go Target](README) | [Next: Chapter 2: Basic Compilation →](02_basic_compilation)
