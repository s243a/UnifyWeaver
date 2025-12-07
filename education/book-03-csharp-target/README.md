<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Book 3: C# Target Language

**Multi-Target Compilation: From Prolog to C# / .NET**

*Part of the [UnifyWeaver Education Series](../README.md)*

This book covers UnifyWeaver's C# code generation targets, enabling you to compile Prolog predicates to idiomatic C# code and .NET assemblies. **This is where fixed-point approaches are introduced.**

## Prerequisites

Before starting Book 3, you should have:
- ✅ Completed [Book 1: Foundations](../book-01-foundations/README.md)
- ✅ Completed [Book 2: Bash Target](../book-02-bash-target/README.md) (recommended)
- ✅ Understanding of Prolog predicates and UnifyWeaver compilation
- ✅ Basic C# knowledge (classes, LINQ, generics helpful but not required)
- ✅ .NET SDK 6.0+ installed (for compiling generated C# code)

## What You'll Learn

- How to compile Prolog to C# source code
- Two C# target approaches: Stream Target vs Query Runtime
- LINQ-based query execution
- Semi-naive fixpoint evaluation and mutual recursion (v0.1)
- Runtime libraries and IR (Intermediate Representation)
- Cross-platform deployment with .NET
- Testing workflow (SKIP_CSHARP_EXECUTION, build-first dotnet runs)

## C# Targets Overview

UnifyWeaver provides two C# compilation targets:

### 1. **Stream Target** (`csharp_stream_target.pl`)
- Generates standalone C# source files
- Direct translation of predicates to LINQ pipelines
- Best for: Simple, non-recursive predicates
- Output: Compilable .cs files with `Main()` entry points

### 2. **Query Runtime** (`csharp_query_target.pl`)
- Generates Intermediate Representation (IR)
- Uses shared runtime library for execution (semi-naive + mutual fixpoint)
- Best for: Recursive predicates, complex queries
- Output: C# projects with runtime dependencies

## Book Structure

### Chapter 1: Introduction to Multi-Target Compilation
- Why compile to C#?
- When to use C# vs Bash
- Target selection strategies
- Installation and setup

### Chapter 2: C# Stream Target
- Compiling facts to C# arrays
- LINQ-based query translation
- Deduplication with `Distinct()`
- Building and running generated code

### Chapter 3: C# Query Runtime
- IR (Intermediate Representation) concept
- Query engine architecture
- Fixpoint iteration for recursion (semi-naive + mutual SCC groups)
- Plan nodes: `FixpointNode`, `MutualFixpointNode`, `CrossRefNode`
- Testing with `SKIP_CSHARP_EXECUTION` and manual dotnet builds

### Chapter 4: Runtime Libraries and Deployment
- Shared runtime components
- Packaging C# projects
- Cross-platform deployment (.NET 6+)
- Integration with existing .NET applications

### Chapter 5: Semantic Crawling and Vector Search
- Semantic Runtime Architecture
- Embedding with ONNX (`OnnxEmbeddingProvider`)
- Crawling with `PtCrawler`
- Vector similarity search (`PtSearcher`)
- Case Study: Focused Knowledge Base Crawling

## Installation Requirements

### Required
- .NET SDK 6.0 or higher
  ```bash
  # Check installation
  dotnet --version

  # Install on Linux/WSL
  wget https://dot.net/v1/dotnet-install.sh
  bash dotnet-install.sh --channel 6.0
  ```

- SWI-Prolog 8.0+ (from Book 1)

### Optional
- Visual Studio Code with C# extension
- Visual Studio 2022 or Rider (for advanced debugging)

## Quick Start Example

```prolog
% Define a simple predicate
parent(alice, bob).
parent(bob, charlie).

% Compile to C#
?- compile_predicate_to_csharp(parent/2, [], CSharpCode).

% Generated C# (simplified):
namespace UnifyWeaver.Generated {
    public static class Parent {
        static readonly (string, string)[] Facts = {
            ("alice", "bob"),
            ("bob", "charlie")
        };

        public static IEnumerable<(string, string)> Stream()
            => Facts;
    }
}
```

## What's New in v0.1

- Query runtime now handles mutual recursion through `MutualFixpointNode` and `CrossRefNode` plan nodes.
- Distinct semantics align with Bash via per-predicate `HashSet<object[]>` collections.
- Updated documentation walks through semi-naive and mutual fixpoint evaluation.
- Recommended regression workflow: [`C# Query Target Test Plan`](https://github.com/s243a/UnifyWeaver/blob/main/docs/development/testing/v0_1_csharp_test_plan.md) (skip dotnet execution by default, build-first for manual validation).
- Education examples include parity checks between Bash and C# pipelines.

## Comparison: Bash vs C# Targets

| Feature | Bash Target | C# Stream | C# Query Runtime |
|---------|-------------|-----------|------------------|
| **Recursion** | Full support | Limited | Full support |
| **Performance** | Good | Excellent | Very Good |
| **Deployment** | Shell scripts | .NET apps | .NET apps |
| **Memory** | Streaming | In-memory | Hybrid |
| **Debugging** | Shell tools | Visual Studio | Visual Studio |
| **Dependencies** | Bash 4.0+ | .NET SDK | .NET SDK + Runtime lib |

## When to Use Each Target

**Use Bash Target when:**
- Working in Unix/Linux environments
- Building data processing pipelines
- Integrating with shell scripts
- Minimal deployment dependencies needed

**Use C# Stream Target when:**
- Compiling simple, non-recursive predicates
- Need maximum performance
- Want standalone C# source files
- Building .NET applications

**Use C# Query Runtime when:**
- Compiling recursive predicates
- Need advanced query optimization
- Building complex relational queries
- Want runtime flexibility

## Learning Path

1. **Start with Chapter 1** - Understand multi-target compilation concepts
2. **Work through Chapter 2** - Build simple C# programs from Prolog
3. **Advance to Chapter 3** - Master query runtime for complex cases
4. **Complete Chapter 4** - Deploy production-ready .NET applications

## Example Projects

Throughout Book 2, you'll build:
- Simple fact query console app
- Family tree query with recursion
- Data analysis pipeline in C#
- Cross-platform .NET tool

## Additional Resources

- Codex context: `context/codex/UnifyWeaver/`
- C# examples: `context/codex/UnifyWeaver/examples/csharp_query/`
- Target implementations: `context/codex/UnifyWeaver/src/unifyweaver/targets/`
- Documentation: `context/codex/UnifyWeaver/docs/targets/`

## Going Further

After completing Book 3, you'll be able to:
- ✅ Compile Prolog predicates to both Bash and C#
- ✅ Understand and apply fixed-point approaches
- ✅ Choose appropriate target for your use case
- ✅ Deploy UnifyWeaver-generated code in .NET environments
- ✅ Extend the C# runtime for custom requirements

**Continue to Book 4** to learn about:
- Workflow design for AI agents
- Strategic planning and playbooks
- Economic decision-making in pipelines

## License

This educational content is licensed under CC BY 4.0.
Code examples are dual-licensed under MIT OR Apache-2.0.

## Feedback

Found an issue or have suggestions?
- GitHub Issues: https://github.com/s243a/UnifyWeaver/issues
- Discussions: https://github.com/s243a/UnifyWeaver/discussions
