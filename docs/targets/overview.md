# UnifyWeaver Target Overview

This document frames how UnifyWeaver maps logical predicates onto executable environments. It explains the guiding principles behind the target system and positions the Bash, C# code generation, and C# query runtime backends.

## Philosophy
- Single logical front-end: The compiler analyses clauses once, producing a target-neutral description of joins, projections, and constraints.
- Pluggable execution back-ends: Each target consumes that description and emits artefacts appropriate for its ecosystem (shell scripts, C# source, or query plans).
- Security first: Targets are filtered by firewall policy, and their capabilities are explicit so operators can reason about side-effects.
- Progressive enhancement: Simpler targets shipping earlier inform richer runtimes; new engines should reuse existing classifiers and clause transforms instead of diverging.

## Target Roles

### Shell Targets
- **Bash** (`target(bash)`)
  Shell scripts remain the baseline. They rely on ubiquitous tooling, make side-effects explicit, and integrate with system pipelines. Recursion support is implemented through memoised loops and specialised templates.

### Systems Language Targets
- **Go** (`target(go)`)
  The most feature-complete procedural target. Generates standalone Go binaries with embedded database support (BoltDB), statistical aggregations, window functions (row_number, rank, LAG/LEAD), and comprehensive observability. Ideal for containerized deployments and high-performance data processing.
- **Rust** (`target(rust)`)
  Generates safe, high-performance Rust programs with zero-cost abstractions. Supports statistical aggregations (stddev, median, percentile), collection aggregations, and observability features. Best for memory-constrained and safety-critical applications.

### .NET Targets
- **C# Code Generation** (`target(csharp_codegen)`)
  Emits idiomatic C# source that mirrors the Bash streaming semantics. Today it focuses on non-recursive predicates; over time it can absorb more patterns (e.g., tail recursion) where direct translation is tractable.
- **C# Query Runtime** (`target(csharp_query)`)
  Produces a declarative intermediate representation (IR) consumed by a reusable LINQ-driven engine. Clause bodies turn into relational operators; recursion is handled by a fixpoint driver that iterates until convergence.

### Declarative Targets
- **SQL** (`target(sql)`)
  Generates declarative SQL queries (SELECT, CREATE VIEW) for execution on relational databases. Unlike other targets that emit executable code, SQL output is meant for external database execution. Supports full SQL feature set including JOINs, aggregations, subqueries, window functions, CTEs, recursive CTEs, and set operations.

### Scripting Targets
- **Python** (`target(python)`)
  Generates Python scripts with strong recursion support, ML integration, and pipeline chaining. Ideal for data science workflows and rapid prototyping.
- **Perl** (`target(perl)`)
  Generates Perl subroutines using continuation-passing style (CPS) with callbacks. Supports tail recursion optimization, linear recursion with memoization, aggregations (count, sum, min, max, avg), and JSON output modes. Ideal for Unix pipeline integration and text processing.
- **Ruby** (`target(ruby)`)
  Generates Ruby methods using block-based CPS with `yield`. Supports tail recursion optimization, linear recursion with memoization, aggregations, and JSON output modes. Idiomatic Ruby code integrates naturally with Rails and Ruby applications.

## Selecting Targets
Preferences (`preferences.pl`) and runtime options choose a target. Planned behaviour:
- `target(csharp_codegen)` forces direct C# emission.
- `target(csharp_query)` forces IR + engine execution.
- `target(csharp)` acts as a smart facade, preferring `csharp_codegen` where features exist and falling back to `csharp_query` when advanced behaviour (e.g., recursion) is required.
- `target(bash)` continues to reference the existing Bash ecosystem (partitioning, fork, etc.).
- `target(sql)` generates SQL queries for database execution rather than standalone programs.

## Why Multiple Targets
- Operational diversity: Bash fits quick shell deployment; C# unlocks integration with managed runtimes, type safety, and IDE tooling.
- Experimentation: Query IR lets us evolve execution strategies (semi-naive evaluation, distributed plans) without regenerating source each time.
- Comparative validation: Running the same logical program through multiple targets helps uncover regressions and clarifies semantics.

Sub-documents in this directory dive into each target family and the comparison matrix that helps choose the right backend for a given deployment.
