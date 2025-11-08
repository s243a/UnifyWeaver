# UnifyWeaver Target Overview

This document frames how UnifyWeaver maps logical predicates onto executable environments. It explains the guiding principles behind the target system and positions the Bash, C# code generation, and C# query runtime backends.

## Philosophy
- Single logical front-end: The compiler analyses clauses once, producing a target-neutral description of joins, projections, and constraints.
- Pluggable execution back-ends: Each target consumes that description and emits artefacts appropriate for its ecosystem (shell scripts, C# source, or query plans).
- Security first: Targets are filtered by firewall policy, and their capabilities are explicit so operators can reason about side-effects.
- Progressive enhancement: Simpler targets shipping earlier inform richer runtimes; new engines should reuse existing classifiers and clause transforms instead of diverging.

## Target Roles
- Bash (`target(bash)` and friends)
  Shell scripts remain the baseline. They rely on ubiquitous tooling, make side-effects explicit, and integrate with system pipelines. Recursion support is implemented through memoised loops and specialised templates.
- C# Code Generation (`target(csharp_codegen)`)
  Emits idiomatic C# source that mirrors the Bash streaming semantics. Today it focuses on non-recursive predicates; over time it can absorb more patterns (e.g., tail recursion) where direct translation is tractable.
- C# Query Runtime (`target(csharp_query)`)
  Produces a declarative intermediate representation (IR) consumed by a reusable LINQ-driven engine. Clause bodies turn into relational operators; recursion is handled by a fixpoint driver that iterates until convergence.

## Selecting Targets
Preferences (`preferences.pl`) and runtime options choose a target. Planned behaviour:
- `target(csharp_codegen)` forces direct C# emission.
- `target(csharp_query)` forces IR + engine execution.
- `target(csharp)` acts as a smart facade, preferring `csharp_codegen` where features exist and falling back to `csharp_query` when advanced behaviour (e.g., recursion) is required.
- `target(bash)` continues to reference the existing Bash ecosystem (partitioning, fork, etc.).

## Why Multiple Targets
- Operational diversity: Bash fits quick shell deployment; C# unlocks integration with managed runtimes, type safety, and IDE tooling.
- Experimentation: Query IR lets us evolve execution strategies (semi-naive evaluation, distributed plans) without regenerating source each time.
- Comparative validation: Running the same logical program through multiple targets helps uncover regressions and clarifies semantics.

Sub-documents in this directory dive into each target family and the comparison matrix that helps choose the right backend for a given deployment.
