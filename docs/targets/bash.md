# Bash Target Overview (`target(bash)`)

The Bash family of targets is the original execution backend for UnifyWeaver. It compiles predicates into shell scripts that operate over textual streams. This document summarises its architecture, strengths, and limitations to help compare it with the newer C# options.

## Design Highlights
- Template-driven code generation: Prolog templates render Bash functions for each predicate, including fact lookup, stream outputs, and reverse streams.
- Data representation: Tuples are encoded as delimited strings (e.g., `arg1:arg2`). Associative arrays (`declare -A`) provide deduplication when `unique(true)` is in effect.
- Recursion support: Memoised loops and generated helper scripts implement transitive closures, tail recursion, and other advanced patterns. Temporary files or in-memory arrays hold intermediate results.
- Pipeline integration: Generated scripts expose functions (e.g., `ancestor_all`) and support direct execution (`bash predicate.sh`). They can be composed using standard shell piping or `source`.

## Strengths
- Universal tooling: Works on any POSIX-like environment without additional runtime dependencies.
- Transparency: Compiled scripts are human-readable; operators can audit behaviour and make one-off tweaks.
- Side-effect control: Shell-level commands are explicit, enabling firewall policies to forbid risky operations.
- Mature recursion support: The Bash target already handles memoization, advanced recursive patterns, and partitioning/fork variants.

## Limitations
- Text-centric: Lacks static typing; needs careful escaping and quoting.
- Performance ceiling: Shell loops and `sort` invocations can become bottlenecks for large datasets.
- OS coupling: Relies on Bash 4+ features (`declare -A`), GNU coreutils, and Unix semantics; Windows support depends on compatibility layers.
- Limited composability: Integrating with managed runtimes or object streaming requires bridging scripts with additional services.

## Variants
- `target(bash)` — default streaming scripts.
- `target(bash_partitioning)` / `target(bash_partitioning_target)` — parallelised partitioning backend.
- `target(bash_fork)` — fork-based parallel execution, using `src/unifyweaver/core/backends/bash_fork.pl`.

Each variant builds on the same template system, so improvements to clause analysis or dedup logic propagate automatically.

## When to Choose Bash
- Rapid prototyping on machines with standard Unix toolchains.
- Scenarios where shell integration (piping into existing scripts) is a priority.
- Environments that favour auditable, text-based artefacts over managed binaries.

Where richer typing, managed APIs, or advanced optimisation strategies are needed, consider the C# code generation or query runtime targets described in the companion documents.
