# WAM Scala Hybrid Implementation Plan

## Purpose

This document breaks the Scala hybrid WAM target into staged
implementation phases.

It assumes the philosophy and specification docs are accepted.

The plan is optimized for:

- incremental correctness
- reuse of existing hybrid-WAM lessons
- minimizing speculative implementation

## Overview

The Scala hybrid WAM target should be implemented in nine phases:

1. target scaffold
2. runtime baseline
3. core instruction execution
4. backtracking and choice points
5. foreign predicate integration
6. streamed foreign solutions
7. artifact/fact backend seam
8. LMDB/JVM seam integration
9. benchmark integration and measurement prep

## Phase S1: Target Scaffold

### Goal

Create the minimum Scala hybrid WAM target structure without full
runtime semantics.

### Deliverables

- `src/unifyweaver/targets/wam_scala_target.pl`
- `templates/targets/scala_wam/`
- generator for:
  - `build.sbt`
  - runtime source
  - generated program source

### Tests

- generator test proves file layout exists
- emitted Scala compiles structurally if the toolchain is present

### Notes

Do not pull in LMDB, foreign streaming, or cache logic yet.

## Phase S2: Runtime Baseline

### Goal

Introduce immutable program context and mutable execution state.

### Deliverables

- `WamProgram`
- `WamState`
- baseline term model
- instruction ADT

### Tests

- runtime smoke tests for project creation
- generated code contains shared instruction table and predicate wrappers

### Design constraint

At this phase, choose mutability deliberately:

- immutable compiled context
- mutable per-run state

Do not mirror Haskell’s “mostly immutable except cache” runtime shape.

## Phase S3: Core Instruction Execution

### Goal

Execute a basic subset of WAM instructions correctly.

### Minimum instruction set

- `call`
- `execute`
- `proceed`
- `jump`
- `allocate`
- `deallocate`
- `get_*`
- `put_*`
- `set_*`
- `unify_*`

### Tests

- direct fact lookup
- simple caller predicates
- structure/list round-trips
- basic arithmetic/builtin calls where already standardized

## Phase S4: Choice Points and Backtracking

### Goal

Add full ordinary WAM backtracking behavior.

### Minimum instruction set

- `try-me-else`
- `retry-me-else`
- `trust-me`
- `switch-on-constant`
- cut-related behavior

### Tests

- multi-clause predicates
- failure fallback
- default-fallthrough in `switch-on-constant`
- cut / if-then-else coverage

### Warning

This phase should not be postponed. A hybrid WAM target without robust
backtracking is not “baseline complete”.

## Phase S5: Foreign Predicate Integration

### Goal

Support target-level `call-foreign` for boolean and deterministic
binding-returning handlers.

### Deliverables

- foreign handler registry
- wrapper generation for foreign predicates
- runtime application of binding maps

### Tests

- foreign success/failure
- foreign output binding
- mixed WAM/foreign calls

## Phase S6: Streamed Foreign Solutions

### Goal

Support streamed multi-solution foreign predicates with backtracking.

### Deliverables

- `ForeignSolutions` protocol
- foreign choice-point snapshots
- restoration logic on failure

### Tests

- two-solution foreign stream
- first-solution failure then backtrack to second
- cut interactions

### Reference

Use the Clojure hybrid WAM streamed foreign semantics as the reference
behavior, not necessarily the same representation.

## Phase S7: Fact Backend Seam

### Goal

Introduce a target-level fact backend abstraction without tying the
runtime to one storage technology.

### Deliverables

- `FactSource`-like interface
- inline and sidecar implementations
- declaration/manifest hooks

### Tests

- same relation through inline vs sidecar backend
- identical predicate answers

### Constraint

This layer must remain separate from cache policy.

## Phase S8: LMDB/JVM Seam Integration

### Goal

Consume the shared JVM LMDB seam from Scala rather than inventing a
separate JNI integration.

### Deliverables

- packaging of helper jar and `.so`
- Scala adapter over `LmdbArtifactReader`
- one relation contract proven end to end

### Initial relation

Start with:

- `category_parent/2`

Later:

- `category_ancestor/4`

### Tests

- generated project packages helper artifacts
- LMDB-backed lookup path works
- stats can be queried

### Constraint

Do not implement a Scala-specific native LMDB seam first.

## Phase S9: Benchmark Integration and Measurement Prep

### Goal

Make the Scala hybrid WAM target benchmarkable without burying runtime
decisions in benchmark-only code.

### Deliverables

- benchmark generator for Scala hybrid WAM
- target-level option plumbing for backend/cache choices
- machine-readable stats output

### Tests

- generated benchmark project shape
- correctness parity with existing targets on small scale

### Important boundary

This phase prepares measurement. It does not decide policy defaults.

## Cross-Phase Reuse Strategy

### Reuse from Clojure

Reuse conceptually:

- hybrid WAM project shape
- instruction-table + wrapper generation
- foreign predicate semantics
- JVM LMDB helper seam

Do not copy directly:

- Clojure map/vector runtime representation
- Clojure-specific dynamic state transitions

### Reuse from Haskell

Reuse conceptually:

- artifact/fact layering
- separation of raw reader from cache policy
- staged LMDB progression

Do not copy directly:

- Haskell’s immutability boundary for hot-path runtime state

### Reuse from Rust

Reuse conceptually:

- willingness to optimize mutable execution state
- explicit runtime structures
- directness in hot paths

## Suggested File Plan

### Prolog

- `src/unifyweaver/targets/wam_scala_target.pl`
- optional helper extraction modules later if the target grows

### Templates

- `templates/targets/scala_wam/build.sbt.mustache`
- `templates/targets/scala_wam/runtime.scala.mustache`
- `templates/targets/scala_wam/program.scala.mustache`

### Tests

- `tests/test_wam_scala_generator.pl`
- `tests/test_wam_scala_runtime_smoke.pl`
- benchmark-specific tests later

## Risks

### Risk 1: Over-copying Clojure

If the Scala target copies the Clojure runtime representation too
closely, it will inherit dynamic-runtime costs without gaining Clojure’s
simplicity benefits.

Mitigation:

- keep Scala mutable in hot-path state
- use types and arrays where they help

### Risk 2: Over-engineering early

If the initial runtime tries to solve generic backend abstraction,
parallelism, and benchmarking all at once, the target will stall.

Mitigation:

- phase strictly
- prove one layer at a time

### Risk 3: Scala-specific JNI duplication

If Scala invents a second LMDB/JNI stack, the JVM-family targets will
diverge unnecessarily.

Mitigation:

- require reuse of the shared JVM LMDB seam

## Exit Criteria by Milestone

### Milestone A: baseline hybrid WAM

Complete when phases S1-S4 are done.

### Milestone B: foreign-capable hybrid WAM

Complete when phases S5-S6 are done.

### Milestone C: artifact-aware hybrid WAM

Complete when phases S7-S8 are done.

### Milestone D: benchmarkable hybrid WAM

Complete when phase S9 is done.

## Recommended Immediate Start

The first implementation PR after these docs should do only:

1. `wam_scala_target.pl`
2. template scaffold
3. generator tests

That is enough to establish the target’s shape without prematurely
committing to runtime internals in code.
