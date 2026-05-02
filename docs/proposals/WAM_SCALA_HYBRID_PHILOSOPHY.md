# WAM Scala Hybrid Philosophy

## Purpose

This document defines the architectural philosophy for a future hybrid
WAM Scala target.

It is intentionally design-heavy. The expected implementation flow is:

1. settle the architecture here
2. use the specification and implementation plan to constrain later code
3. let faster implementation agents execute against a stable design

The Scala hybrid WAM target should not be treated as “Clojure, but in
Scala syntax.” It should reuse the same successful abstractions where
that helps, but it should also exploit the fact that Scala gives us a
different part of the JVM design space:

- stronger static typing
- better control over data representation
- both functional and imperative runtime styles
- easier packaging of reusable JVM helper layers

## Why Scala Is Worth a Hybrid WAM Target

Scala is useful here for three reasons:

1. it is already a supported non-WAM target in the codebase
2. it sits on the same JVM family as the newer Clojure LMDB/JNI work
3. it can express high-level generated code while still hosting a
   low-level runtime with explicit arrays, stacks, and caches

That mix makes Scala a good candidate for a hybrid WAM target that is:

- more statically constrained than Clojure
- less FFI-hostile than a pure scripting target
- easier to evolve into a library-grade runtime than a benchmark-only
  code generator

## Core Position

### 1. Share the architecture, not the syntax

The Scala hybrid WAM target should share the **architectural layers**
already proven in Clojure, Haskell, and Rust:

1. generated program surface
2. immutable compiled code/context
3. per-query execution state
4. foreign predicate seam
5. external fact/materialization seam
6. optional cache/policy layers

But it should not share implementation style mechanically.

For example:

- Clojure uses maps/vectors and a dynamic runtime core
- Haskell leans hard on immutable structures and pure wrappers
- Scala can deliberately choose mutable arrays and mutable stacks in the
  hot path while keeping compiled code/context immutable

### 2. Scala does not need Haskell’s mutability boundary

This is the most important design choice.

Haskell kept almost all state immutable except cache layers because that
was the right tradeoff for Haskell’s semantics and optimizer story.
Scala does **not** need to follow that boundary.

For Scala, the better split is:

- immutable `WamProgram` / `WamContext`
- mutable per-run `WamState`
- mutable hot-path structures inside `WamState`
- optional caches outside or beside `WamState`, depending on sharing
  requirements

That means Scala should be comfortable with:

- `Array[AnyRef]` registers
- mutable choice-point stacks
- mutable trail buffers
- imperative program-counter stepping
- specialized value representations where useful

The philosophical rule is:

> Keep shared compiled context immutable. Keep per-query execution state
> cheap and mutable.

This is closer to the existing Rust/C#/Go performance posture than to
the Haskell posture.

### 3. Reuse the JVM LMDB seam rather than inventing a second one

The recent Clojure work already produced a usable JVM LMDB seam:

- manifest-backed artifact contract
- JNI bridge over native LMDB
- row-oriented JVM API
- target-local packaging conventions

Scala should consume that seam, not replace it.

That does **not** mean Scala should depend on Clojure-specific code.
It means Scala should depend on the same JVM helper layer that Clojure
already proved:

- `LmdbArtifactReader`
- `LmdbRow`
- cache-policy wrappers
- JNI `liblmdb_artifact_jni.so`

The philosophy here is:

- one JVM LMDB substrate
- multiple JVM-family targets using it
- target-specific adapters on top

### 4. Generated Scala should remain inspectable

The generated Scala project should be understandable by a human reading
the emitted files.

That argues for:

- a small set of runtime classes
- generated instruction tables as ordinary Scala values
- explicit wrapper methods per predicate
- minimal macro or metaprogramming magic

The target should prefer explicit generated code over “clever” code
generation that is difficult to debug.

### 5. Separate mechanism from policy

This lesson has repeated across targets.

The Scala design must keep these separate:

- how to read facts
- whether to cache
- how to choose a backend
- how to report stats

The target must not hard-code “LMDB implies cache mode X” or “artifact
implies representation Y”.

Instead:

- the runtime supports several policies
- the generated project chooses one
- measurement determines the default

## Architectural Layers

The Scala hybrid WAM target should be built around six layers.

### Layer A: Generated wrappers

Generated wrapper methods are the public predicate entry points.

Example:

```scala
def categoryAncestor(a1: WamTerm, a2: WamTerm, a3: WamTerm, a4: WamTerm): Boolean =
  runtime.runPredicate(sharedCode, sharedLabels, categoryAncestorStartPc, Array(a1, a2, a3, a4), foreignHandlers)
```

These wrappers should stay thin. They are not where optimization logic
belongs.

### Layer B: Immutable program context

This layer contains everything that can be shared safely across runs:

- resolved instruction table
- label index map
- predicate dispatch
- foreign handler registry
- optional artifact manifests
- optional intern tables or literal pools

Example sketch:

```scala
final case class WamProgram(
  instructions: Array[Instruction],
  labels: Map[String, Int],
  dispatch: Map[String, PredicateEntry],
  foreignHandlers: Map[String, ForeignHandler]
)
```

This layer should be immutable and reusable.

### Layer C: Mutable execution state

This is the hot path.

Example sketch:

```scala
final class WamState(
  val regs: Array[WamTerm],
  val envStack: IntStack,
  val choiceStack: ChoicePointStack,
  val trail: TrailBuffer,
  var pc: Int,
  var failed: Boolean,
  var halted: Boolean
)
```

This layer should use mutation deliberately. The goal is cheap stepping,
cheap backtracking, and low allocation.

### Layer D: Foreign predicate adapter layer

Scala should support the same broad foreign contract categories as the
newer Clojure runtime:

1. boolean success/failure
2. deterministic binding maps
3. streamed/multi-solution results

Example protocol:

```scala
sealed trait ForeignResult
case object ForeignFail extends ForeignResult
case object ForeignTrue extends ForeignResult
final case class ForeignBindings(bindings: Map[Int, WamTerm]) extends ForeignResult
final case class ForeignSolutions(solutions: Vector[Map[Int, WamTerm]]) extends ForeignResult
```

The important philosophy is that `call-foreign` should be semantically
integrated with WAM backtracking, not bolted on as a side effect.

### Layer E: Fact backend adapter layer

Scala must support multiple fact access shapes without rewriting the WAM
runtime:

- inline literals
- sidecar TSV/EDN/JSON-like inputs
- preprocessed artifacts
- LMDB-backed exact lookup

This adapter layer should expose a narrow capability-oriented interface.

Example:

```scala
trait FactSource[K, V] {
  def lookup(key: K): Vector[V]
  def scan(): Iterator[(K, V)]
}
```

The exact runtime representation can be optimized later, but the design
should start from capabilities, not storage technology.

### Layer F: Cache and stats layer

Caching should sit above raw fact access.

Possible modes:

- `none`
- `memoize`
- `shared`
- `two_level`

And stats should be reported by the same layer, not buried in JNI code.

## Runtime Representation Philosophy

### Terms

A Scala hybrid WAM target should not start with a maximally abstract
term representation.

The initial design should distinguish only what the WAM runtime needs:

- variable/reference
- atom/string
- integer/number
- structure
- list / list-as-structure

Example baseline:

```scala
sealed trait WamTerm
final case class Ref(id: Int) extends WamTerm
final case class Atom(value: String) extends WamTerm
final case class IntTerm(value: Int) extends WamTerm
final case class Struct(functor: String, args: Array[WamTerm]) extends WamTerm
```

This is not the final word. It is the right baseline for correctness and
inspectability.

### Instructions

Instructions should be pre-resolved into a Scala representation before
execution begins.

Bad:

- parse strings during stepping
- store unresolved labels in the hot loop

Good:

```scala
sealed trait Instruction
final case class Call(pc: Int, pred: String, arity: Int) extends Instruction
final case class TryMeElse(targetPc: Int) extends Instruction
final case object Proceed extends Instruction
```

This follows the same core idea already used in the Clojure WAM target:
resolve control-flow metadata once, not on every step.

## Foreign Predicate Philosophy

The Scala target should support the same semantic categories as the
Clojure path, but it does not need the same representation choices.

It should treat foreign predicates as:

- part of normal predicate dispatch
- allowed to backtrack
- allowed to bind outputs
- allowed to consume external fact backends

For example, `category_ancestor/4` should be allowed to produce
streamed solutions from an LMDB-backed parent store without requiring a
second query engine.

## Materialization and Externalization Philosophy

Scala should follow the C# and shared preprocess direction:

- declarations describe the storage intent
- manifests describe the artifact contract
- targets choose adapters against that contract

Scala should not hide materialization decisions inside ad hoc codegen.

Instead:

- declarations say what relation shape is desired
- generator emits metadata and bindings
- runtime consumes that metadata

This keeps Scala aligned with broader cross-target artifact work.

## Example End-State

At a mature stage, a generated Scala hybrid WAM project should be able
to express:

```scala
object GeneratedProgram {
  val program: WamProgram = ...

  def categoryParent(a1: WamTerm, a2: WamTerm): Boolean =
    WamRuntime.run(program, categoryParentStartPc, Array(a1, a2))

  def categoryAncestor(a1: WamTerm, a2: WamTerm, a3: WamTerm, a4: WamTerm): Boolean =
    WamRuntime.run(program, categoryAncestorStartPc, Array(a1, a2, a3, a4))
}
```

with a runtime that:

- uses mutable execution state
- shares immutable compiled code
- optionally consumes LMDB through the shared JVM seam
- supports streamed foreign solutions
- can emit benchmark or debug stats without changing semantics

## Non-Goals

The initial Scala hybrid WAM design should **not** try to solve all of
these at once:

- advanced parallel query execution
- broad Spark/Hadoop integration
- deep Scala macro generation
- higher-kinded effect systems for the runtime
- every possible external source backend

The right philosophy is staged maturity:

1. correct baseline hybrid WAM
2. target-specific runtime quality
3. shared artifact/backend seams
4. measured optimization

## Summary

The Scala hybrid WAM target should be:

- architecturally aligned with Clojure/Haskell/Rust
- operationally closer to Rust/C#/Go in hot-path mutability
- explicitly integrated with the shared JVM LMDB seam
- generated as readable, inspectable Scala code
- policy-driven rather than hard-coded

The most important philosophical decision is this:

> Scala should keep shared compiled context immutable, but it should be
> unapologetically mutable in the per-run WAM execution state.
