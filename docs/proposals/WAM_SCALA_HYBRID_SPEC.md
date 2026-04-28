# WAM Scala Hybrid Specification

## Scope

This document specifies the structure of a future hybrid WAM Scala
target.

It assumes:

- the existing non-WAM Scala target remains intact
- the Scala hybrid WAM target is a separate target module
- the target can reuse existing shared lowering logic where appropriate
- the target may reuse the shared JVM LMDB seam established for Clojure

## 1. Target Identity

### Prolog module

The target should live in:

- `src/unifyweaver/targets/wam_scala_target.pl`

### Template root

Templates should live under:

- `templates/targets/scala_wam/`

### Tests

At minimum:

- `tests/test_wam_scala_generator.pl`
- `tests/test_wam_scala_runtime_smoke.pl`
- optionally benchmark-oriented tests later

## 2. Generated Project Layout

The generated Scala hybrid WAM project should use a simple SBT layout.

### Required files

```text
<project>/
  build.sbt
  project/
    build.properties
  src/main/scala/generated/wam_scala/runtime/WamRuntime.scala
  src/main/scala/generated/wam_scala/core/GeneratedProgram.scala
```

### Optional runtime support

When LMDB or other JVM helpers are used:

```text
<project>/
  lib/
    lmdb-artifact-reader.jar
    liblmdb_artifact_jni.so
```

The generated project must not assume ambient installation of helper
artifacts.

## 3. Runtime Model

### 3.1 Shared immutable program

The generated code must create one immutable program value.

Example:

```scala
final case class WamProgram(
  instructions: Array[Instruction],
  labels: Map[String, Int],
  foreignHandlers: Map[String, ForeignHandler],
  dispatch: Map[String, PredicateEntry]
)
```

This structure is shared across runs.

### 3.2 Mutable per-query state

The runtime must use mutable per-query execution state.

Baseline shape:

```scala
final class WamState(
  val regs: Array[WamTerm],
  val envStack: IntStack,
  val choiceStack: ChoicePointStack,
  val trail: TrailBuffer,
  var pc: Int,
  var halted: Boolean,
  var failed: Boolean
)
```

This is a baseline; exact helper types may evolve.

### 3.3 Execution contract

The runtime entry point should look conceptually like:

```scala
object WamRuntime {
  def runPredicate(
    program: WamProgram,
    startPc: Int,
    regs: Array[WamTerm]
  ): Boolean = ???
}
```

This contract is intentionally close to the Clojure target’s generated
wrapper shape, but Scala may use richer static types internally.

## 4. Instruction Representation

Instructions must be pre-resolved before stepping begins.

### 4.1 Required categories

The runtime must support at least:

- `call`
- `execute`
- `call-foreign`
- `try-me-else`
- `retry-me-else`
- `trust-me`
- `jump`
- `proceed`
- `allocate`
- `deallocate`
- term-building and unification ops
- `switch-on-constant`
- arithmetic/builtin call nodes
- cut / if-then-else support

### 4.2 Example representation

```scala
sealed trait Instruction

final case class Call(pred: String, arity: Int) extends Instruction
final case class Execute(pred: String, arity: Int) extends Instruction
final case class CallForeign(pred: String, arity: Int) extends Instruction
final case class TryMeElse(targetPc: Int) extends Instruction
final case class RetryMeElse(targetPc: Int) extends Instruction
final case object TrustMe extends Instruction
final case object Proceed extends Instruction
```

The exact hierarchy may be compressed for performance later, but the
first implementation should optimize for clarity.

## 5. Term Representation

### 5.1 Baseline

The target must support:

- references/variables
- atoms
- integers
- structures
- list encoding

Baseline:

```scala
sealed trait WamTerm
final case class Ref(id: Int) extends WamTerm
final case class Atom(value: String) extends WamTerm
final case class IntTerm(value: Int) extends WamTerm
final case class Struct(functor: String, args: Array[WamTerm]) extends WamTerm
```

### 5.2 Term operations

The runtime must expose:

- dereference
- unify
- read/write structure/list helpers
- trail on variable binding

## 6. Foreign Predicate Contract

### 6.1 Supported result categories

Foreign predicates must support:

1. boolean success/failure
2. deterministic output bindings
3. streamed/multi-solution results

### 6.2 Result protocol

Suggested baseline:

```scala
sealed trait ForeignResult
case object ForeignFail extends ForeignResult
case object ForeignTrue extends ForeignResult
final case class ForeignBindings(bindings: Map[Int, WamTerm]) extends ForeignResult
final case class ForeignSolutions(solutions: Vector[Map[Int, WamTerm]]) extends ForeignResult
```

### 6.3 WAM integration requirements

`call-foreign` must:

- participate in backtracking
- apply bindings to registers/terms correctly
- preserve cut semantics
- support foreign choice-point snapshots for streamed results

This should follow the same semantic contract as the Clojure hybrid WAM
path, even if Scala uses different data structures.

## 7. Choice Points and Backtracking

### 7.1 Ordinary choice points

The runtime must preserve:

- `pc`
- trail length
- environment stack position
- register state needed for restoration
- any additional execution metadata required for correctness

### 7.2 Foreign choice points

Foreign streamed solutions should use a narrower snapshot if possible.

Suggested baseline:

```scala
final case class ForeignChoicePoint(
  resumePc: Int,
  trailLen: Int,
  regsSnapshot: Array[WamTerm],
  remainingSolutions: Vector[Map[Int, WamTerm]]
)
```

The exact snapshot can be refined later, but it must remain separate
from ordinary WAM choice points conceptually.

## 8. Fact Backend Seam

### 8.1 Capability contract

The runtime should define a small capability-oriented fact-source
interface.

Example:

```scala
trait FactSource[K, V] {
  def lookup(key: K): Vector[V]
  def scan(): Iterator[(K, V)]
}
```

This interface is intentionally simple. The initial Scala hybrid WAM
target does not need a full generic query engine backend.

### 8.2 Initial backends

Initial backends should include:

- inline literals
- sidecar file materialization
- LMDB-backed exact lookup through the shared JVM reader seam

### 8.3 LMDB reuse

Scala should reuse the existing JVM helper seam:

- `generated.lmdb.LmdbArtifactReader`
- `generated.lmdb.LmdbRow`
- JNI shared library

It should not define a second JNI protocol.

## 9. Cache Policy Surface

### 9.1 Supported policies

The Scala design should reserve the same basic surface as the current
Clojure LMDB path:

- `none`
- `memoize`
- `shared`
- `two_level`

### 9.2 Placement

These policies live above raw fact access.

They are not part of the LMDB JNI contract.

### 9.3 Stats

The cache layer should expose machine-readable stats:

- local hits
- shared hits
- misses

This should be represented as data, not just stderr logs.

## 10. Scala-Specific Runtime Policy

This is the most important target-specific specification choice.

### 10.1 Shared immutable context

Must be immutable.

### 10.2 Per-query execution state

May and should be mutable.

This is a required design choice, not an implementation accident.

### 10.3 Caches

Caches may be:

- thread-local mutable
- shared mutable with synchronization or lock-free discipline

The initial target should not over-specify the exact shared-cache
mechanism until measurement warrants it.

## 11. Target Option Surface

The Scala hybrid WAM target should eventually support options in these
areas:

### 11.1 Core generation

- namespace / package
- module name
- emitted main wrapper toggle

### 11.2 Foreign lowering

- `foreign_predicates([...])`
- explicit foreign handlers
- foreign lowering enable/disable

### 11.3 Artifact/fact access

- sidecar vs artifact
- preprocess declaration consumption
- LMDB relation declarations

### 11.4 Cache/reporting

- cache mode
- stats mode
- debug logging

## 12. Benchmark Integration Contract

The benchmark path for Scala hybrid WAM should mirror the current
Clojure/Haskell/Rust strategy:

1. benchmark generator emits a generated project
2. project uses target-level runtime features
3. backends are chosen by declarations/options, not hard-coded logic

The benchmark generator should not become the only place where Scala
LMDB or artifact behavior exists.

## 13. Example Target-Level Declarations

Illustrative examples only:

```prolog
write_wam_scala_project(
    [user:category_parent/2, user:category_ancestor/4],
    [ package('generated.wam_scala'),
      wam_scala_lmdb_foreign_relations([
          category_parent/2-'data/generated/category_parent_lmdb',
          category_ancestor/4-'data/generated/category_parent_lmdb'
      ]),
      wam_scala_lmdb_cache_policy(two_level),
      wam_scala_lmdb_ancestor_max_depth(10)
    ],
    ProjectDir).
```

## 14. Non-Requirements for Phase 1

The first Scala hybrid WAM implementation does not need:

- parallel query execution
- Spark or Hadoop integration
- advanced JIT specialization
- full generic materialization planners
- every relation backend type

It needs a correct hybrid WAM baseline with room to grow.

## 15. Success Criteria

The Scala hybrid WAM target reaches an acceptable first milestone when:

1. it can generate a runnable Scala project
2. simple WAM predicate execution works
3. backtracking works
4. `call-foreign` works for boolean and binding-returning handlers
5. streamed foreign solutions work
6. one LMDB-backed relation path is proven through the shared JVM seam
7. benchmark-target integration is possible without special-casing the
   runtime architecture
