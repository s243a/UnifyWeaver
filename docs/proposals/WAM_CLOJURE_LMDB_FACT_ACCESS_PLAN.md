# WAM Clojure LMDB Fact Access

## Philosophy

The Clojure hybrid WAM target should follow the same progression that
proved useful on the Haskell LMDB path:

1. separate raw LMDB access from higher-level lookup policy
2. make thread-local reader reuse explicit
3. keep memoization policy outside the raw reader
4. only then consider shared cross-thread cache layers

That sequencing matters because the failure modes are different:

- raw LMDB access is about correctness, JNI/JVM packaging, and
  relation-shape contracts
- thread-local reuse is about avoiding repeated open/read overhead
  without introducing shared mutable cursor state
- memoization is a workload policy decision, not a storage primitive
- shared caches introduce contention, invalidation, and memory-budget
  tradeoffs that should not be entangled with the base reader

The recent Haskell work is the clearest reference for this ordering:

- `ac9e0963` split raw LMDB lookup into IO and higher-level wrappers
- `ec77bd98` introduced per-thread dupsort cursor reuse
- `b4866198` added an L1 cache as an explicit second layer
- `1db9dc36` tightened the L1 hot path around capability-indexed state
- `d124e1d1` added a shared L2 and two-level composition

The lesson for Clojure is not “copy Haskell literally.” The lesson is
that reader mechanics and cache policy should remain distinct seams.

For the JVM path, the equivalent low-level invariants are:

- LMDB rows must be copied into owned JVM objects before they cross the
  JNI boundary
- raw LMDB pointers should not become part of the Clojure-level API
- one read transaction / cursor per thread is the safe scaling model
- generated projects should carry their own helper jar and native
  library wiring explicitly rather than assuming ambient classpaths

## Current Status

The Clojure benchmark path has already completed several prerequisite
steps:

1. sidecar externalization for large generated benchmark data
2. grouped artifact mode for denser `arg1`-oriented relation access
3. per-relation storage policy overrides
4. shared preprocess metadata integration
5. a reusable JVM LMDB row API and JNI seam
6. an opt-in LMDB-backed `category_parent` path in the generated
   effective-distance benchmark runner and foreign handlers

What is implemented today:

- `category_parent` may resolve to `lmdb` through:
  - `wam_clojure_benchmark_relation_data_mode/2`
  - `benchmark_relation_data_mode/2`
- generation writes:
  - `category_parent.tsv`
  - `category_parent_lmdb/manifest.json`
  - `lib/lmdb-artifact-reader.jar`
  - `lib/liblmdb_artifact_jni.so`
- generated Clojure `category_parent/2` and `category_ancestor/4`
  handlers can consume `generated.lmdb.LmdbArtifactReader`
- the benchmark launcher can place the helper jar on the Java classpath
  and the JNI library directory on `java.library.path`

This is intentionally narrow:

- only `category_parent` uses LMDB today
- the existing EDN and grouped-TSV paths remain the stable defaults
- there is no reader pooling, L1 cache, or shared cache policy yet

## Specification

### 1. Reader Layers

The Clojure/JVM LMDB path should be divided into these layers:

1. **Artifact contract**
   - relation artifact directory
   - manifest metadata
   - physical LMDB layout
   - exact access contracts

2. **Raw JVM reader seam**
   - open artifact from manifest
   - `lookupArg1`
   - `scan`
   - owned `LmdbRow` results

3. **Thread-local reader reuse seam**
   - one read-only transaction / cursor per thread
   - dupsort-safe lookup reuse
   - no shared cursor mutation

4. **Clojure relation adapter**
   - `category_parent/2` boolean membership
   - `category_ancestor/4` recursive parent traversal
   - benchmark runner parent-group materialization

5. **Optional cache policy**
   - no cache
   - thread-local L1 memoization
   - future shared cache tiers only after separate validation

### 2. Public Policy Surface

The public benchmark surface should remain stable:

- top-level benchmark data modes stay:
  - `inline`
  - `sidecar`
  - `artifact`
  - `auto`

- relation-level overrides may refine the chosen storage mode:
  - `article_category`
  - `category_parent`

The current Clojure-specific extension is:

- `category_parent -> lmdb`

This is intentionally relation-local, not a new top-level benchmark mode.

### 3. Correctness Rules

Any future reader reuse or cache work must preserve:

1. `category_parent/2` returns the same truth value as the sidecar and
   grouped-TSV variants
2. `category_ancestor/4` emits the same streamed solutions as the
   current non-LMDB path
3. the no-argument effective-distance benchmark emits the same digest as
   the non-LMDB generated runner for identical facts
4. the reader seam never exposes raw LMDB pointers to Clojure code
5. JNI-backed access must continue to work with explicit project-local
   runtime wiring, not ambient machine-global configuration

### 4. Threading Model

The expected threading model is:

- one JVM thread owns its own read-only LMDB transaction
- one JVM thread owns its own cursor(s)
- lookup results are copied into owned JVM objects before returning
- thread joins do not justify shared raw-pointer APIs at the Clojure
  level; correctness should not depend on user-managed transaction
  lifetimes

This mirrors the practical Haskell lesson from `ec77bd98`: thread-local
cursor state is the safe unit of reuse.

## Implementation Plan

### Phase C1: Documentation and seam cleanup

Status: **partly done**

Done already:

- shared artifact metadata seam exists
- JVM LMDB row API exists
- Clojure benchmark generator can consume LMDB for `category_parent`

Next:

1. add a dedicated JVM-side reader reuse abstraction instead of using
   direct `LmdbArtifactReader/open` calls inline in generated Clojure
2. describe that seam in the Clojure docs and in the artifact metadata
   guidance

### Phase C2: Thread-local reader reuse

Status: **not started**

Goal:

- avoid repeated open/setup cost on hot lookup paths

Implementation shape:

1. introduce a JVM helper that maintains thread-local reader state
2. keep one read txn / cursor pair per thread
3. expose the same logical API:
   - `lookupArg1`
   - `scan`
4. switch generated Clojure `category_parent/2` and
   `category_ancestor/4` handlers to that helper

Validation:

- Clojure benchmark generator tests
- generated-project predicate execution tests
- no-argument benchmark digest parity

### Phase C3: Optional L1 memoization

Status: **not started**

Goal:

- allow repeated `category_parent` lookups to avoid duplicate JNI and
  LMDB traversal work on overlap-heavy workloads

Important constraint:

- do not fold memoization into the raw reader seam

Implementation shape:

1. add an explicit no-cache vs L1 policy choice
2. prefer thread-local / per-thread memoization first
3. keep policy local to `category_parent`
4. benchmark overlap-heavy vs low-overlap workloads separately

Reference:

- Haskell `b4866198` and `1db9dc36` show why hot-path structure matters
  once memoization exists at all

### Phase C4: Shared cache tiers

Status: **deferred**

Do not start until:

1. raw reader seam is stable
2. thread-local reuse is validated
3. L1 policy is benchmarked and justified

Reference:

- Haskell `d124e1d1` is useful here, but it is guidance for a later
  phase, not the immediate next step for Clojure

### Phase C5: Broader relation coverage

Status: **deferred**

Possible future work:

- `article_category` LMDB-backed exact/grouped access
- runtime selection between grouped TSV and LMDB from preprocess
  metadata
- alignment with Elixir and future JVM-facing targets through the same
  helper jar / artifact contract

## Decision Summary

The next Clojure LMDB work should be:

1. a cleaner reader seam
2. thread-local reader reuse
3. optional L1 memoization only after that
4. shared caches later

That is the clearest lesson from the recent Haskell commits, and it is
the safest way to improve the current Clojure target without mixing up
storage access mechanics and cache policy.
