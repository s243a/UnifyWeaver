# WAM Clojure Lowered Tier Plan

## Purpose

The Clojure target now has three partially-developed strands that need
to be treated as one architecture instead of separate experiments:

1. classic native clause-body lowering inherited from the TypR-led
   shared lowering work
2. hybrid WAM project generation, foreign predicates, and benchmark
   kernel wiring
3. JVM-side artifact and LMDB infrastructure

What it still lacks, compared with Rust, is a real **middle tier**
between:

- full native clause lowering
- and full WAM interpretation

That missing tier is a **lowered WAM emitter**: direct Clojure
functions compiled from a supported subset of WAM instructions, with
the interpreter still available for the rest.

This document defines the philosophy, specification, and implementation
plan for that work. It also records a second issue exposed by the Scala
design work: unlike the Haskell and Rust WAM targets, the current
Clojure WAM runtime still relies heavily on string-based map lookups
instead of an interned atom/functor representation.

The core conclusion is:

- Clojure should adopt **both** TypR-style native clause lowering and
  Rust-style hybrid tier routing
- if the two strategies want different defaults, the defaults should be
  overridable rather than treated as mutually exclusive
- atom/functor interning should be designed into the lowered tier now,
  not bolted on afterward

## Current State

### What Clojure already has

The ordinary Clojure target already uses the shared clause-body
analysis infrastructure and supports the standard native-lowering
shapes:

- multi-clause guard chains
- arithmetic outputs
- assignment outputs
- nested if-then-else

Primary files:

- [src/unifyweaver/targets/clojure_target.pl](../../src/unifyweaver/targets/clojure_target.pl)
- [tests/core/test_clojure_native_lowering.pl](../../tests/core/test_clojure_native_lowering.pl)

The hybrid Clojure WAM target already has:

- shared code/label tables
- pre-resolved label-based control flow
- foreign predicate stubs
- streaming foreign solutions
- LMDB-backed foreign relation seams
- cache-policy and cache-stats seams for LMDB-backed fact access

Primary files:

- [src/unifyweaver/targets/wam_clojure_target.pl](../../src/unifyweaver/targets/wam_clojure_target.pl)
- [templates/targets/clojure_wam/runtime.clj.mustache](../../templates/targets/clojure_wam/runtime.clj.mustache)
- [tests/test_wam_clojure_generator.pl](../../tests/test_wam_clojure_generator.pl)
- [tests/test_wam_clojure_runtime_smoke.pl](../../tests/test_wam_clojure_runtime_smoke.pl)
- [tests/test_wam_clojure_benchmark_generator.pl](../../tests/test_wam_clojure_benchmark_generator.pl)

### What Clojure still lacks

Compared with Rust, Clojure still lacks:

1. a dedicated lowered WAM emitter
2. explicit per-predicate routing across multiple hybrid tiers
3. a hot-loop term representation that avoids string-heavy equality and
   lookup costs

Rust already has all three in some form:

- classic native lowering in [src/unifyweaver/targets/rust_target.pl](../../src/unifyweaver/targets/rust_target.pl)
- hybrid WAM routing in [src/unifyweaver/targets/wam_rust_target.pl](../../src/unifyweaver/targets/wam_rust_target.pl)
- dedicated lowered WAM emitter in [src/unifyweaver/targets/wam_rust_lowered_emitter.pl](../../src/unifyweaver/targets/wam_rust_lowered_emitter.pl)
- atom interning in [templates/targets/rust_wam/state.rs.mustache](../../templates/targets/rust_wam/state.rs.mustache)

## Philosophy

### 1. Clojure should have both lowering families

There are two different lowering families in this project:

1. **TypR-style clause-body native lowering**
   - source-level Prolog clause analysis
   - target-idiomatic `cond` / local expressions / target-native data
   - best for deterministic predicates that fit the shared clause-body
     analysis model

2. **Rust-style hybrid WAM lowered emission**
   - start from WAM
   - lower supported instruction slices into direct host functions
   - keep interpreter fallback for unsupported instructions or
     non-deterministic control flow

These are not substitutes for one another.

The first works best when a predicate can be understood and emitted
cleanly from source-level structure. The second works best when source
level lowering has already failed or would require much more pattern
work, but the compiled WAM body is still regular enough to optimize.

Clojure should therefore support both.

### 2. Defaults may conflict; that is acceptable

Sometimes the two systems will prefer different routes:

- classic native clause lowering might succeed and be the obvious
  default
- a lowered WAM function may still be preferable for consistency with
  hybrid project structure or for easier fallback integration

That is not a reason to collapse the distinction. It is a reason to
make the routing policy explicit and overridable.

The correct rule is:

- preserve a sensible default route
- expose target options that let users override the route when needed

### 3. Lowering and interning are linked

The Scala design work exposed a real Clojure issue: the current hybrid
runtime is still string- and map-heavy in its hot path.

Examples in the current runtime:

- label lookups use string-keyed maps
- registers are string-keyed
- terms and functors remain string-centric
- foreign dispatch keys are string-based

See:

- [templates/targets/clojure_wam/runtime.clj.mustache](../../templates/targets/clojure_wam/runtime.clj.mustache)

By contrast, the Rust WAM runtime already has atom interning:

- `atom_intern: HashMap<String, u32>`
- `atom_deintern: Vec<String>`
- `intern_atom(&mut self, s: &str) -> u32`

See:

- [templates/targets/rust_wam/state.rs.mustache](../../templates/targets/rust_wam/state.rs.mustache)

For Clojure, this matters because a lowered WAM tier only pays off if
the runtime values it manipulates are also cheaper than the current
string-heavy interpreter path.

So the right stance is:

- do not block the first lowered-emitter slice on full interning
- but do design the lowered-emitter tier so atom/functor interning can
  slot into it without rewriting the tier later

### 4. Rust is the hybrid reference; TypR is the clause-body reference

The references should be separated cleanly:

- **TypR** is the reference for extending classic native clause-body
  lowering breadth
- **Rust** is the reference for hybrid WAM tier routing and lowered
  emission

Clojure should take architecture from both, not treat either as the
single source of truth.

This proposal complements rather than replaces
[WAM_TIERED_LOWERING.md](../design/WAM_TIERED_LOWERING.md):

- `WAM_TIERED_LOWERING.md` is the cross-target tiering concept
- this document specializes that idea for the current Clojure gap:
  adding a lowered WAM middle tier and planning for interning

## Specification

### 1. Tier menu for Clojure

The Clojure target should expose a four-tier menu:

1. **Tier A: classic native clause lowering**
   - existing `clojure_target.pl` path
   - source-level Prolog to target-native Clojure

2. **Tier B: lowered WAM function emission**
   - new `wam_clojure_lowered_emitter.pl`
   - deterministic or limited clause-1 WAM slices compiled to direct
     Clojure functions

3. **Tier C: foreign/kernel lowering**
   - existing `call-foreign` and explicit foreign handler path
   - LMDB-backed fact relations remain part of this family

4. **Tier D: full WAM interpretation**
   - existing shared table + runtime interpreter path

This ordering expresses preference, not a rigid implementation order.

### 2. Default routing policy

The initial routing policy should be:

1. if explicit foreign/kernel relation is selected, use Tier C
2. else if classic source-level native lowering succeeds, the predicate
   is deterministic, and the selected output mode is an ordinary
   Clojure target rather than an explicitly hybrid WAM project, use
   Tier A
3. else if lowered WAM emission succeeds, use Tier B
4. else use Tier D

Initial option surface:

- `clojure_lowering_mode(auto|native|lowered_wam|foreign|wam)`
- `clojure_prefer_native(true|false)` when `auto`
- `clojure_enable_lowered_wam(true|false)` when `auto`

Decision heuristic:

- prefer **Tier A** when source-level analysis cleanly captures the
  predicate and the user wants ordinary target-native Clojure
- prefer **Tier B** when source-level native lowering is unavailable or
  would be awkward, but the compiled WAM body is still regular enough
  to optimize directly
- prefer **Tier C** when the predicate is explicitly modeled as a
  foreign/kernel relation
- fall back to **Tier D** for the remainder

The point is not to expose every future strategy immediately. The point
is to avoid baking a single hidden policy into the generator.

### 3. Lowered-emitter scope

The first lowered-emitter slice should stay narrow and mirror the Rust
shape:

- deterministic predicates lower fully
- multi-clause predicates may lower clause 1 only when the remaining
  clauses still live in interpreter-accessible WAM form
- supported instructions should be the common head/body/core control
  set first:
  - `get_*`
  - `put_*`
  - `unify_*`
  - `set_*`
  - `call`
  - `execute`
  - `proceed`
  - `fail`
  - `builtin_call`
  - `call_foreign`
- `try_me_else` / `retry_me_else` / `trust_me` support may begin in the
  same conservative clause-1 style Rust uses

Non-goals for the first slice:

- replacing the full interpreter
- advanced aggregate lowering
- purity-driven parallel routing
- rewriting the LMDB fact-access subsystem

### 4. Interning contract

The first lowered-emitter implementation should define an intern-table
contract even if not all hot paths use it immediately.

Baseline target shape:

```clojure
{:string->id {"true" 0, "fail" 1, "[]" 2, ...}
 :id->string ["true" "fail" "[]" ...]}
```

Phase-1 usage may be limited to:

- atoms
- structure functors
- possibly predicate keys for internal dispatch

The important design rule is:

- lowered-emitter code should be written so an atom can become either a
  string or an interned integer behind a small set of constructors and
  equality helpers

That avoids a second disruptive rewrite later.

### 5. Runtime hot-path targets

The following current string-heavy surfaces should be treated as
future interning candidates:

1. term atoms
2. structure functors
3. possibly labels
4. possibly register names

Priority order:

1. atoms/functors
2. foreign-dispatch predicate keys if still hot after profiling
3. labels only if startup resolution remains measurable
4. registers last, because those may be better solved structurally than
   through interning

In particular, labels are already pre-resolved into PCs at load time in
the current runtime, so they are less urgent than atom equality in the
query loop.

## Implementation Plan

### Phase 1: proposal and routing contract

Status: **this document**

Deliverables:

- define the four-tier menu
- define routing defaults
- define overridable options
- define interning as part of lowered-tier design

### Phase 2: `wam_clojure_lowered_emitter.pl`

Add:

- [src/unifyweaver/targets/wam_clojure_lowered_emitter.pl](../../src/unifyweaver/targets/wam_clojure_lowered_emitter.pl)

Model it on:

- [src/unifyweaver/targets/wam_rust_lowered_emitter.pl](../../src/unifyweaver/targets/wam_rust_lowered_emitter.pl)

Scope:

- parser for WAM text
- lowerability predicate
- direct Clojure function emission for supported deterministic bodies
- conservative clause-1 lowering for selected multi-clause bodies

Tests:

- new `tests/test_wam_clojure_lowered_emitter.pl`
- focused deterministic lowering cases
- clause-1 multi-clause lowering cases
- unsupported-instruction rejection cases

### Phase 3: route lowered tier through `wam_clojure_target.pl`

Update:

- [src/unifyweaver/targets/wam_clojure_target.pl](../../src/unifyweaver/targets/wam_clojure_target.pl)

Add:

- explicit classification step similar in spirit to Rust’s hybrid
  routing
- generation of direct lowered functions alongside existing shared WAM
  tables and foreign handlers

Tests:

- route selection tests
- explicit override tests
- mixed project tests containing Tier B, Tier C, and Tier D predicates

### Phase 4: intern-table scaffolding

Add a small intern-table surface to the Clojure runtime and code
generation without yet forcing the entire runtime onto it.

Potential files:

- [templates/targets/clojure_wam/runtime.clj.mustache](../../templates/targets/clojure_wam/runtime.clj.mustache)
- [src/unifyweaver/targets/wam_clojure_target.pl](../../src/unifyweaver/targets/wam_clojure_target.pl)

Scope:

- generated intern table for atoms/functors
- helper constructors/equality helpers
- no requirement yet to remove every string from runtime state

Success condition:

- the lowered tier can emit intern-aware atoms without forcing the
  interpreter to be fully rewritten in the same PR

### Phase 5: port selected TypR-driven clause-body improvements

After the lowered tier exists, evaluate whether remaining Clojure gaps
are:

- classic clause-body lowering gaps
- or hybrid lowered-tier gaps

If the former, use TypR as the reference and extend
[src/unifyweaver/targets/clojure_target.pl](../../src/unifyweaver/targets/clojure_target.pl),
not the hybrid WAM target.

This keeps the two lowering families conceptually clean.

## Recommended Defaults

The recommended defaults are:

- ordinary standalone Clojure target:
  - prefer classic native clause lowering first
- hybrid Clojure WAM target:
  - prefer foreign/kernel lowering when explicitly configured
  - otherwise prefer lowered WAM tier when source-level native lowering
    is not the selected mode
- interning:
  - off as a hard requirement for the first lowered-tier PR
  - on as a designed-for capability from the first PR onward

If these defaults prove awkward for a specific workflow, expose options
to override them rather than collapsing the architecture.

## Non-Goals

This proposal does **not** recommend doing all of the following at
once:

- full interpreter rewrite around interned integers
- desktop benchmark campaign
- purity-driven intra-query parallelism
- generalized non-benchmark LMDB relation expansion beyond the current
  narrow relation contracts
- merging classic clause-body lowering and hybrid lowered emission into
  one codepath

Those are separate efforts.

## Suggested Early PR Milestones

1. docs only: this proposal
2. scaffold `wam_clojure_lowered_emitter.pl` + unit tests
3. wire lowered tier into `wam_clojure_target.pl`
4. add intern-table scaffolding for atoms/functors
5. only then decide whether more TypR-style native-lowering breadth is
   the highest-value next step

## References

- [native-clause-lowering.md](../design/native-clause-lowering.md)
- [WAM_TIERED_LOWERING.md](../design/WAM_TIERED_LOWERING.md)
- [src/unifyweaver/targets/typr_target.pl](../../src/unifyweaver/targets/typr_target.pl)
- [src/unifyweaver/targets/clojure_target.pl](../../src/unifyweaver/targets/clojure_target.pl)
- [src/unifyweaver/targets/wam_clojure_target.pl](../../src/unifyweaver/targets/wam_clojure_target.pl)
- [src/unifyweaver/targets/rust_target.pl](../../src/unifyweaver/targets/rust_target.pl)
- [src/unifyweaver/targets/wam_rust_target.pl](../../src/unifyweaver/targets/wam_rust_target.pl)
- [src/unifyweaver/targets/wam_rust_lowered_emitter.pl](../../src/unifyweaver/targets/wam_rust_lowered_emitter.pl)
- [templates/targets/rust_wam/state.rs.mustache](../../templates/targets/rust_wam/state.rs.mustache)
- [WAM_SCALA_HYBRID_SPEC.md](./WAM_SCALA_HYBRID_SPEC.md)
