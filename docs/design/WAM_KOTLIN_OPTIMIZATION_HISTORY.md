<!--
SPDX-License-Identifier: MIT OR Apache-2.0
-->
# WAM-Kotlin optimization history

Log of profiling-driven runtime optimizations for the Kotlin hybrid WAM
target. Mirrors the structure of
[`WAM_WAT_OPTIMIZATION_HISTORY.md`](WAM_WAT_OPTIMIZATION_HISTORY.md):
**what we did** (with rationale) and **what we skipped** (and why).

Complements [`../WAM_KOTLIN_BENCH.md`](../WAM_KOTLIN_BENCH.md) (numbers)
and [`../WAM_PERF_CROSS_TARGET.md`](../WAM_PERF_CROSS_TARGET.md) (fleet
case-study style).

---

## Baseline (BENCH-KOTLIN, 2026-07-14)

In-process `tryRun` timing showed a **reproducible** regression on deep
tail recursion under `emit_mode(functions)`:

| program | speedup (lowered / interpreter, ≈) |
|---|---:|
| append_100 | 0.71–0.80× |
| append_500 | 0.55–0.64× |

Short non-recursive cases were noise-dominated (up to 8× run-to-run) at
2 warmup / 5 batches. Hypothesis: `tryRun` called `snapshotForNative()`
on **every** native/recursive `execute`→`dispatch` hop, deep-copying the
register/heap map while `H<n>` vars accumulate → ≈O(depth²).

---

## KT-DISPATCH-SNAPSHOT-OPT

### Prior art consulted

| Source | What we took |
|---|---|
| `WAM_PERF_CROSS_TARGET.md` (F# `putReg`, PR #2428) | **Method**: profile → find the copy-the-world-per-op antipattern → remove it. Pattern only — F# mutated arrays in place; we skip redundant snapshots. |
| `WAM_RUST_STATE_MANAGEMENT_RETROSPECTIVE.md` | Trail-based undo as the long-term model for option (b); not implemented here. |
| `WAM_WAT_OPTIMIZATION_HISTORY.md` | Doc shape: did / skipped+why; profile-then-fix for dispatch overhead. |
| `wam_go_dispatch_t6_perf.md`, bind-through sweep | Cross-target reminder that dispatch/bind costs are target-specific. |
| `WAM_ELIXIR_STATUS.md` (Y-reg) | Register representation can dominate; here the hot cost was snapshot copy volume, not Y naming. |
| `PROLOG_TARGET_OPTIMIZATION.md` | Semantics-preserving divergence of the compiled/fast path (cited for the idea only). |

### Profile (append_500, functions mode, 80 iters × 5 timed batches)

| config | min_ms | snap_fraction_of_wall | snap_count | native_entries |
|---|---:|---:|---:|---:|
| **BEFORE** `SKIP_RECURSIVE_SNAP=0` | 419.7 | **0.307** | 200 400 | 200 400 |
| **AFTER** skip recursive snap (default) | 270.9 | **0.000** | 400 | 200 400 |

Snapshot was ~31% of timed wall and ran once per native entry (every
recursive hop). After the fix, snaps drop to one per top-level query
(400 = 80 × 5 batches).

### What we did

1. **`nativeDepth` + `skipRecursiveNativeSnapshot` (default true)** in
   `WamRuntime.tryRun`: recursive native hops call the native fn
   **without** `snapshotForNative` and **without** bytecode fallback.
   Top-level still snapshots + falls back (T5 unbound A1, incomplete
   lowering safety net).
2. **Hardened bench**: default 5 warmup / 15 timed batches; report
   **min** and median batch-ms (`Main.kt.mustache` + harness).
3. Optional `WAM_KT_PROFILE=1` / `PROFILE=1` and
   `SKIP_RECURSIVE_SNAP=0` for A/B profiling.

### Hardened bench after fix (min speedup)

| program | speedup_min |
|---|---:|
| append_100 | **1.03×** |
| append_500 | **0.85×** (was ~0.55×) |
| member_100 / 500 | 1.44–1.56× |
| short cases | see bench doc — still variable |

### What we skipped (and why)

| Idea | Why not now |
|---|---|
| **(b) Trail-based undo for T4 `_t4`** | Trail currently stores names only; CP restore replaces the whole register map. Real trail-undo needs old values (interpreter-wide change) or a native-only side trail. |
| **(c) Heap outside snapshotted map** | Larger representation change; correct long-term but out of scope for this card. |
| **Skip T4 `_t4` entirely** | Unsafe between clauses when clause 1 mutates then fails; recursive child failure relies on parent restore, but intra-pred clause choice still needs a restore point. |
| **COW register map** | First write after snap still copies O(heap) per depth when clause 2 mutates — same asymptotics for append. |

### Remaining gap (append_500 ≈0.85×)

Each lowered T4 entry still does `val _t4 = state.snapshotForNative()` for
clause backtracking. Recursive append therefore still copies the growing
`H<n>` map once per depth. **tryRun** no longer doubles that. Follow-up:
**KT-HEAP-SNAPSHOT-OPT-2** (below).

---

## KT-HEAP-SNAPSHOT-OPT-2

### Prior art consulted

| Source | What we took |
|---|---|
| `WAM_PERF_CROSS_TARGET.md` (F# `putReg`, KT-DISPATCH case study) | Profile → remove copy-the-world on the hot path. |
| `WAM_RUST_STATE_MANAGEMENT_RETROSPECTIVE.md` | Trail undo as the long-term model for option (b); not chosen here. |
| `WAM_WAT_OPTIMIZATION_HISTORY.md` | Doc shape: did / skipped+why. |
| KT-DISPATCH-SNAPSHOT-OPT (above) | Residual `_t4` cost; hardened min/median bench. |

### Profile (append_500, functions, 80 iters × 5 timed batches, `WAM_KT_PROFILE=1`)

Counters now live inside `WamState.snapshotForNative` so **T4 `_t4`** is
attributed (KT-DISPATCH’s post-fix “~0% snap” only measured tryRun snaps).

| config | min_ms | snap_fraction_of_wall | snap_count | native_entries | max_register_map_size |
|---|---:|---:|---:|---:|---:|
| **BEFORE** (entry `_t4` every hop) | 305.6 | **0.481** | 200 800 | 200 400 | **508** |
| **AFTER** (peel leading `get_constant`) | 17.0 | **0.086** | 800 | 200 400 | 508 |

`avg_register_map_size_at_snap` ≈257 before/after at the remaining snaps
(top-level tryRun + base-case nil path). The win is **not** shrinking the
heap — it is **not copying it on every cons hop**.

### What we did

1. **Peel leading `get_constant` / `get_nil` / `get_integer` in T4 emit**
   (`wam_kotlin_lowered_emitter.pl`): closed discriminant miss (ground
   mismatch) jumps to later clauses with **no** entry snapshot. Bindable
   cases (`null` / `Var` / exact match) still snapshot before committing
   clause 1. Last remaining clause needs no snapshot.
2. **PROFILE** aggregates all `snapshotForNative` cost + max/avg register
   map size at snap (`Main.kt.mustache` / `WamRuntime.kt.mustache`).
3. Skip restore after the last failed T4 clause (no next alternative).

Append’s recursive cons path is therefore snap-free → append_500
~**30×** vs interpreter (hardened 5/15 min harness).

### Hardened bench after fix (min speedup)

| program | speedup_min |
|---|---:|
| append_100 | **7.21×** (was ~1.03×) |
| append_500 | **30.33×** (was ~0.85×) |
| member_100 / 500 | 1.37–1.40× (unchanged class — first instr is `get_list`) |

### What we skipped (and why)

| Idea | Why not now |
|---|---|
| **(a) Bound/reclaim `H<n>` outside snapshotted map** | Highest leverage for *every* snapshot, but peel already removed the hot-path copy for append. Larger representation change; keep as follow-up if CP / non-peeled T4 snaps dominate. |
| **(b) Trail-based undo for T4** | Trail still stores names only; needs old values or a native side trail. Bigger than peel; defer. |
| **(c) Blind “skip `_t4` on last clause” only** | Entry snapshot still ran before clause 1; peel is the stronger form of (c) for fail-closed discriminants. |
| **Peel `get_structure` / `get_list`** | Not fail-closed on vars (enters write mode); member already wins without it. |

---

## EMIT-KOTLIN-5 (this change)

### Boundary (deterministic-only mid-body `call`)

Inline `if (!dispatch("P/N", state)) return false` takes the **first**
solution only and cannot backtrack into the callee if a later body goal
fails. Lower mid-body `call` **only** when every target is:

1. **self-recursion**, or
2. a **single-clause deterministic** predicate (whose own mid-body calls
   pass the same gate).

Multi-clause callees (e.g. `choice(a). choice(b)` used as
`choice(X), X = b`) **decline**. When unsure → decline. Top-level tryRun
snapshot+fallback backstops a wrong `false`, not a wrong first-only
`true`.

Arithmetic `builtin_call` (`is/2`, compares, `=/2`, `true/0`) lowers by
calling shared `kotlinLoBuiltinCall` → `wamEvalArith` (same helpers as the
interpreter — no re-implementation).

### Stack ceiling (tree / mid-body recursion)

Linear mid-body-call recursion overflows the default JVM stack around
**~750–780** frames (measured). Fib/ack need O(n) frames; conformance
`fib(10)` / `ack(2,3)` are safe. Practical fib depth is usually time-bound
before stack. Prefer decline over wrong answers if a workload would
overflow.

### Hardened bench after landing (min speedup)

| program | speedup_min |
|---|---:|
| fib_15 | **1.85×** |
| ack_23 | **1.78×** |
| append_500 | ~28× (unchanged class) |

### What we skipped (and why)

| Idea | Why not now |
|---|---|
| Mid-body call to multi-clause callees | First-solution ≠ Prolog when later goals reject the first answer. |
| Classic non-tail `reverse` via `append` | `append/3` is multi-clause — declined as a mid-body callee. Acc-reverse already lowers via `execute`. |
| Full trail/CP continuation for nondet call | Out of scope; would be a different card. |

---

## Correctness gates

Differential unit suite + `CONFORMANCE_TARGETS=kotlin,kotlin_functions`
must stay green whenever this seam changes. Top-level snapshot+fallback
preserved deliberately for T5 unbound.
