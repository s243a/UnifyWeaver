<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM lowered tier — Stage 2 measurement spike (region fusion + minimal snapshot)

Throwaway spike for Stage 2 of
[`../proposals/WAM_RUST_LOWERED_TIER_THROUGHPUT_PLAN.md`](../proposals/WAM_RUST_LOWERED_TIER_THROUGHPUT_PLAN.md)
§5. Decides whether to invest in a full build; the output that matters is the
numbers and the verdict. Nothing here is wired into the default build. All
numbers measured on the build box, `LC_ALL=C.UTF-8`, release + LTO, against the
frozen `resolver.pl` (not modified).

## TL;DR — GO / NO-GO

**GO.** Region fusion + a minimal-locals snapshot beats the interpreter
decisively, and it clears exactly the two obstructions (O1, O2) that made Stage 1
(F11) net-negative.

- **The O2 crux (the whole question): a minimal-locals snapshot is ~15× cheaper
  than `save_regs`** at the real B2 register profile (58 ns vs 904 ns), and the
  gap widens with register-file pressure (up to 46× at the deep end). This is the
  cost that made F11 lose; fusion removes it.
- **Per call, the interpreter spends ~2,990 ns/element on choice-point machinery
  alone** (three full `save_regs` + two `restore_regs` per matching element,
  verified against the emitted bytecode); the **fused native region does the
  entire job — machinery *and* body — in ~65 ns/element (~46×)**, and that number
  is a conservative floor (the interpreter model omits `Allocate`/`Deallocate`,
  head unification, `GetLevel`/`CutTo`, and the re-dispatch, none of which the
  fused region pays either).
- **This one family is not a toy slice: over the real 2,600-case B2 corpus the
  `matching_deps → dep_to_req` region alone is 19.3% of every `save_regs` and
  18.7% of every backtrack** the interpreter executes.
- Against the plan's required baseline (interpreter 21% backtrack + 12%
  `restore_regs`): the region owns ~19% of that machinery, and fusion converts
  its per-element cost from ~2,990 ns to ~65 ns. **Per-call/per-solution cost
  beats the baseline.**

**Expected B2 impact of the full build:** fusing the deterministic family that
this general mechanism covers (~65–70% of B2 dispatches — see §6) should remove
on the order of **15–21% of B2 wall time from choice-point machinery alone**,
before counting the step-dispatch removed by direct native calls. Fusing just the
single `matching_deps → dep_to_req` region is a **~6% B2 floor** from its
machinery share alone.

**Correctness caveat (read this):** the per-call and snapshot numbers come from a
microbenchmark that vendors the runtime's real `Value`/`Args` types and copies
`save_regs`/`restore_regs` **verbatim**, and is parameterised to the **measured**
register distribution and **verified against the emitted bytecode**. It is a
faithful cost model, **not** a full wall-clock A/B of a native region wired into
the differential — reimplementing a byte-faithful native `matching_deps` in the
live runtime was judged too much reimplementation risk for a throwaway (see §7).
The projection is triangulated from three independently-measured real quantities
(per-call machinery cost, region machinery share, and the recorded profile), not
a single end-to-end delta. The fused region's answers were validated identical to
a reference walk on representative inputs; the interpreter baseline itself is the
real binary at **2600/0 divergences vs SWI**.

## The family and why this one

Target: **`matching_deps/4 → dep_to_req/3`** (B2 census: `matching_deps/4` =
15.2% of dispatches). Chosen over `matching_versions/4 → satisfies/2` (11.1%)
because it is the *cleaner* prototype and the *cleaner attribution*:

- `dep_to_req/3` is a pure 2-clause structural rewrite (no arithmetic, no
  recursion), whereas `satisfies/2` pulls in the whole `version_lt`/`segs_lt`
  arithmetic chain.
- `dep_to_req/3` is called **only** from `matching_deps/4` in the term resolver,
  so the region's machinery can be attributed exactly by pc-range. `satisfies/2`
  is shared across many callers, so its cost cannot be cleanly assigned to
  `matching_versions`.

```prolog
matching_deps([], _Name, _Ver, []).
matching_deps([depends(N,V,D,C)|Rest], Name, Ver, Out) :-
    ( N==Name, V==Ver -> dep_to_req(D,C,Req), Out=[Req|Rs] ; Out=Rs ),
    matching_deps(Rest, Name, Ver, Rs).
dep_to_req(alternatives(Alts), _C, req(alternatives(Alts), any)) :- !.
dep_to_req(D, C, req(D, C)).
```

Both are deterministic-in-practice: `matching_deps` is a total function on a
proper list (exactly one answer), and `dep_to_req` commits via its first-clause
cut / mutually-exclusive shapes.

## What was prototyped (P1 + P2)

`examples/pkg_resolver/rust/stage2_spike/` — a standalone throwaway crate
(`publish=false`, not in any workspace, not referenced by `build.sh`).

- **`src/value.rs`** — vendored **verbatim** from the runtime, so `Value::clone`
  cost is exactly the runtime's: `Atom`/`Unbound`/`Str`-functor clone a `String`
  (heap alloc); `List`/`Str`-args clone an `Arc` (O(1) atomic bump); numbers are
  trivial.
- **`save_regs` / `restore_regs`** — copied **verbatim** from `state.rs`. These
  two functions are what the spike is about.
- **P2 — minimal snapshot** (`MiniSnap`/`mini_snapshot`): saves only the region's
  own working locals (`A1` list cursor, `A2` Name, `A3` Ver, `A4` output tail) +
  trail/heap marks. Fixed size, independent of how many registers the caller left
  live.
- **P1 — direct native call** (`dep_to_req`): the fused loop calls `dep_to_req`
  as a plain Rust function; it never sets `vm.pc`/`vm.run()`.
- **Correctness oracle**: `reference_matching_deps` (a plain recursion matching
  the Prolog semantics) — the fused region must return an identical list; both
  `dep_to_req` clauses are exercised. Asserted on every run before timing.

The **interpreter path** is modeled faithfully against the emitted bytecode
(`lib.rs`, `matching_deps/4` at pc 2535): `switch_on_term` is a **NoOp** in this
runtime (no first-arg indexing), so every element pays a `TryMeElse` choice point
(`save_regs`), the base clause `[]` head-match fails and backtracks
(`restore_regs`), then the ITE guard pushes **a second** `TryMeElse`
(`L_ite_else_41`, `save_regs` again), and on a match `dep_to_req` is a `Call` that
pushes **a third** `TryMeElse`. Three full `save_regs` per matching element, two
per non-matching. The model omits `Allocate`/`Deallocate`, head unification,
`GetLevel`/`CutTo` and the re-dispatch — so it **understates** interpreter cost.

## The numbers

### Real baseline (the plan's required anchor)

`run_differential_rust.sh`, 2,600-case seeded B2 differential, this box:

| leg | wall time | ratio |
|---|---:|---:|
| SWI-Prolog | 2.65 s | 1.0× |
| Rust WAM (interpreter, committed default) | 23.13 s | **8.7×** |
| divergences | **0 / 2600** | — |

(Matches the plan's "~8.8× SWI" and the F11 report's "~9.0× F11-off".)

### Real `save_regs` profile over the corpus (instrumented, throwaway)

`save_regs` instrumented (env-gated `UW_SNAP_STATS`, reverted — not committed),
run over the exact 2,600-case corpus:

| quantity | value |
|---|---:|
| `save_regs` calls / corpus | 1,188,875 |
| live A+X registers / call — **average** | **17.94** |
| live A+X registers / call — **max** | **29** |
| String clones / call (Atom/Unbound/Str-functor) | 9.57 |
| Arc clones / call (List/Str-args) | 13.17 |
| live-count distribution | ~99.9% in the 9–32 range; **none above 32** |

So the realistic operating point is **~18 live registers**, max 29. (Y-registers
live in stack frames and are not in `save_regs`.) The microbench is parameterised
to this point.

### Region attribution over the corpus (the "not a toy slice" number)

Same instrumented run, machinery attributed to the region by pc-range
(`matching_deps/4` = pc 2535–2583, its only callee `dep_to_req/3` = pc 1191–1210):

| machinery | region count | of total | **share** |
|---|---:|---:|---:|
| `save_regs` calls | 229,678 | 1,188,875 | **19.32%** |
| backtracks | 222,846 | 1,194,358 | **18.66%** |

**One family owns ~19% of the interpreter's entire snapshot + backtrack
machinery.**

### [A] Snapshot cost — the O2 crux

Per snapshot, at the real profile (live ≈ 18) and across register pressure:

| live A+X regs | full `save_regs` | minimal snap | **ratio** |
|---:|---:|---:|---:|
| 14 | 662 ns | 60 ns | 11× |
| **18 (real avg)** | **904 ns** | **58 ns** | **15.5×** |
| 29 (real max) | 1,308 ns | 56 ns | 23× |
| 44 | 1,588 ns | 63 ns | 25× |
| 104 | 2,721 ns | 59 ns | 46× |

The minimal snapshot is ~constant (~58 ns; three String clones + one Arc bump);
`save_regs` scales with the caller's live-register count. **This is the O2
obstruction, priced. F11 lost because it paid two of the full column per shallow
call; the minimal column is what flips it.**

### [B] Per-call machinery — interpreter vs fused region

Whole `depends` list, body work held constant between paths; per-element:

| depends list length | interpreter ns/elem | fused ns/elem | **speedup** |
|---:|---:|---:|---:|
| 4 (short, worst amortisation) | 3,070 | 88 | 35× |
| **16 (representative)** | **2,988** | **65** | **46×** |
| 64 (long) | 3,608 | 58 | 62× |

Breakdown at the representative point: the interpreter's ~2,988 ns/element is
**almost entirely choice-point machinery** (≈2.15 × `save_regs` @904 + 2 ×
`restore_regs` @~466 per element); the fused region's ~65 ns/element is the actual
work (list peel + guard + direct `dep_to_req` + cons) plus the amortised
one-per-region minimal snapshot. Longer lists amortise the snapshot better.

### [C] Nondet resumable choice-point cycle — extensibility

One push + one resume, full vs minimal representation (see §5):

| live regs | full save+restore | minimal save+resume | **ratio** |
|---:|---:|---:|---:|
| **18 (real avg)** | **1,325 ns** | **125 ns** | **10.6×** |
| 29 (real max) | 2,118 ns | 129 ns | 16× |

A resumable choice point carrying a minimal saved-locals set costs ~1/11th of the
interpreter's own full-`save_regs` + `restore_regs` cycle **per re-entry** — i.e.
per solution.

### Comparison against the recorded baseline (plan §5 requirement)

Interpreter profile: **21% backtrack + 12% `restore_regs`** ≈ 33% of B2 is
choice-point machinery. The region owns **18.66% of backtracks** and **19.32% of
`save_regs`**. So the region's machinery ≈ 0.187 × 33% ≈ **6.2% of B2 wall time**,
which fusion replaces with one amortised minimal snapshot (near-zero). Per element
the machinery drops from ~2,990 ns to ~65 ns. **Per-call and per-solution cost
beat the 21%+12% baseline — clearly, not marginally.**

## Projected B2 impact

Two levels, both grounded in measured quantities:

1. **This one family (floor):** region = 18.7% of the 33% machinery envelope ≈
   **~6% of B2** removable from choice-point machinery alone, plus the
   step-dispatch its re-`run()` calls cost (each `matching_deps` element is an
   `execute` re-dispatch; each match is a `dep_to_req` `Call`), which direct
   native calls (P1) remove from the 65% step bucket. Realistic single-family
   B2 gain: **~8–12%**.
2. **The full deterministic class (§6):** ~65–70% of B2 dispatches are reachable
   by this general mechanism. Deterministic walkers are choice-point-dense (2–3
   CPs/element, like `matching_deps`), so they own *at least* their dispatch
   share of the 33% machinery — **~20% of B2 removable from machinery**, plus a
   large share of the 65% step-dispatch converted to direct native calls.

This would move the SWI ratio from ~8.7× materially toward the target for the
first time — the lever the whole plan is chasing (fewer dispatches, cheaper
snapshots), now measured rather than assumed.

## 6. Generalising the eligibility class (what the full build should target)

The mechanism is three orthogonal capabilities: **(P1)** direct native
lowered→lowered calls, **(P2)** minimal-locals snapshots, **(fusion)** compiling a
chain of deterministic-in-practice predicates into one native region entered once.
Cross-referencing the census
([`wam_rust_f11_census_and_stage1.md`](wam_rust_f11_census_and_stage1.md)) and the
taxonomy ([`../proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md`](../proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md),
T1–T11), the deterministic class it covers is broader than one family:

| pattern (taxonomy) | example predicates | B2 share | covered by | note |
|---|---|---:|---|---|
| **Tail-recursive pure accessors** (F11 / T11) | provides_list, conflicts_list, base_list, depends_list, layers_list, packages, excluded_list, alias_list, requested_list, installed_list, key_pkg_rows, names_of | ~12% | **P2** | already the F11 bank; minimal snapshot **flips their net-negative** (they lost only to `save_regs`) |
| **Det body with det callee** (fusion) | matching_deps→dep_to_req (15.2%), matching_versions→satisfies (11.1%), direct_on→dep_mentions (1.9%), no_acc_conflicts→conflicts_in (1.3%), key_dep_rows→dep_to_req, group_keyed→same_key | ~30% | **P1+P2+fusion** | the core Stage-2 target; needs direct native calls to the callee |
| **Non-tail deterministic walkers** (T4 + fusion) | lookup_held (7.2%), long_enough (5.2%), selected_ver (2.6%), scan_base_holds (2.4%), same_key (1.2% B2 / 24.6% B3) | ~18% | **P1+P2** | native fn with host recursion / explicit accumulator; not tail so no F11 loop, but region-fusible |
| **Non-recursive leaves reached via `run()`** (T1/T4) | item_ver (6.3%), provides_sat (2.5%), dep_mentions (1.8%), conflicts_in (1.8%) | ~12% | **P1** | direct native call removes the dispatch + snapshot even without recursion |
| **Genuinely nondet drivers** | pick/7, blocked_from/4, dep_breaks/5 (4.9%), build_tree/4 (1.6% B2 / 21.9% B3) | ~remainder | **nondet extension (§5)** | real backtracking across solutions — not covered by the deterministic mechanism |

**Coverage: the deterministic mechanism reaches ~65–70% of B2 dispatches** (the
first four rows), leaving a ~30% genuinely-nondet remainder. On **B3** the same
mechanism reaches key_dep_rows (16.4%) + dep_to_req (16.4%) + group_keyed (10.9%)
+ key_pkg_rows (8.2%) + same_key (24.6%, as a det compare) ≈ **~76%**, with
build_tree (21.9%) the main nondet holdout. Crucially, **the same
`dep_to_req`/`same_key` callees recur across B2 and B3**, so one set of lowered
callees serves both workloads.

The full build should therefore target the **general deterministic class**, not
one family: a shared eligibility gate ("deterministic-in-practice: mutually
exclusive heads or committed-choice body, every user callee itself
deterministic-in-practice") + the fusion transform that inlines the callee chain
into one region entered with one minimal snapshot.

## 7. Nondet extensibility — one mechanism or two?

**Assessment: one general mechanism can cover both.** The minimal-locals snapshot
(P2) is exactly the representation a nondet resume state needs. mprolog's F3 idea
(a choice point is a saved resume label; a second solution is a *jump*, not a
re-call) maps directly onto the runtime's existing CP-leaving precedents
(`builtin_state`/`resume_builtin`, T9 `fact_table_attempt`):

- Extend `ChoicePoint` with `lowered_state: Option<LoweredResume>` carrying
  `(resume_arm: u16, saved_locals: <minimal set>, clause_index: u16)` — the
  *same* minimal-locals payload P2 already builds for the deterministic region's
  rollback, plus a resume arm and clause index.
- `backtrack()`'s resume branch re-enters the region's state machine at
  `resume_arm` instead of restoring a full register file and re-dispatching.

The deterministic region is then the degenerate case (`resume_arm` = "done",
never re-entered); a nondet predicate is the same machine with ≥1 live resume arm.
So **P1 + P2 + a resume field is a single lowered execution model** spanning both
deterministic region-fusion and nondet resumable predicates — not two mechanisms.

**Rough per-solution measurement ([C] above):** a minimal save+resume cycle is
**125 ns vs 1,325 ns** for the interpreter's full `save_regs`+`restore_regs`
cycle at the real profile — **10.6× cheaper per re-entry**. This directly answers
the plan's hard requirement for 2b ("measure per-solution cost against the 21%
backtrack + 12% `restore_regs` baseline before it lands"): a resumable tier built
on the minimal representation **beats that baseline per solution**, whereas one
built on `save_regs` (the thing 2a/F11 already priced as neutral-to-negative)
would not. The nondet extension is feasible and worth a dedicated round **after**
the deterministic build banks its win; it should reuse the P2 representation
rather than inventing a second one.

## Verdict and recommended scope

**GO.** Build the deterministic region-fusion tier: P1 (direct native
lowered→lowered calls) + P2 (minimal-locals snapshots) + the fusion transform.

Concrete first regions (highest measured share, cleanest callees, shared across
B2/B3):

1. **`matching_deps/4` ⊕ `dep_to_req/3`** → one native region (19.3% of
   `save_regs`; the prototype here).
2. **`matching_versions/4` ⊕ `satisfies/2` ⊕ `version_lt/2`** → one region
   (11.1%); `satisfies`/`version_lt` are deterministic-in-practice tests.
3. **`key_dep_rows/3` ⊕ `dep_to_req/3`** and **`group_keyed/2` ⊕ `same_key/4`**
   → the B3 index-build regions (reuse the same callees).
4. **Re-enable the F11 accessor bank under P2** (minimal snapshot) — it flips from
   the recorded −7% to positive, since it lost only to `save_regs`.

Then, as a separate later round, the **nondet resume extension** (§7) on the same
minimal representation, for the ~30% B2 / ~22% B3 genuinely-nondet drivers.

Keep every gate green (contract 51/51, term 2600/0, store 503/0) with the tier
default-off until its own numbers land, per the plan's prove-then-commit rule.

## Reproduce

```
# real baseline + differential (2600/0):
bash examples/pkg_resolver/rust/build.sh
bash examples/pkg_resolver/rust/run_differential_rust.sh

# the crux numbers (standalone, ~4 s build):
cd examples/pkg_resolver/rust/stage2_spike
cargo build --release
target/release/stage2_spike 14 16 0.15   # live~18, list=16, 15% match (real profile)
target/release/stage2_spike 25 16 0.15   # live~29 (real max)
```

The `save_regs` region-attribution instrumentation (`UW_SNAP_STATS`) was
env-gated and reverted; it is described in §"Real `save_regs` profile" and is not
part of any committed build.
