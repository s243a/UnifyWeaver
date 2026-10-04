<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM: backtrack without cloning the choice point (D121)

**Date:** 2026-10-04. **Ledger:** D121. **Author:** Opus.
**What:** cut the Rust WAM target's per-resolve backtracking cost. After D120,
backtracking was the largest single cost in a small resolve. Every backtrack
cloned the whole top `ChoicePoint` and reset all 200 A/X registers, and every
new choice point scanned all 200 registers to collect its snapshot. Output is
byte-identical.
**Result:** about **−6.7 M Ir per warm resolve at every catalog size**: −16.1%
at N=40 (41.68 M → 34.95 M Ir) and −1.6% at N=5000 (414.18 M → 407.44 M).
The warm in-process N=40 resolve drops from 6.37 ms to 5.65 ms (−11%).

## Diagnosis (verified)

The probe and catalogs are the same as in D120
(`docs/reports/wam_rust_yreg_slot_frames.md`): `resolve_layered([p30])` on
`gen_scale_catalog.mjs` catalogs, capped by `scale_to_case.mjs` at N = 40 /
250 / 1000 / 5000. The base is origin/main `1e159e3` (D120 merged). It
reproduces D120's numbers exactly: 41.68 / 57.65 / 113.78 / 414.18 M Ir per
warm resolve.

An N=40 resolve backtracks 1,296 times and pushes about 1,000 choice points.
In the base profile:

- **`backtrack`: 7.93 M Ir inclusive (19.0%).** `WamState::backtrack`
  (emitted by `compile_backtrack_to_rust` in `wam_rust_target.pl`) began with
  `self.choice_points.last().cloned()`. That cloned the entire `ChoicePoint`
  (the `saved_args` Vec and every `Value` in it, the `levels:
  Vec<(String, usize)>`, and `builtin_state`) on every backtrack. Yet the CP
  normally stays on the stack for the next clause, and `levels` is never read
  in `backtrack`. The function's own share was 0.64 M.
- **`restore_regs`: 4.18 M Ir (10.0%), about 3,200 Ir per call.** It wrote
  `Value::Uninit` into all 200 A/X slots (a drop and a store each), then cloned
  the saved values back.
- **`save_regs`, shown as `Vec::from_iter` in `step`: 4.88 M Ir inclusive.**
  Each new choice point scanned all 200 slots, filtered out `Uninit`, and
  collected the rest into a Vec that started empty and grew by reallocation.

## The change

### Fix 1: no `ChoicePoint` clone in `backtrack` (landed)

`src/unifyweaver/targets/wam_rust_target.pl`, `compile_backtrack_to_rust`. The
top CP is now read in place:

```rust
let (next_pc, trail_len, heap_len, saved_cp, saved_cut_barrier, stack, has_builtin) =
    match self.choice_points.last() {
        Some(cp) => (cp.next_pc, cp.trail_len, cp.heap_len, cp.cp, cp.cut_barrier,
                     cp.stack.clone(), cp.builtin_state.is_some()),
        None => break,
    };
// ... unwind trail, restore stack/trail/heap as before ...
self.restore_regs_from_top_cp();          // restores from a borrow of saved_args
// ... cp, cut_barrier, pending_cut_barrier, pending_level as before ...
if has_builtin {
    if let Some(state) = self.choice_points.pop().and_then(|cp| cp.builtin_state) {
        if self.resume_builtin(state) { return true; }
    }
    continue;
}
return true;
```

The new code copies the scalars and bumps the stack `Arc` (the old code also
made one O(1) `Arc` clone, inside the CP clone). It restores registers through
`restore_regs_from_top_cp`, which borrows `self.choice_points` and
`self.regs`/`self.ax_dirty` as separate fields. `builtin_state` is moved out
only on the path that already popped the CP. The control flow is unchanged:

- the `backtrack_floor` loop guard;
- the order of the steps (pc, trail unwind, stack/trail/heap, registers, then
  control state);
- the builtin-resume path, which still pops the CP before `resume_builtin` and
  `continue`s to the next CP when the resume fails.

Nothing between the read and the pop touches `choice_points`
(`unwind_trail_bindings_only` only touches the trail and bindings), so the CP
that gets popped is the one that was read. The rest of the popped CP is dropped
inside the `and_then` closure, before `resume_builtin` runs, just as the old
`self.choice_points.pop();` statement dropped it. So the stack `Arc` refcount
seen by `resume_builtin` is unchanged as well.

This is the only `.cloned()` of a `ChoicePoint` in the Rust target. Every
other path either borrows the CP (`last_mut` for retry, `iter` for `CutTo`) or
pops or truncates it. `lo_clause_snapshot`/`lo_restore_clause` (lowered T4/F11),
the first-solution and `call_goal_value` paths, the dynamic-DB paths and
`par_aggregate` (a whole-`WamState` clone) already used
`save_regs`/`restore_regs`, so they now use the new versions automatically.

### Fix 2: restore only the registers that can be live (landed)

`templates/targets/rust_wam/state.rs.mustache`. `WamState` gains a private
`ax_dirty: AxDirty` field, a `[u64; 4]` bitmask over the A/X window
`regs[0..AX_REGS]` (`AX_REGS = 200`).

**Invariant:** for every `i < 200`, `regs[i] != Uninit` ⇒ bit `i` is set. The
mask may include extra slots, but it never misses a live one.

- **How the invariant is kept.** `regs` changed from `pub` to private, so the
  compiler enforces that every write goes through a method in `state.rs`.
  There are exactly six write sites, and each keeps the mask:
  - `set_reg`, `set_reg_str`, `init_args` and the two `set_heap_or_list`
    register fix-ups set the bit;
  - `reset_query` clears the mask together with the register file;
  - `restore_regs` rebuilds it.

  `WamState::new` starts with all slots `Uninit` and an empty mask.
  `#[derive(Clone)]` copies the mask with the registers (par_aggregate forks).
  The only readers outside the module were three benchmark generators doing
  `vm.regs.get(3)`. They now use the new read-only accessor `vm.regs()`.
- **Why `restore_regs` is exact.** The old code set all of `regs[0..200]` to
  `Uninit`, then wrote the saved values. The new code sets only the masked
  slots to `Uninit`. By the invariant, every unmasked slot is already
  `Uninit`, so all of `regs[0..200]` ends up `Uninit` exactly as before. It
  then writes the same saved values, so the final register file is identical.
  The new mask is exactly the set of restored slots below 200, which preserves
  the invariant. Slots ≥ 200 were never cleared and still are not.

The cost fell from about 3,200 to about 1,290 Ir per restore. What remains is
mostly dropping the stale values and cloning the saved ones. That work is the
actual state change, so it has to stay.

### Fix 3: `save_regs` (partly landed)

`save_regs` now visits only the masked slots, in ascending order, instead of
scanning all 200. By the invariant it returns the same `(index, value)` list in
the same order. It also sizes the Vec up front (`with_capacity(mask popcount)`)
instead of growing it from empty. The function went from about 3.2 M to about
0.9 M Ir per resolve.

**Not done: pooling or recycling the `saved_args` Vec** (for example, reusing
a popped CP's capacity). CPs are discarded in many places (`TrustMe`, cuts,
`CutTo` truncation, the builtin pop, `reset_query`), and a pool would have to
hook all of them. With the Vec now allocated exactly once per CP, the
remaining malloc and free is about 0.15–0.2 M Ir per resolve (under 0.6%). The
gain is not worth the extra surface.

### Tests

`mod d121_backtrack_tests` in the template, compiled only under `cargo test`,
adds four tests:

1. **`mask_matches_old_full_scan_under_random_ops`.** 4,000 deterministic LCG
   steps of register writes, run in parallel on a shadow register file that
   uses the old save/restore verbatim.
   - The writes cover A, X and Y names, including the odd indices `X0` → 99
     and `A150` → 149, `Uninit` writes, `init_args`, saves, restores of earlier
     snapshots, and resets.
   - After every step: the register file must equal the shadow, `save_regs`
     must equal the old full scan, and the mask must cover every live slot.
   - Mutation check: removing the `mark` from `set_reg_str` makes this test
     fail.
2. **`backtrack_restores_in_place_and_keeps_the_choice_point`.** Three
   backtracks into the same CP restore pc, cp, cut_barrier, heap, stack and
   registers. `pending_level` is cleared. The CP stays on the stack with its
   `saved_args`, `levels` and `builtin_state` unchanged.
3. **`builtin_cp_is_popped_and_a_failed_resume_falls_through`.** A builtin CP
   whose resume fails is popped, and `backtrack` continues to the CP below it.
4. **`backtrack_respects_the_floor`.** CPs at or below `backtrack_floor` are
   never resumed. When the only CPs are failing builtin CPs, all of them are
   popped and `backtrack` returns `false`.

The full generated-crate lib suite passes: **240/240** (D120's 236 plus these
4).

## Gates

Run from the repo root with `LANG=C.utf8 LC_ALL=C.utf8` on fresh `build.sh`
builds (term and store) of this tree and of an untouched origin/main `1e159e3`
export.

| gate | result |
| --- | --- |
| `rust/run_differential_rust.sh` | `cases: 2600 / divergences: 0 / crashes: 0` |
| `rust/run_corpus_rust.sh` | `corpus-under-rust: 51/51 matched SWI` |
| `rust_store/run_differential_rust_store.sh` | `cases: 503 / divergences: 0 / crashes: 0` |
| `rust_store/run_corpus_rust_store.sh` | `corpus-under-rust-store: 51/51 matched SWI`; `rust_store corpus IDENTICAL to term corpus (51/51)` |
| byte identity vs base build | `cmp`-identical: term diff `rust.jsonl` (2600 lines), term corpus `rust.jsonl` (51), store diff `rust.jsonl` (503), store corpus `rust_store.jsonl` (51); scale `--bench` stdout identical at N=40/250/1000/5000 |
| generated crate `cargo test --release --lib` | `test result: ok. 240 passed; 0 failed` |
| Rust WAM plunit (`tests/test_wam_rust_*.pl`, `tests/core/*rust*.pl`, 55 files) | same per-file rc as base (27 rc=0, 28 rc≠0) and the same 42 failing test names |
| CI rust conformance (`CONFORMANCE_TARGETS=rust`, `CONFORMANCE_PROGRAMS=member,builtins`) | rc=0, both unsampled and with `CONFORMANCE_SAMPLE=2` |

**Plunit, base vs new:** the failing tests are the same 42 that D120
recorded. They are the pre-existing breakage in the test harnesses' Rust
(lowered-emitter `Value::Atom(String)` against interned `Sym`). The compile
errors in the new tree's logs are identical to base, including counts (316
`E0308`, 16 `String: Borrow<Sym>`, …). None mentions `regs`, `ax_dirty` or a
private field.

**Benchmark generators:** `generate_wam_rust_matrix_benchmark.pl` (also used by
`tests/core/test_wam_lmdb_cross_target_conformance.pl`'s Rust leg) and
`generate_wam_effective_distance_benchmark.pl` now call `vm.regs()`. The matrix
crate already failed `cargo check` on origin/main, with the same 20 `Sym`
errors in its lowered `lib.rs` on base and new. So whether its `main.rs`
compiles cannot be checked end to end today. The edit is the plain swap of a
field read for the accessor that returns the same slice.

## Measurements

The method is the same as D120. Binaries are `--release` with the crate's
default features (mimalloc on via the pkg_resolver `build.pl`). Ir comes from
`valgrind --tool=callgrind` on an uncommitted scratch copy of the shim that
repeats `call_pred` in-process. Each run checks that every repeat gives the
same answer. Ir per warm resolve = (Ir(1+k) − Ir(1)) / k, with k=10 at N=40
and k=4 elsewhere.

### Ir per warm resolve (deterministic)

| N | base (D120) | D121 | Δ | Δ absolute |
| ---: | ---: | ---: | ---: | ---: |
| 40 | 41.68 M | **34.95 M** | **−16.1%** | −6.73 M |
| 250 | 57.65 M | **50.99 M** | −11.6% | −6.66 M |
| 1000 | 113.78 M | **107.06 M** | −5.9% | −6.72 M |
| 5000 | 414.18 M | **407.44 M** | −1.6% | −6.74 M |

As with D120, the saving is flat (~6.7 M Ir) at every N. Backtracking depends
on the closure's search, not on catalog size. Whole-process Ir for one cold
resolve falls from 53.0 M to 46.2 M at N=40.

Where the N=40 saving comes from (per warm resolve):

| site | base | D121 |
| --- | ---: | ---: |
| `backtrack` (inclusive) | 7.93 M (19.0%) | 3.89 M (11.1%) |
| `restore_regs` → `restore_ax_regs` | 4.18 M | 1.70 M |
| `backtrack` self | 0.64 M | 0.29 M |
| CP snapshot (`Vec::from_iter` → `save_regs`) | ~3.2 M self, 4.88 M incl. | ~0.9 M |
| `Vec::clone` (inclusive) | 2.67 M | 1.65 M |

### Wall clock (shared 4-core container, noisy)

Base and new were interleaved, 9 rounds per size. "Cold" is the shim's own
`resolve_ms` from a fresh process. "Warm" is the median of 11 in-process
repeats, then the median over the 9 rounds.

| N | cold base | cold new | warm base | warm new | Δ warm |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 9.89 ms | **7.06 ms** | 6.37 ms | **5.65 ms** | −11% |
| 250 | 9.98 ms | **9.10 ms** | 8.69 ms | **7.94 ms** | −9% |
| 1000 | 18.63 ms | **17.39 ms** | 17.63 ms | **16.13 ms** | −8% |
| 5000 | 70.30 ms | **63.90 ms** | 69.33 ms | **66.78 ms** | −4% |

The cold-process numbers were very noisy in this session. At N=40, cold base
ranged from 7.7 to 33.7 ms, with 3 of 9 rounds above 24 ms, while cold new
ranged from 6.9 to 7.3 ms. The warm numbers are tighter and track the Ir
change: at N=40, warm base was 6.19–6.57 ms and warm new was 5.55–6.11 ms, each
with one outlier. At N=5000 the Ir gain (−1.6%) is within the noise, so that
wall number is only indicative.

## What dominates next (post-D121 callgrind, N=40, 34.95 M Ir/resolve)

| cost | share | notes |
| --- | ---: | --- |
| `execute_builtin` (inclusive) | 13.1% | builtin dispatch tries `execute_arith/io/type/term/ext/meta` in sequence, matching the op by string name |
| `backtrack` (inclusive) | 11.1% | now mostly `restore_ax_regs` (4.9%, dropping stale values and cloning saved ones, which is the actual state change) plus `unwind_trail_bindings_only` (SipHash on `bindings`) |
| per-step instruction clone | ~9% | `run` does `self.fetch().cloned()` every step: `String::clone` 8.6% incl. plus `drop_in_place<Instruction>` 3.9% incl. |
| mimalloc malloc/free | ~15% incl. | spread across the sites in this table |
| `trail_binding` (inclusive) | 9.0% | |
| `get_reg` / `get_reg_raw` | 8.6% incl. | `get_reg_raw` clones the `Value`; `reg_index` parses the name string |
| `bindings` SipHash | 5.3% | `hash_one` + `sip::write`, on `bindings: HashMap<String, Value>` |
| `format!` in `step` | 5.1% | |

The clearest next lever is the per-step `Instruction` clone (borrow it via
`Arc<[Instruction]>` code or index-based dispatch). It is byte-identical by
construction and worth ~9% of a small resolve. After that come builtin dispatch
by pre-resolved id instead of string matching, and moving `bindings` to
FxHash or an id-keyed map (the D112 pattern). Each of these is a few percent.
The N=5000 term path is still bound by `deref_heap` and
`term_compare_derefed`.

## Files

- `src/unifyweaver/targets/wam_rust_target.pl`: `compile_backtrack_to_rust`
  (Fix 1).
- `templates/targets/rust_wam/state.rs.mustache`: `AX_REGS`, `AxDirty`, the
  private `regs` plus `ax_dirty`, the `regs()` accessor, mask upkeep at the six
  write sites, and the new `save_regs`, `restore_regs`, `restore_ax_regs` and
  `restore_regs_from_top_cp` (Fixes 2 and 3), plus `mod d121_backtrack_tests`.
- `examples/benchmark/generate_wam_rust_matrix_benchmark.pl`,
  `examples/benchmark/generate_wam_effective_distance_benchmark.pl`:
  `vm.regs` → `vm.regs()`.

`resolver.pl` and `resolver_store.pl` are unmodified. The regenerated
checked-in crate is not committed.
