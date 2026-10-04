<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM: heap-cell variable spike, measured first (D128)

**Date:** 2026-10-04. **Ledger:** D128.
**What:** A measure-first spike for the planned big-bang rewrite of the Rust
WAM runtime's representation. It profiles the current runtime and sorts the
cost into buckets. It then prototypes, as throwaway code that was never
committed: heap-cell variables with conditional trailing, choice points that
save only the clause's argument registers, and (as an upper-bound probe only)
no register trail. Everything was measured against the same baseline.
**Design that follows from it:** `docs/proposals/wam_rust_heap_cell_rewrite_design.md`.

## Headline

| variant (cumulative) | Ir/warm resolve N=40 | Δ | Ir/warm resolve N=5000 | Δ | warm wall N=40 | warm wall N=5000 | gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base (origin/main `7da1210`, D127) | 23.30 M | — | 388.51 M | — | 4.26 ms | 64.0 ms | 2600/0/0, 51/51 |
| s1: + heap-cell variables, conditional trail | 22.91 M | −1.7% | 389.37 M | **+0.2%** | 4.12 ms | 62.7 ms | 2600/0/0, 51/51, byte-identical |
| s2: + clause CPs save only A1..An | 21.08 M | **−9.5%** | 387.52 M | −0.3% | 3.61 ms | 60.3 ms | 2600/0/0, 51/51, byte-identical, lib 260/260 |
| s3: + no register trail (upper-bound probe, not sound in general) | 19.20 M | **−17.6%** | 385.66 M | −0.7% | **3.21 ms** | 62.8 ms | 2600/0/0, 51/51, byte-identical |

Wall time is the median of 11 in-process repeats, then the median over 15
interleaved rounds (base/s3, then s1/s2). The box is a shared 4-core container,
so N=5000 wall differences are noise. Ir is the primary metric.

Five findings decide the design:

1. **The interpreter runs only ~21.7 K instructions per resolve, and the count
   is the same at every N.** At N=40 that is ~1,075 Ir per instruction. SWI
   runs the whole resolve in 0.26 ms. The fixed cost is per-instruction
   overhead, not the amount of work.
2. **All of the N-dependent cost is in builtins and the native Stage-2
   regions working on `Arc`'d `Value` trees.** At N=5000, `execute_builtin`
   is 55.0% (23 sort-family calls cost 196 M Ir) and `lowered_call` (the
   regions) is 35.5%. Inside those, `deref_heap` materialization is 59% of
   inclusive Ir, `term_compare` 29%, `Arc` drop 21%, malloc 14%, and the
   interner's `"f/N"` functor decomposition 12%. The interpreter mechanics
   (step overhead, trail, registers, choice points) are ~17.6 M Ir at N=5000,
   the same as at N=40.
3. **The named-variable map is no longer a cost.** After D125 it is ~1.5% of
   a small resolve. Replacing it with heap cells saved 0.38 M Ir at N=40. At
   N=5000 it *cost* 0.86 M net, because the extra `Value` variant stopped two
   hot helpers from being inlined (`same_cell` +4.5 M, `deref_var` +3.1 M).
   `deref_var`'s real cost is returning an owned clone (an `Arc` refcount),
   not the lookup.
4. **The register trail does nothing on this workload.** Each resolve pushes
   10,790 register entries and `unwind_trail_to` consumes **0** of them.
   Backtracking restores A/X registers from the choice point and Y registers
   from the stack snapshot, so these entries are only ever dropped. Turning
   the trail off kept all 2,600 differential cases and the 51-case corpus
   byte-identical. It saved 1.87 M Ir per resolve (8.1% of base).
5. **Clause choice points saved ~10 registers where 2.8 were live.** Saving
   only A1..A<arity> at a clause `try_me_else` saved 1.84 M Ir (7.9% of base)
   and stayed byte-identical. `restore_ax_regs` fell from 1.73 M to 0.47 M,
   because fewer registers are dirty after a restore.

## Method

Same probe and method as D120–D127 (see
`docs/reports/wam_rust_register_trail_static_names.md`).

- `export LANG=C.utf8 LC_ALL=C.utf8`, `mkdir -p output/advanced`, run from
  the repo root. `bash examples/pkg_resolver/rust/build.sh` at origin/main
  `7da1210` (4 m 25 s).
- Cases: `node examples/pkg_resolver/store/gen_scale_catalog.mjs <dir>`, then
  `node examples/pkg_resolver/rust/scale_to_case.mjs <dir> N` for N = 40, 250,
  1000 and 5000. The probe is `resolve_layered([p30])`.
- Ir per warm resolve comes from a scratch copy of the shim (never committed)
  that repeats `call_pred` in-process and checks that every answer is equal.
  It is `(Ir(1+k) − Ir(1)) / k` under callgrind, with k=10 at N=40 and k=2 at
  N=5000. The base reproduces D127 exactly: 23.30 M / 388.51 M (D127:
  23.30 M / 388.52 M).
- Per-function attribution: `callgrind_annotate` self and inclusive cost,
  plus the caller and callee trees, each differenced between the 1-rep and
  (1+k)-rep runs.
- Event counts (binds, trail entries, choice-point slot counts) come from a
  separate spike build with a `spike_stats` feature (relaxed atomic counters).
  That build was never used for Ir or wall numbers.
- The spike crate is a scratch copy of the generated crate
  (`examples/pkg_resolver/rust/uw_resolve_wam/`), edited directly and built
  with `cargo build --release` (~4 min per build). The generator and
  templates were not touched.
- Gates for each variant: the term differential (same 2,600 cases and SWI
  oracle output as `run_differential_rust.sh`, compared by `compare_jsonl.mjs`
  and `cmp`'d against the base `rust.jsonl`), the term corpus (51 cases,
  `compare_corpus.mjs`, `cmp`'d against base), and the scale `--bench` stdout
  at N=40/250/1000/5000, `cmp`'d against base.

## Baseline: where a resolve goes

### N=40 (23.30 M Ir per warm resolve)

Per warm resolve: `step` is called 21,659 times, `backtrack` 1,296 times and
`execute_builtin` 923 times. There are 1,004 choice-point pushes: 383 at clause
`try_me_else`, ~560 ITE guards and aggregates, and ~60 for builtins. There are
10,790 register trail entries, 894 variable binds and 1,059 fresh variables.

The buckets do not overlap. Inclusive costs are split so that each Ir is
counted once. A callee that is shared, such as `deref_var` inside `get_reg`,
is counted in the bucket it belongs to.

| bucket | Ir / resolve | share | what is in it |
| --- | ---: | ---: | --- |
| (a) variable bind/deref/trail via the named-var map | ~1.5 M | ~6.5% | `deref_var` 0.83 M (all callers; mostly the owned-clone return), `fresh_var_sym` 0.41 M (interning `_V<n>`/`_H<n>`), `bind_var` 0.14 M, `unwind_trail_bindings_only` 0.13 M |
| (b) register name decode, register file and register trail | ~4.3 M | ~18.5% | `trail_binding` 1.55 M (its `get_reg` 0.87 M), `put_reg` 1.33 M (Y-frame `YRegs` copy-on-write 0.31 M, slot `resize` 0.34 M), `get_reg` from `step` 0.57 M, `set_reg` 0.29 M, `get_reg_raw` 0.26 M, the string-keyed `labels` map 0.79 M (SipHash 0.54 M, 2,677 lookups), minus the deref counted in (a) |
| (c) `Value`/spine cloning and the copy-on-write stack | ~2.3 M | ~9.7% | `Arc::make_mut` on `stack` 0.75 M (8,042 calls; a CP holds the `Arc`, so a push or pop of `UnifyCtx`/`WriteCtx` copies the frame vector), `StackEntry::clone` 0.49 M (`self.stack.last().cloned()` in every unify instruction), `args[1..].to_vec()` 0.42 M, dropping `Option<StackEntry>` 0.33 M and `Vec<Value>` 0.26 M |
| (d) choice-point save/restore of the register file | ~3.7 M | ~15.8% | `restore_ax_regs` 1.70 M (1,296 calls, ~1,300 Ir each), `save_regs` 0.97 M (~960 Ir per CP), `ChoicePoint` drops 0.35 M, `stack` `Arc` drops in `backtrack` 0.39 M, `backtrack` self 0.28 M |
| (e) builtins | ~4.7 M | ~20.3% | `execute_builtin` 3.87 M plus `resume_builtin` 0.87 M. Dispatch is ≤0.25 M (D124); the rest is builtin work: `execute_ext_builtin` 2.74 M (`sort_by` 1.25 M, `deref_list_arg` 0.77 M, `dedup_by`), `execute_term_builtin` 0.49 M, `unify` for `=/2` 0.33 M |
| (f) instruction dispatch and everything else | ~6.7 M | ~29% | `step` self 1.65 M (~76 Ir per instruction), `run` self 0.51 M, term construction 1.91 M (`set_heap_or_list` 1.36 M: functor-string parse, a linear scan of 200 registers for the `Ref(marker)`, placeholder atoms re-interned 0.34 M, `strv` 0.21 M), native Stage-2 regions 1.47 M, `Vec` pushes 0.40 M, `call_pred` + `reset_query` 0.36 M |

### N=5000 (388.51 M Ir per warm resolve)

| part | Ir / resolve | share |
| --- | ---: | ---: |
| `execute_builtin` (incl.) | 213.7 M | 55.0% |
| — `execute_ext_builtin`: 23 sort-family calls | 196.2 M | 50.5% |
| —— `deref_list_arg` (materialize list + elements) | 73.8 M | |
| —— `sort_by` (`term_compare_derefed`) | 56.3 M | |
| —— per-element `deref_heap` (`from_iter`) | 42.7 M | |
| —— `dedup_by` | 23.1 M | |
| `lowered_call`: Stage-2 regions (incl.) | 138.0 M | 35.5% |
| — `group_keyed` (`terms_identical` 28.8 M, `region_key_val` 17.1 M) | 59.8 M | |
| — `key_dep_rows` (`strv` 18.6 M, `dep_to_req` 12.0 M, alloc 8.9 M) | 49.8 M | |
| — `build_tree` + `key_pkg_rows` | 28.1 M | |
| `reset_query` (dropping the previous query's terms) | 19.1 M | 4.9% |
| interpreter mechanics (rest of `step`, plus `backtrack`) | ~17.6 M | ~4.5% |

Across all of the above (inclusive, overlapping): `deref_heap` 228.9 M (59%),
`term_compare_derefed` 113.8 M (29%), `Arc::drop_slow` 82.6 M (21%),
`__rust_alloc` 55.8 M (14%), `interner::decomp` 45.7 M (12%), and `"f/N"`
string scanning (`memrchr` + `CharSearcher`) ~23 M (6%).

The event counters are **identical at N=40 and N=5000**: the same steps,
backtracks, choice points, binds and trail entries. Only the size of the terms
the builtins and regions touch grows.

## The spike

All changes are to a scratch copy of the generated crate, in five steps,
tracked in a scratch git repo. None of it is committed here.

### s1: heap-cell variables with conditional trailing (`value.rs`, `state.rs`)

```rust
// value.rs
pub enum Value { …, Unbound(Sym), Var(usize) /* (n << 1) | tag; tag 1 = 'H' */, … }
// Display: Var(p) prints "_V<n>" / "_H<n>", exactly the name the named path
// gave, so output text and the standard order of variables are unchanged.

// state.rs (WamState)
pub vars: Vec<Value>,      // cell n; Value::Uninit = unbound; never truncated within a query
pub cond_mark: MarkCell,   // var_counter at the most recent rollback point

pub fn new_cell_var(&mut self, tag: u8) -> Value;   // PutVariable, UnifyVariable(write),
                                                     // SetVariable, RecurseCategoryAncestorPc
pub fn bind_cell(&mut self, p: usize, val: Value) {
    let i = p >> 1;
    if i < self.cond_mark.get() {                    // conditional trailing
        let old = match &self.vars[i] { Value::Uninit => None, v => Some(v.clone()) };
        self.trail.push(TrailEntry { key: TrailKey::Cell(i), old_value: old });
    }
    self.vars[i] = val;
}
pub fn trail_mark(&self) -> usize {                  // every `self.trail.len()` read (109 sites)
    self.cond_mark.set(self.var_counter);            // now goes through here
    self.trail.len()
}
```

- **Dual representation.** Only the four step arms above create cells.
  Builtins, the boundary and copy/rename keep creating named `Unbound(Sym)`
  variables. Both kinds are handled everywhere:
  - deref (`deref_var`, `deref_chain`, `deref_heap`, `deref_shallow`, the
    atom fast paths);
  - unify (both `n1 == n2` arms; binding goes through a new `bind_value`
    that dispatches on the kind, and the bind direction is unchanged);
  - the 13 runtime `bind_var` sites;
  - the trail undo paths (`TrailKey::Cell`/`TrailUndo::Cell`);
  - `same_cell`;
  - the 66 wildcard `Value::Unbound(_)` patterns, which became
    `Value::Unbound(_) | Value::Var(_)`;
  - standard order and text output (`var_name_string`);
  - `copy_term_walk`, the read-term relabel and `variant_terms`, which rename
    a cell through its printed name.
- **Conditional trailing and why it is sound here.** Cells are never reused
  within a query, so no heap-top reset is needed. A rollback point is any
  read of the trail length, because that is how every choice point and every
  non-CP unwind mark (lowered ITE, regions, `\+`, `call_goal_once`, findall,
  …) is taken. Each such read raises the mark to the current var counter. A
  cell numbered at or above the mark was created after every live rollback
  point, so no restored state can reach it, and its binding needs no undo.
  `var_counter` can be rolled back (`lo_restore_clause`, the region-1
  decline), so `new_cell_var` resets a reused cell. The mark is only ever
  raised, which is the conservative direction.
- **Measured.** 894 binds per resolve, of which 571 were trailed: **36% of
  binds skip the trail**, even with this conservative mark. Named binds per
  resolve dropped to 1.

### s2: clause choice points save only A1..A<arity>

```rust
Instruction::TryMeElse(label) => { …
    let __saved = match Self::spike_clause_arity(label) {   // "L_<pred>_<arity>_<k>", not "L_ite_*"
        Some(n) => self.save_regs_upto(n),                   // A1..An only (slots 0..n)
        None => self.save_regs(),                            // ITE guards etc.: full dirty set
    };
```

At a clause-alternative `try_me_else`, only the argument registers are live.
The compiler keeps everything that is live across a call in Y registers.
`restore_ax_regs` still clears every dirty slot, so a temporary that was not
saved comes back as `Uninit`. An instruction that read such a slot would fail,
and the differential would show a divergence. None did. Per resolve: 383
clause CPs, with an average of 10.25 dirty slots at the push but only 2.78
saved.

The generator knows the arity, so the rewrite emits it as an operand instead of
parsing the label.

### s3: no register trail (upper-bound probe)

`trail_binding` became a no-op. This is **not sound in general**. Any non-CP
rollback (`unwind_trail_to`) that needs a register back would see the
clobbered value. The `spike_stats` build showed that 0 of 10,790 register
entries per resolve are ever consumed. All gates stayed byte-identical over
2,600 differential cases and 51 corpus cases. So the register trail is not
load-bearing for this program. The rewrite drops it and makes non-CP rollback
points save the registers they need (see the design doc).

### Not covered by the spike

- Structures and lists are still `Arc`'d `Value` trees, still built through
  the `WriteCtx` scratch heap and still read through `UnifyCtx` frames on the
  copy-on-write stack. The spike moved variables only.
- Registers are still decoded from `String` names on every access. Labels are
  still a `HashMap<String, usize>`. Builtins are still dispatched by name.
  Numeric registers were not prototyped: they need the generator to change
  the `Instruction` operand types and all 103 emitted instruction forms.
- ITE-guard choice points (about 560 per resolve, 7.1 saved slots on average)
  still save the whole dirty set. Narrowing them needs the compiler's
  live-register set at the guard.
- No heap-top reset and no environment-stack protection. Frames are still the
  `Arc<Vec<StackEntry>>` copy-on-write stack.
- Builtins, regions, the boundary and fact sources are unchanged. They create
  named variables and read cells through the deref helpers.
- The store lane (`rust_store/`) was not rebuilt with the spike.

### Gates

| variant | differential (2600) | corpus (51) | byte identity vs base | generated-crate `cargo test --release --lib` |
| --- | --- | --- | --- | --- |
| s1 | 0 divergences / 0 crashes | 51/51 | diff, corpus and scale N=40/250/1000/5000 all `cmp`-identical | — |
| s2 (+ stats code, feature off) | 0 / 0 | 51/51 | identical | **260 passed, 0 failed** (after one test-only fix: a test `Model` that `self.trail_mark()` had captured) |
| s3 | 0 / 0 | 51/51 | identical | not run: `trail_binding` is a no-op, so the D127 register-trail tests would fail by design |

## What moved, per function (N=40, inclusive Ir per warm resolve)

| function | base | s1 | s2 | s3 |
| --- | ---: | ---: | ---: | ---: |
| `fresh_var_sym` → `new_cell_var` | 0.41 M | 0.09 M | 0.09 M | 0.09 M |
| `bind_var` → `bind_value` | 0.14 M | 0.06 M | 0.06 M | 0.06 M |
| `deref_var` | 0.83 M | 0.87 M | 0.81 M | 0.58 M |
| `save_regs` (+ `save_regs_upto`) | 0.97 M | 0.96 M | 0.38 M | 0.38 M |
| `restore_ax_regs` | 1.70 M | 1.73 M | 0.47 M | 0.46 M |
| `backtrack` | 3.38 M | 3.36 M | 2.07 M | 1.86 M |
| `trail_binding` | 1.55 M | 1.55 M | 1.51 M | 0 |
| `get_reg` | 1.44 M | 1.44 M | 1.38 M | 0.55 M |

## The new floor (s3, N=40, 19.20 M Ir)

| cost | Ir | share | rewrite item that removes it |
| --- | ---: | ---: | --- |
| builtins (`execute_builtin` + `resume_builtin`) | 4.61 M | 24.0% | cell-native sort/compare/list builtins, no materialization |
| `backtrack` | 1.86 M | 9.7% | `Copy` cells: memcpy restore, no drops; arity/live-set saves |
| copy-on-write stack for unify contexts (`make_mut` 0.92 + `StackEntry` clone 0.50 + drop 0.35) | 1.77 M | 9.2% | S register + read/write mode; env stack protected by B |
| `step` self | 1.57 M | 8.2% | numeric operands, PC-resolved calls, builtin ids |
| native regions | 1.46 M | 7.6% | regions ported to the cell API |
| term construction (`set_heap_or_list` 1.37 + `strv` 0.73 + `intern` 0.66, overlapping) | ~2.1 M | ~11% | put_structure writes cells in place; functor ids |
| `put_reg` | 1.33 M | 6.9% | numeric registers; Y slots in a plain env stack |
| `labels` lookups | 0.79 M | 4.1% | labels resolved to PCs by the generator |
| `get_reg` + `set_reg` + `get_reg_raw` | 1.31 M | 6.8% | numeric registers, `Copy` cells |

At N=5000 the floor is unchanged in kind. Builtins (216 M) and regions
(137 M) are 91% of Ir, dominated by `deref_heap`, `term_compare`, `Arc` drop,
malloc and functor decomposition. None of the three spike levers reaches them.
That is why the design makes the term representation itself (heap cells with
functor ids, read in place by builtins) part of the rewrite rather than a
follow-up.

## Files

Nothing in this report's spike is committed. The spike diff (s1–s3, ~2.2 K
diff lines, mostly the 109 mechanical `trail_mark` and 66 wildcard-pattern
substitutions) lived in a scratch copy of the generated crate. The committed
outputs are this report, the design doc and ledger row D128.
`resolver.pl` and `resolver_store.pl` are unmodified. The regenerated crate is
not committed.
