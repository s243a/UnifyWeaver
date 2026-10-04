<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM: borrow the instruction instead of cloning it per step (D122)

**Date:** 2026-10-04. **Ledger:** D122. **Author:** Opus.
**What:** stop the Rust WAM target's main loop from cloning the current
`Instruction` on every step. Output is byte-identical.
**Result:** about **−5.7 M Ir per warm resolve at every catalog size**: −16.2%
at N=40 (34.97 M → 29.30 M Ir) and −1.5% at N=5000 (407.42 M → 401.35 M).
The warm in-process N=40 resolve drops from 5.91 ms to 5.34 ms (−10%).

## Diagnosis (verified)

The probe and catalogs are the same as in D120 and D121
(`docs/reports/wam_rust_yreg_slot_frames.md`,
`docs/reports/wam_rust_backtrack_no_cp_clone.md`): `resolve_layered([p30])`
on `gen_scale_catalog.mjs` catalogs, capped by `scale_to_case.mjs` at N = 40 /
250 / 1000 / 5000. The base is origin/main `59dc49c` (D121 merged). It
reproduces D121's numbers: 34.97 / 50.98 / 107.05 / 407.42 M Ir per warm
resolve (D121 reported 34.95 / 50.99 / 107.06 / 407.44 M).

`WamState::run` (emitted by `compile_run_loop_to_rust` in
`wam_rust_target.pl`) did this on every step:

```rust
if let Some(instr) = self.fetch().cloned() {
    if !self.step(&instr) { ... }
```

`fetch()` returns `Option<&Instruction>` into `self.code`. The `.cloned()`
was there only so that `self` was no longer borrowed when `self.step(&instr)`
took `&mut self`. Most `Instruction` variants carry `String` register and
functor names, and some carry `Value`s (`GetConstant(Value, String)`,
`Call(String, usize)`, `PutValue(String, String)`, …). So every step allocated
and freed one or two strings. In the base N=40 profile (per warm resolve):

- `String::clone`: 3.00 M Ir inclusive (8.6%), almost all from `run`;
- `drop_in_place<Instruction>`: 1.37 M inclusive (3.9%);
- `run` self: 1.79 M (the inlined `Instruction::clone` match);
- mimalloc `malloc`/`free` inclusive: 3.60 M / 1.58 M, part of it from here.

## The change

### Program held in an `Arc`, borrowed through a snapshot

`templates/targets/rust_wam/state.rs.mustache`:

- `pub code: Vec<Instruction>` becomes `pub code: Arc<Vec<Instruction>>`.
- `WamState::new(code: Vec<Instruction>, labels)` keeps its public signature
  and wraps the program (`Self::new_shared(Arc::new(code), labels)`). Every
  existing caller (the pkg_resolver shims, `main.rs.mustache`,
  `materialisation_setup.rs.mustache`, the benchmark generators, the test
  harnesses) still passes a `Vec`.
- New `WamState::new_shared(Arc<Vec<Instruction>>, labels)` builds a machine
  over a program that is already shared.
- `fetch()` is now `Self::fetch_in(&self.code, self.pc)`. The new
  `fetch_in(code, pc)` holds the bounds rule (pc 0 or pc > len ⇒ `None`), and
  `run` uses the same function, so the two cannot drift apart.

`src/unifyweaver/targets/wam_rust_target.pl`, `compile_run_loop_to_rust`:

```rust
pub fn run(&mut self) -> bool {
    let mut code = Arc::clone(&self.code);          // O(1) snapshot
    loop {
        if self.pc == 0 { return true; }
        if self.step_limit > 0 && self.step_count >= self.step_limit {
            return false;
        }
        if !Arc::ptr_eq(&code, &self.code) {        // program changed?
            code = Arc::clone(&self.code);
        }
        if let Some(instr) = Self::fetch_in(&code, self.pc) {
            if !self.step(instr) { ... unchanged ... }
            self.step_count += 1;
        } else {
            return false;
        }
    }
}
```

`code` is a local, so borrowing an instruction from it does not borrow
`self`, and `step` gets `&code[pc - 1]` directly.

**Why this is exact.** The old loop fetched from the *current* `self.code` at
every step. The new loop fetches from the snapshot, but first checks with
`Arc::ptr_eq` that the snapshot is still the current program. That check
catches every way the program can change while a snapshot is held:

- **Assignment** (`self.code = new_arc`). The snapshot keeps the old
  allocation alive, so a new program can never get the same address (no ABA).
  Re-installing the *same* `Arc` keeps the pointer, but then the contents are
  the same too.
- **In-place edit.** `Arc` gives no `&mut` access except through
  `Arc::make_mut`/`Arc::get_mut`. While `run` holds its snapshot the refcount
  is at least 2, so `make_mut` copies to a new allocation (pointer changes) and
  `get_mut` returns `None`. `Instruction` has no interior mutability.

So whenever the program changes, the next step re-snapshots and fetches the
same instruction the old per-step fetch would have cloned. The bounds rule is
the same function. The snapshot keeps the instruction being stepped alive even
if `step` replaces `self.code` mid-instruction, which is what the old owned
clone guaranteed. The rest of the loop (pc 0, step limit, the throw check,
backtrack, `step_count`) is unchanged. The re-check costs one pointer compare
per step. It replaces a refcount bump per step (an atomic pair), which would
also have been exact but slower.

**Rejected alternatives.**

- An Arc clone *per step*: also exact and O(1), but it adds two atomic RMWs
  per step.
- `std::mem::take(&mut self.code)` around `step`: no allocation, but wrong.
  `step` reads `self.code` itself (the `GetLevel` arm peeks at
  `code[pc]` for a following `TryMeElse`, `BeginAggregate` scans forward for
  its `EndAggregate`), and nested `run()`s (meta-calls, the `read_term`
  parser, lowered dispatch) would see an empty program.
- A separate field borrow: `step` needs `&mut self` as a whole, so no split
  borrow exists without changing ownership or using `unsafe`.

### Callers updated (the field type changed)

Code that *assigns* `vm.code` now builds an `Arc`. None of it is on the
resolve hot path.

| site | before | after |
| --- | --- | --- |
| `foreign_wrapper_setup` (foreign-lowered wrapper), `rust_bidirectional_wrapper_code`, `rust_boundary_wrapper_code` (`wam_rust_target.pl`); 6 foreign wrappers in `rust_target.pl` | `vm.code = Vec::new();` | `vm.code = std::sync::Arc::new(Vec::new());` |
| `foreign_wrapper_setup` (standalone per-predicate WAM wrapper) | `vm.code = code;` | `vm.code = std::sync::Arc::new(code);` |
| shared table `SHARED_WAM` / `get_shared_wam` (single-vec and chunked forms) | `OnceLock<(Vec<Instruction>, …)>` | `OnceLock<(Arc<Vec<Instruction>>, …)>`, built once |
| shared-table wrappers (`compile_wam_predicate_to_rust_shared`) | `vm.code = code.clone();` (a deep copy of the whole program per call) | same text, now an O(1) `Arc` clone |
| `shared_wam_program()` | returns `(Vec, HashMap)` | unchanged signature and result (clones the `Vec` out of the `Arc`) |
| `read_term` runtime-parser bridge | `WamState::new(self.code.clone(), …)` (deep copy) | `WamState::new_shared(Arc::clone(&self.code), …)` |

Side effects that are not on the measured path: a shared-table wrapper call,
the `read_term` parser machine, and a `par_aggregate` fork (`WamState` is
`#[derive(Clone)]`) no longer deep-copy the program. The contents are never
mutated in place, so sharing is observably identical.

## Code-mutation audit

Searches covered every Rust-emitting generator, template, fixture and
benchmark generator in the repo: `self.code`, `vm.code`, `.code =`,
`code.push`/`extend`/`truncate`/`insert`/`iter_mut`, `Vec<Instruction>`,
`SHARED_WAM`, `shared_wam_program`, `WamState::new(`, and struct literals.

- **Inside `run`/`step`: nothing writes `self.code`.** The only reads are
  `fetch`, the `GetLevel` peek (`self.code.get(self.pc)`) and the
  `BeginAggregate` forward scan. Both work unchanged through `Arc` deref.
- **Dynamic DB** (`assert*`/`retract`/`clause`/`consult`, in
  `dynamic_db_methods.rs.mustache`): uses `dynamic_db: HashMap<String,
  Vec<Value>>` and never touches `code`. **par_aggregate**: clones the whole
  `WamState`, so the fork shares the program. **Lowered tier** and **regions**:
  `fn(&mut WamState)` bodies that call `vm.step(&Instruction::…)` with a
  temporary they build themselves, and never assign `vm.code`.
- **Whole-program replacement from outside `run`:** the wrapper functions
  above, and `tests/fixtures/wam_rust_dynamic_builtins.rs` (`vm.code =
  vec![…]` between runs). These could in principle run inside a `run` (a
  native/lowered body calling a public wrapper with the same `vm`). The
  `ptr_eq` re-check covers that case exactly.
- **In-place edits:** only `materialisation_setup.rs.mustache` and the matrix
  and effective-distance benchmark generators (`append_fact2`,
  `resolve_targets`, `optimize_benchmark_code` on a local
  `&mut Vec<Instruction>` *before* `WamState::new`, so unaffected), and one
  `vm.code.extend([...])` in the dynamic-builtins fixture between runs. That
  one is now `Arc::make_mut(&mut vm.code).extend(...)`. The refcount is 1
  there, so it does not even copy.
- The pkg_resolver shims (`rust/shim`, `rust_store/shim`) only call
  `WamState::new(code, labels)` and read `vm.labels`. They are unchanged.

**Tests whose expected strings moved.** `tests/test_wam_rust_target.pl` checks
for `'vm.code = Vec::new();'` in 13 places. These now check for
`'vm.code = std::sync::Arc::new(Vec::new());'`. The checks for
`'vm.code = code.clone()'` and `'(code.clone(), labels.clone())'` still match
the emitted text unchanged.

### Tests

`mod d122_code_snapshot_tests` in the template, compiled only under
`cargo test`, adds four tests:

1. **`run_matches_the_old_fetch_cloned_loop`.** The pre-D122 loop is kept
   verbatim in the test. Both loops run a backtracking program
   (`query(X) :- color(X), pick(X)`, which needs two backtracks into
   `color/1`) for 36 start states: success, failure with and without choice
   points, pc 0, pc past the end, and every step limit from 1 to 29. The test
   compares the result, pc, cp, `step_count`, `backtrack_count`, CP depth,
   heap and trail length, and the bound `X`.
2. **`fetch_in_keeps_the_fetch_bounds`.** pc 0, len+1 and an empty program
   give `None`. In-range pcs return the exact element (pointer equality), and
   `fetch()` equals `fetch_in` over the machine's own program.
3. **`every_program_change_moves_the_snapshot_pointer`.** With a snapshot
   held, a `make_mut` edit and an assignment both change the pointer, and the
   snapshot keeps the old program. Re-installing the same `Arc` does not change
   it. A `WamState::clone()` fork and `new_shared` share the program.
4. **`a_replaced_program_is_what_the_next_run_executes`.**

The full generated-crate lib suite passes: **244/244** (D121's 240 plus these
4).

A mid-run replacement cannot be triggered from a unit test without hot-path
hooks: nothing inside `step` writes `code`, and `lowered_call` is a static
per-crate table. Test 3 pins the mechanism the re-check relies on instead.

## Gates

Run from the repo root with `LANG=C.utf8 LC_ALL=C.utf8` on fresh `build.sh`
builds (term and store) of this tree and of an untouched origin/main `59dc49c`
export.

| gate | result |
| --- | --- |
| `rust/run_differential_rust.sh` | `cases: 2600 / divergences: 0 / crashes: 0` |
| `rust/run_corpus_rust.sh` | `corpus-under-rust: 51/51 matched SWI` |
| `rust_store/run_differential_rust_store.sh` | `cases: 503 / divergences: 0 / crashes: 0` |
| `rust_store/run_corpus_rust_store.sh` | `corpus-under-rust-store: 51/51 matched SWI`; `rust_store corpus IDENTICAL to term corpus (51/51)` |
| byte identity vs base build | `cmp`-identical: term diff `rust.jsonl` (2600 lines), term corpus `rust.jsonl` (51), store diff `rust.jsonl` (503), store corpus `rust_store.jsonl` (51); scale `--bench` stdout identical at N=40/250/1000/5000 |
| generated crate `cargo test --release --lib` | `test result: ok. 244 passed; 0 failed` |
| Rust WAM plunit (`tests/test_wam_rust_*.pl`, `tests/core/*rust*.pl`, 55 files) | same per-file rc as base (27 rc=0, 28 rc≠0) and the same 42 failing test names |
| CI rust conformance (`CONFORMANCE_TARGETS=rust`, `CONFORMANCE_PROGRAMS=member,builtins`) | rc=0, both unsampled and with `CONFORMANCE_SAMPLE=2` |

**Plunit, base vs new:** the failing tests are the same 42 that D120 and D121
recorded, the pre-existing breakage in the test harnesses' Rust
(`Value::Atom(String)` against interned `Sym`). The compile-error profile over
all 55 logs is identical, including counts (316 `E0308`, 26 `E0277`, 16
`String: Borrow<Sym>`, …). No error mentions `code`, `Arc<Vec<Instruction>>`
or `Vec<Instruction>`.

`test_wam_rust_dynamic_builtins` fails on base and new the same way, with only
the two pre-existing `Sym` errors in the fixture's `at()`/`ub()` helpers. To
check the fixture edits for real, a scratch copy of that test was run with
just those two helpers patched to `.into()`. It passes (rc=0). That covers
the `Arc::new(vec![…])` assignments, the `make_mut(...).extend` edit, and the
compiled `read_term` parser path, which now goes through `new_shared`. The
`Sym` fix itself is left out so that the failing set stays identical to base.

## Measurements

The method is the same as D120/D121. Binaries are `--release` with the
crate's default features (mimalloc on via the pkg_resolver `build.pl`). Ir
comes from `valgrind --tool=callgrind` on an uncommitted scratch copy of the
shim that repeats `call_pred` in-process. Each run checks that every repeat
gives the same answer. Ir per warm resolve = (Ir(1+k) − Ir(1)) / k, with k=10
at N=40 and k=4 elsewhere.

### Ir per warm resolve (deterministic)

| N | base (D121) | D122 | Δ | Δ absolute |
| ---: | ---: | ---: | ---: | ---: |
| 40 | 34.97 M | **29.30 M** | **−16.2%** | −5.67 M |
| 250 | 50.98 M | **45.30 M** | −11.2% | −5.68 M |
| 1000 | 107.05 M | **101.29 M** | −5.4% | −5.75 M |
| 5000 | 407.42 M | **401.35 M** | −1.5% | −6.06 M |

The saving is roughly flat (~5.7 M Ir) because it scales with the number of
WAM steps, which is about the same at every N for this probe. It grows a
little at N=5000, which takes slightly more steps. Whole-process Ir for one
cold resolve falls from 46.26 M to 40.58 M at N=40.

Where the N=40 saving comes from (per warm resolve):

| site | base | D122 |
| --- | ---: | ---: |
| `String::clone` (inclusive) | 3.00 M (8.6%) | 0.11 M (0.4%) |
| `drop_in_place<Instruction>` (inclusive) | 1.37 M (3.9%) | 0 |
| `run` self (inlined `Instruction::clone`) | 1.79 M | 0.51 M |
| `mi_malloc_aligned` / `mi_free` (inclusive) | 3.60 M / 1.58 M | 2.27 M / 0.89 M |
| `step` (inclusive) | 24.47 M | 24.43 M |

`step` itself is untouched (24.47 M → 24.43 M). All of the saving is the
removed clone, drop and allocation.

### Wall clock (shared 4-core container, noisy)

Base and new were interleaved, 9 rounds per size. "Cold" is the shim's own
`resolve_ms` from a fresh process. "Warm" is the median of 11 in-process
repeats, then the median over the 9 rounds.

| N | cold base | cold new | warm base | warm new | Δ warm |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 7.11 ms | **6.91 ms** | 5.91 ms | **5.34 ms** | −10% |
| 250 | 9.03 ms | **8.16 ms** | 8.06 ms | **7.22 ms** | −10% |
| 1000 | 17.55 ms | 19.80 ms | 15.67 ms | **15.18 ms** | −3% |
| 5000 | 59.41 ms | 68.13 ms | 64.18 ms | **63.44 ms** | −1% |

The warm numbers track the Ir change. At N=40, warm base was 5.51–6.13 ms
(plus one 8.9 ms outlier) and warm new was 4.83–5.37 ms (plus three rounds at
7.3–7.8 ms).

The cold numbers at N=1000 and N=5000 are bimodal on this box and should not
be read as a regression:

- At N=1000, cold base ranged 16.6–59.5 ms and cold new 15.9–48.9 ms.
- At N=5000, cold base ranged 57.1–159.0 ms and cold new 56.9–105.0 ms.
- The minima still favour new (15.87 vs 16.62 ms at N=1000, 56.91 vs 57.13 ms
  at N=5000). The deterministic Ir is lower at every size.

## What dominates next (post-D122 callgrind, N=40, 29.30 M Ir/resolve)

| cost | share | notes |
| --- | ---: | --- |
| `execute_builtin` (inclusive) | 15.6% | `execute_ext_builtin` alone is 9.6%, of which `sort_by` (`sort/msort` over `term_compare_derefed`) is 4.1%; the dispatch still tries `execute_arith/io/type/term/ext/meta` in sequence by string name |
| `backtrack` (inclusive) | 13.3% | mostly `restore_ax_regs` (5.8%, the drop and clone of the actual register state change) plus the trail unwind |
| `trail_binding` (inclusive) | 10.8% | |
| `get_reg` / `get_reg_raw` | 10.2% / 4.1% incl. | `get_reg_raw` clones the `Value`; `reg_index` parses the name string on every access |
| `bindings` SipHash | 6.4% + 3.0% | `hash_one` + `sip::write`, on `bindings: HashMap<String, Value>` |
| `format!` in `step` | 6.1% | `format_inner` + `fmt::write` (4.9%) |
| mimalloc malloc/free | ~11% incl. | spread across the sites above |
| `lowered_call` | 5.4% | the native regions (incl.) |

The next levers, roughly in order of size and safety:

1. **Pre-resolved register indices.** Instructions still carry register
   *names*, so every `get_reg`/`set_reg` re-parses `"A1"`/`"X3"`/`"Y2"`. Now
   that `step` borrows instructions, a pre-decoded index form would be read
   for free.
2. **`bindings` on FxHash or an id-keyed map.** This is the D112 pattern.
3. **Builtin dispatch by pre-resolved id** instead of trying each family by
   string name.
4. **Removing the `format!` calls in `step`.**

Each is a few percent of a small resolve. The N=5000 term path is still bound
by `deref_heap` and `term_compare_derefed`.

## Files

- `templates/targets/rust_wam/state.rs.mustache`: `code: Arc<Vec<Instruction>>`,
  `new` wraps, `new_shared`, `fetch_in`, `mod d122_code_snapshot_tests`.
- `src/unifyweaver/targets/wam_rust_target.pl`: `compile_run_loop_to_rust`
  (snapshot + `ptr_eq` loop); `read_term` parser bridge (`new_shared`);
  foreign-lowered, standalone, bidirectional and boundary wrappers (`Arc`
  assignments); the shared table in both single-vec and chunked forms
  (`Arc` in the `OnceLock`, `shared_wam_program` unchanged).
- `src/unifyweaver/targets/rust_target.pl`: 6 foreign wrappers
  (`vm.code = std::sync::Arc::new(Vec::new());`).
- `tests/test_wam_rust_target.pl`: 13 expected strings.
- `tests/fixtures/wam_rust_dynamic_builtins.rs`: `Arc::new(vec![…])`
  assignments and one `Arc::make_mut(...).extend`.

`resolver.pl` and `resolver_store.pl` are unmodified. The regenerated
checked-in crate is not committed.
