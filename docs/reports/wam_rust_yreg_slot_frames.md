<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM: slot-indexed Y registers + per-frame COW (D120)

**Date:** 2026-10-04. **Ledger:** D120. **Author:** Opus.
**What:** remove the Rust WAM target's per-resolve "floor" cost: the
environment frame's permanent variables were a `HashMap<String, Value>`, and
the copy-on-write environment stack deep-cloned every frame's map after each
choice point. Replaced with a slot-indexed register vector (Stage 1) held
behind a per-frame `Arc` (Stage 2). Output is byte-identical.
**Result:** about **−40 M Ir per resolve at every catalog size**: −49% at N=40
(81.9 M → 41.7 M Ir) and −8.8% at N=5000 (454 M → 414 M). The N=40 resolve's
wall time drops from ~10.9 ms to ~6.5 ms (cold process, median of 9).

## Diagnosis (re-verified)

Probe: `resolve_layered([p30])` on `gen_scale_catalog.mjs` catalogs, capped by
`scale_to_case.mjs` at N = 40 / 250 / 1000 / 5000. Dependencies only point to
earlier packages, so p30's closure is about the same at every N. The resolve
work is close to constant, which makes the N=40 cost a clean measure of the
resolver's core search cost.

- The base binary at origin/main `5d45b741f` costs **81.9 M Ir per warm resolve
  at N=40** (coordinator measured ~82 M). SWI does the whole resolve in ~0.64 ms.
- Root cause: `WamState.stack: Arc<Vec<StackEntry>>` is copy-on-write via
  `smut()` → `Arc::make_mut`. Each `ChoicePoint` holds an O(1) `Arc` clone of the
  stack. Each frame was `StackEntry::Env(usize, HashMap<String, Value>)`. After
  any choice point, the next stack mutation (Allocate, Deallocate, a
  `UnifyCtx`/`WriteCtx` push or pop, or a Y write) deep-cloned **every** frame's
  hash table and `String` keys (~200 frames deep in this resolver). On top of
  that, every Y read or write SipHashed a `String`, and every Y write allocated
  a key.
- `D94` (`wam_rust_hotpath_deep_profile.md`) had measured this stack at 0.7%.
  At that time other costs were ~70× larger, so it was left alone. Once those
  costs were fixed (D96–D113), it became the floor.

## The change

All in `templates/targets/rust_wam/state.rs.mustache`, plus the `Allocate` arm
in `src/unifyweaver/targets/wam_rust_target.pl`. The rust_wam templates and
generator use the Y-register map at only four sites: the `get_reg_ref` and
`get_reg` Y paths, the `put_reg` Y path, and the Allocate push. The Deallocate
pops discard the payload (`Env(old_cp, _)`), so they are unchanged. Nothing
iterates the registers, so HashMap iteration order never reached any output.

```rust
pub enum StackEntry { Env(usize, YRegs), UnifyCtx(Vec<Value>), WriteCtx(usize) }

pub struct YRegs(Option<Arc<YSlots>>);          // None = nothing written yet
pub struct YSlots {
    slots: Vec<Option<Value>>,                  // index n-1 for canonical "Yn"
    other: Option<Box<HashMap<String, Value>>>, // non-canonical names (never emitted)
}
```

**Stage 1: slot-indexed registers.** `Yn` maps to slot `n-1`. The parse
allocates nothing and runs a short digit loop instead of SipHash. Writes grow
the vector. The old map semantics are preserved exactly:

- An empty slot (`None`), or a slot past the end, means "key absent". This is
  different from a stored `Value::Uninit`, which a trail unwind writes through
  `put_reg(reg, Value::Uninit)`. `get_reg` still returns `Some(Uninit)` for it
  (the map returned the stored value), and `get_reg_ref` still hides it.
- A name the compiler never emits (leading zero such as `Y01`, `Y0`, trailing
  non-digits, or n > 4096) goes to a lazily allocated String-keyed side map, so
  it behaves exactly as before and cannot collide with a canonical slot.
- `put_reg` with no Env frame on the stack is still a silent no-op. Writes
  still go to the topmost Env frame, and `smut()` is still called first.

**Stage 2: per-frame copy-on-write.** The slots sit behind their own `Arc`.
When `smut()` copies the outer `Vec<StackEntry>` after a choice point, it now
copies one pointer per frame (a refcount bump) instead of each frame's
registers. A later Y write calls `Arc::make_mut` on the one frame it touches,
and clones that frame only if a choice point still shares it. `YRegs::new()` is
`None`, so `Allocate` allocates nothing, just as `HashMap::new()` did.
Choice-point snapshots stay fully independent because every mutation goes
through `make_mut`, and `YSlots` fields are private, so no other write path
exists. `WamState` remains `Clone + Send` (the static assertion still compiles),
and the T7 par_aggregate fork shares frames safely through `Arc`.

Stage 2 was taken because the post-Stage-1 profile still showed the O(depth)
copy as material. At N=40, `Arc::make_mut` was 16.2% inclusive and
`drop_in_place<StackEntry>` 6.9% (dropping the old copies), about 12 M of the
remaining 52 M Ir per resolve. After Stage 2, the residual outer-Vec copy
(`Arc::clone_from_ref_in`) is ~0.65 M Ir per resolve (~1.5%).

Unit tests (`mod d120_yregs_tests` in the template, compiled only under
`cargo test`) pin four behaviours: absent vs stored `Uninit`, non-canonical
names, topmost-frame writes and the no-frame no-op, and snapshot independence
after a later write. All 4 pass, and so does the full generated-crate lib suite
(236/236 in the regenerated pkg_resolver crate).

## Gates

Run from the repo root with `LANG=C.utf8 LC_ALL=C.utf8` on fresh
`build.sh` builds (term and store) of this tree.

| gate | result |
| --- | --- |
| `rust/run_differential_rust.sh` | `cases: 2600 / divergences: 0 / crashes: 0` |
| `rust/run_corpus_rust.sh` | `corpus-under-rust: 51/51 matched SWI` |
| `rust_store/run_differential_rust_store.sh` | `cases: 503 / divergences: 0 / crashes: 0` |
| `rust_store/run_corpus_rust_store.sh` | `51/51 matched SWI`; `rust_store corpus IDENTICAL to term corpus (51/51)` |
| byte identity vs base build | `cmp`-identical: term diff `rust.jsonl` (2600 lines), term corpus `rust.jsonl` (51), store diff `rust.jsonl` (503), store corpus `rust_store.jsonl` (51); scale `--bench` stdout identical at N=40/250/1000/5000 |
| Rust WAM plunit (`tests/test_wam_rust_*.pl`, `tests/core/*rust*.pl`, 55 files) | same pass/fail set as base (see below) |
| CI conformance leg (`CONFORMANCE_TARGETS=rust`, `CONFORMANCE_PROGRAMS=member,builtins`) | rc=0, both unsampled and `CONFORMANCE_SAMPLE=2` |

**Plunit, base vs new:** each of the 55 files was run with
`swipl -q -g run_tests -t halt <file>`, once in this tree and once in a clean
copy of origin/main. The exit code per file and the set of failing test names
per file are **identical**: 27 files rc=0 and 28 rc≠0, with the same 42
failing test names in both trees. Every failure predates this change. Most come
from the test harnesses' hand-written Rust (`tests/*.rs`, lowered-emitter
`lib.rs`) still building `Value::Atom("..".to_string())` and similar after `Value`
moved to interned `Sym` names (`E0308 expected Sym, found String`, `Sym: Ord`,
`Sym: AsRef<Path>`). None of the compile errors mentions
`YRegs`/`StackEntry`/`y_regs`. These failures are unrelated and were left as
they are.

## Measurements

Binaries: the base is origin/main `5d45b741f` built by its own `build.sh`.
"New" is this tree. Both use `--release` with the crate's default features
(the pkg_resolver `build.pl` passes `mimalloc(true)`, so mimalloc is on for
both, as checked in). Ir per resolve comes from
`valgrind --tool=callgrind` on a scratch copy of the shim (not committed) that
repeats `call_pred` in-process. Each run is checked to give the same answer on
every repeat. Per-resolve Ir = (Ir(1+k) − Ir(1)) / k, so start-up, load and the
first resolve cancel out (k=10 at N=40, k=4 elsewhere).

### Ir per warm resolve (deterministic)

| N | base | Stage 1 only | Stage 1+2 (shipped) | Δ shipped | Δ absolute |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 81.90 M | 52.34 M (−36.1%) | **41.69 M** | **−49.1%** | −40.2 M |
| 250 | 97.88 M | | **57.67 M** | **−41.1%** | −40.2 M |
| 1000 | 153.90 M | | **113.80 M** | **−26.1%** | −40.1 M |
| 5000 | 454.23 M | 424.77 M (−6.5%) | **414.19 M** | **−8.8%** | −40.0 M |

The saving is a flat ~40 M Ir per resolve at every N. That fits a cost tied to
the closure's search, not to catalog size. Whole-process Ir for one cold
resolve (start-up + load + resolve) is 93.3 M → 53.0 M at N=40 and
681.2 M → 641.1 M at N=5000.

### Wall clock (shared 4-core container — noisy)

Base and new runs were interleaved, 9 rounds per size. "Cold" is the shim's own
`resolve_ms` from a fresh process (`uw_resolve --bench`). "Warm" is the median
of 11 in-process repeats, then the median over the 9 rounds.

| N | cold base | cold new | Δ | warm base | warm new | Δ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 10.93 ms | **6.54 ms** | −40% | 9.43 ms | **5.23 ms** | −45% |
| 250 | 12.40 ms | **8.39 ms** | −32% | 11.02 ms | **6.98 ms** | −37% |
| 1000 | 18.68 ms | **15.31 ms** | −18% | 18.94 ms | **14.54 ms** | −23% |
| 5000 | 57.28 ms | **50.72 ms** | −11% | 55.39 ms | **53.27 ms** | −4% |

Spread is roughly ±10–20% and includes one outlier round per size, but every
size's new distribution sits below base: at N=40, cold base was 8.97–12.8 ms
plus one 35 ms outlier, cold new was 5.57–7.17 ms, and warm new was 4.98–6.64
ms against 8.90–9.95 ms for base. At N=5000 the Ir gain (−8.8%) is within the
box's noise, so the wall number there is only indicative.

## Verdict

D120 has landed in both stages. The environment-stack floor is gone: the stack
COW is now ~1.5% of an N=40 resolve, down from about half of it. The change
cuts Ir per small resolve roughly in half and is byte-identical on every gate.
The N=5000 term path is still dominated by `deref_heap` and
`term_compare_derefed`, which were out of scope here.

## What dominates the floor next (post-D120 callgrind, N=40, 41.7 M Ir/resolve)

| cost | share | site |
| --- | ---: | --- |
| `backtrack` (inclusive) | 18.6% | `choice_points.last().cloned()` clones the whole `ChoicePoint` (its `saved_args` Vec, `levels: Vec<(String, usize)>`, `builtin_state`) on **every** backtrack before restoring. `restore_regs` alone is 9.8% exclusive. |
| choice-point arg snapshot | 8.7% | `Vec::from_iter` in `step`, which collects `saved_args` for each new choice point |
| `execute_builtin` (inclusive) | 10.9% | builtin dispatch: `execute_arith/io/type/…` tried in sequence by string name |
| per-step instruction clone | ~7.5% | `run` does `self.fetch().cloned()` on every step, so each `Instruction`'s `String` register and functor names are cloned (`String::clone` 6.0%, called from `run` ~27 K times per resolve) and then dropped (`drop_in_place<Instruction>` 1.4%) |
| `format!` in `step` | 4.1% | ~2.5 K `format_inner` calls per resolve from `step` |
| bindings SipHash | 4.4% | `bindings: HashMap<String, Value>` probed in `deref_var`, `bind_var` and the trail unwind (the same SipHash-on-String pattern D112 removed from the interner) |
| mimalloc malloc/free | ~9% | spread across the sites above |

The cheapest next levers are (1) borrowing the instruction instead of cloning
it per step (e.g. `Arc<[Instruction]>` code or index-based dispatch) and (2)
backtracking from a borrowed choice point instead of a full clone. Each is
worth a few percent of a small resolve, and both are byte-identical by
construction.
