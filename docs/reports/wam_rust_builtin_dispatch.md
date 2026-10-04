<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM: core builtins ahead of the family cascade (D124)

**Date:** 2026-10-04. **Ledger:** D124. **Author:** Opus.
**What:** in the Rust WAM target's `execute_builtin`, match the four core
builtins (`true/0`, `fail/0`, `!/0`, `=/2`) before the six-family dispatch
cascade instead of after it. Every other name runs the same cascade in the
same order. Output is byte-identical.
**Result:** about **−0.11 M Ir per warm resolve at every catalog size**: −0.41%
at N=40 (27.98 M → 27.86 M Ir) and −0.03% at N=5000. This is small. The
profile shows that almost all of `execute_builtin`'s 16% is the builtins'
own work, not dispatch (see below), so there was not much more to take
without changing builtin semantics.

## Profile first: dispatch vs builtin work

The base is the D123 commit `080f9bb`. The probe, catalogs and method are the
same as in D120–D123 (`docs/reports/wam_rust_register_decode.md`). Base:
27.98 M Ir per warm resolve at N=40.

`execute_builtin` was:

```rust
if self.execute_arith_builtin(op, arity) { return true; }
if self.execute_io_builtin(op, arity) { return true; }
if self.execute_type_builtin(op, arity) { return true; }
if self.execute_term_builtin(op, arity) { return true; }
if self.execute_ext_builtin(op, arity) { return true; }
if self.execute_meta_builtin(op, arity) { return true; }
match op { "true/0" => …, "fail/0" => …, "!/0" => …, "=/2" => …, _ => false }
```

Each family is a `match op` on `&str`. rustc lowers it to a length switch plus
inline byte compares, so a failed family match costs tens of instructions,
not a chain of `memcmp`s.

Per warm resolve at N=40 (callgrind call counts and costs, base):

| | Ir / resolve | share of 27.98 M |
| --- | ---: | ---: |
| `execute_builtin` inclusive | 4.48 M | 16.0% |
| — `execute_ext_builtin` (`sort_by` 1.25 M, `deref_list_arg` 0.81 M, `dedup_by` 0.24 M, `unify`, …) | 2.81 M | 10.0% |
| — `execute_term_builtin` (`deref_heap` 0.32 M, `unify` 0.16 M, …) | 0.63 M | 2.2% |
| — `unify` for `=/2` | 0.57 M | 2.0% |
| — `execute_meta_builtin` | 0.16 M | 0.6% |
| — `execute_io_builtin` (claims nothing on this workload: pure dispatch) | 0.04 M | 0.15% |
| **all self cost of the dispatch functions** (`execute_builtin` + the five non-inlined families) | **0.25 M** | **0.9%** |

The last row is an upper bound on dispatch. It also includes the bodies of
the core arms and of the inlined arith/type families. **So dispatch was at most
~0.25 M of the 4.48 M.** The rest is real builtin work: sorting, list
dereferencing and unification. That is out of D124's scope, since it would
mean rewriting builtin semantics.

Where the dispatch cost lands follows from the call counts. There are about
923 `execute_builtin` calls per resolve. Arith handles 84 and term about 73.
Then **743 calls (80%) fall all the way through to `execute_meta_builtin`**,
having failed every family match. Of these, 353 are `=/2` (the `unify` calls
from `execute_builtin`) and about 108 are `true/0`/`fail/0`/`!/0`. The others
are meta-family names or family builtins that failed and fell through.

## Choosing the fix

The cascade order matters as semantics. A family that claims a name but fails
returns `false`, and the cascade then goes on to the later families. If two
families claimed the same name, the second could run. So the first question
was whether any names are shared.

- The six family match statements were parsed out of the generated
  `state.rs`, with a lexer that skips strings, chars and comments and tracks
  brace depth. Every top-level arm is a plain string literal or an or-pattern
  of literals, plus a final `_ => false` (`_ => unreachable!()` in type,
  behind a pure `matches!` pre-check). No arm has a guard and no arm uses a
  binding pattern. The only code ahead of a family `match op` is the type
  family's pure name check and a `use` in ext.
- The 216 names (arith 8, io 87, type 9, term 29, ext 58, meta 21, core 4)
  are **pairwise disjoint**: no name is claimed by more than one family or by
  the core match.
- The family code does not depend on generator options:
  `compile_wam_helpers_to_rust(_Options, …)` ignores them, and io's template
  partials render with fixed parameters. So this holds for every generated
  crate, not just pkg_resolver.

**Options considered:**

1. **Resolve the name to a family id once** (a `match op → u8` table, or a
   per-pc decode), then call only that family. It would remove all failed
   family matches, but it needs a hand-kept name table that has to mirror
   ~216 arms in 6 generator atoms and 5 templates. If someone adds an arm and
   forgets the table, that builtin silently stops working. The table cannot
   be generated without parsing the families' Rust text in Prolog. The extra
   gain over option 2 is bounded by the 0.25 M ceiling.
2. **Move the core match ahead of the cascade.** This takes the four
   commonest names (461 of the 743 full fall-throughs per resolve) out of the
   cascade. It needs no table: the core arms already live in
   `execute_builtin`. It is exact as long as no family claims a core name,
   which a test can pin cheaply.
3. **Reorder the families.** This would be exact today, because the names
   are disjoint, but the profile gives no order that clearly wins, and
   option 2 already covers the biggest group.

Option 2 was taken.

## The change

`compile_execute_builtin_to_rust` in `src/unifyweaver/targets/wam_rust_target.pl`:

```rust
pub fn execute_builtin(&mut self, op: &str, arity: usize) -> bool {
    match op {
        "true/0" => { self.pc += 1; true }
        "fail/0" => false,
        "!/0" => { self.choice_points.truncate(self.cut_barrier); self.pc += 1; true }
        "=/2" => { /* unchanged unify of A1/A2 */ }
        _ => {
            if self.execute_arith_builtin(op, arity) { return true; }
            if self.execute_io_builtin(op, arity) { return true; }
            if self.execute_type_builtin(op, arity) { return true; }
            if self.execute_term_builtin(op, arity) { return true; }
            if self.execute_ext_builtin(op, arity) { return true; }
            if self.execute_meta_builtin(op, arity) { return true; }
            false
        }
    }
}
```

**Why it is exact.** For a core name, the old code ran six families that do
not claim it. Each returned `false` at `_ => false` without touching the
machine, so the core arm then ran on the unchanged state. The new code runs
the same core arm on the same state. For any other name, the old code ran the
cascade and then the core match, whose `_ => false` returned `false`. The new
code runs the same cascade in the same order and returns `false`. The arm
bodies are unchanged text.

### Tests

- **`tests/test_wam_rust_builtin_dispatch.pl`** (new plunit file, 3 tests)
  works at the generator level, on the actual family code atoms:
  - `no_family_claims_a_core_builtin`: none of the 4 core names appears in a
    pattern position (a quoted literal followed by `=>`/`|`, or preceded by
    `|`) in any of the 6 families. The detector over-approximates (an inner
    match or a `||` would also count), which can only make the test
    stricter.
  - `detector_finds_known_family_patterns`: a sanity check of that detector.
    It finds `compare/3`, `sort/2`, `is/2`, `==/2`, `=</2` (middle of an
    or-pattern) and `maplist/2`, and it ignores `"=/2".to_string()`, which
    the term family uses as data.
  - `core_arms_precede_the_family_cascade_in_order`: every core arm comes
    before the first family call, and the six calls are still in the order
    arith, io, type, term, ext, meta.
- **`mod d124_builtin_dispatch_tests`** in the template (generated crate,
  `cargo test` only) works at runtime:
  - `no_family_claims_a_core_builtin`: each family, called directly with each
    core name on 8 machine states, returns `false` and leaves the machine's
    fingerprint unchanged. The fingerprint covers pc, cp, choice points, cut
    barrier, trail, heap, stack, every live register and every binding.
  - `execute_builtin_matches_the_old_cascade`: the pre-D124 dispatcher is
    kept verbatim. Old and new are compared on 12 names (the 4 core names,
    names from arith/type/ext/term, and an unknown name) × 8 states. The
    states cover unifiable, clashing, var-binding, chain-bound and `Uninit`
    A1/A2, plus a 3-deep choice-point stack with a cut barrier. Both the
    result and the full fingerprint must match.

The generated-crate lib suite passes: **250/250** (D123's 248 plus these 2).

## Gates

Run from the repo root with `LANG=C.utf8 LC_ALL=C.utf8` on fresh `build.sh`
builds (term and store) of this tree. The base outputs are the D123 build's
(`080f9bb`), which are in turn `cmp`-identical to origin/main `6649944`.

| gate | result |
| --- | --- |
| `rust/run_differential_rust.sh` | `cases: 2600 / divergences: 0 / crashes: 0` |
| `rust/run_corpus_rust.sh` | `corpus-under-rust: 51/51 matched SWI` |
| `rust_store/run_differential_rust_store.sh` | `cases: 503 / divergences: 0 / crashes: 0` |
| `rust_store/run_corpus_rust_store.sh` | `corpus-under-rust-store: 51/51 matched SWI`; `rust_store corpus IDENTICAL to term corpus (51/51)` |
| byte identity vs base build | `cmp`-identical: term diff `rust.jsonl` (2600 lines), term corpus `rust.jsonl` (51), store diff `rust.jsonl` (503), store corpus `rust_store.jsonl` (51); scale `--bench` stdout identical at N=40/250/1000/5000 |
| generated crate `cargo test --release --lib` | `test result: ok. 250 passed; 0 failed` |
| Rust WAM plunit (`tests/test_wam_rust_*.pl`, `tests/core/*rust*.pl`) | the 55 existing files give the same per-file rc as base (27 rc=0, 28 rc≠0) and the same 42 failing test names; the new `test_wam_rust_builtin_dispatch.pl` is rc=0 (3/3) |
| CI rust conformance (`CONFORMANCE_TARGETS=rust`, `CONFORMANCE_PROGRAMS=member,builtins`) | rc=0, both unsampled and with `CONFORMANCE_SAMPLE=2` |

## Measurements

The method is the same as D120–D123: callgrind Ir per warm resolve =
(Ir(1+k) − Ir(1)) / k, with k=10 at N=40 and k=4 elsewhere, on an
uncommitted scratch shim that repeats `call_pred` in-process and checks every
answer.

### Ir per warm resolve (deterministic)

| N | base (D123) | D124 | Δ | Δ absolute |
| ---: | ---: | ---: | ---: | ---: |
| 40 | 27.98 M | **27.86 M** | **−0.41%** | −0.115 M |
| 250 | 43.97 M | **43.86 M** | −0.26% | −0.113 M |
| 1000 | 99.98 M | **99.88 M** | −0.10% | −0.098 M |
| 5000 | 400.08 M | **399.96 M** | −0.03% | −0.118 M |

At N=40, 461 calls per resolve no longer reach the cascade (`execute_io_builtin`
calls 839 → 378 and `execute_meta_builtin` calls 743 → 282 per resolve). That
saves about 250 Ir per core call: six failed matches plus the call overhead.
The self cost of the dispatch functions drops from 0.25 M to 0.18 M. The
remaining 0.18 M is the core-arm bodies, the inlined arith/type families, and
the ~380 non-core calls that still walk the cascade.

### Wall clock (shared 4-core container, noisy)

Base and new were interleaved, 9 rounds per size. "Cold" is the shim's own
`resolve_ms` from a fresh process. "Warm" is the median of 11 in-process
repeats, then the median over the 9 rounds.

| N | cold base | cold new | warm base | warm new |
| ---: | ---: | ---: | ---: | ---: |
| 40 | 7.85 ms | 7.00 ms | 5.08 ms | 5.03 ms |
| 250 | 7.90 ms | 8.20 ms | 7.23 ms | 7.50 ms |
| 1000 | 23.99 ms | 19.84 ms | 15.59 ms | 15.30 ms |
| 5000 | 64.44 ms | 67.67 ms | 66.41 ms | 64.91 ms |

A 0.4% change is far below this box's noise, so the wall numbers neither show
nor rule out the change. At N=40, warm base was 4.80–6.76 ms and warm new
4.77–7.05 ms, with nearly identical lower halves. Ir is the metric for this
item.

## What dominates next (post-D124 callgrind, N=40, 27.86 M Ir/resolve)

| cost | share | notes |
| --- | ---: | --- |
| `execute_builtin` (inclusive) | 15.7% | builtin work: `execute_ext_builtin` 10.0% (`sort_by` over `term_compare_derefed`, `deref_list_arg`, `dedup_by`), `unify` for `=/2`, term builtins' `deref_heap`. Dispatch is now ≤0.18 M |
| `backtrack` (inclusive) | 14.0% | `restore_ax_regs` 6.1% |
| `trail_binding` (inclusive) | 9.0% | `get_reg` (deref'd old value) + a `String` key alloc per trailed register |
| `get_reg` (inclusive) | 7.6% | mostly `deref_var`, the `bindings` lookup |
| `bindings` SipHash | ~9.8% | `hash_one` + `sip::write` on `bindings: HashMap<String, Value>` |
| `format!` in `step` | 6.4% | 1.4 K per resolve build a `"f/N"` key only to compare it (`GetStructure` read mode, `SwitchOnStructure`), and 1.1 K are fresh-variable names |

The next levers are listed in D123's report: `bindings` on an interned-id Fx
map, the `format!`s in `step`, and an alloc-free register trail entry. On the
builtin side, any further win is in the builtins themselves. For example,
`sort/2` re-dereferences every element per comparison, which is a semantics
review, not dispatch.

## Files

- `src/unifyweaver/targets/wam_rust_target.pl`: `compile_execute_builtin_to_rust`
  (core arms first; cascade unchanged inside `_ =>`).
- `templates/targets/rust_wam/state.rs.mustache`: `mod d124_builtin_dispatch_tests`.
- `tests/test_wam_rust_builtin_dispatch.pl`: new generator-level plunit file.

`resolver.pl` and `resolver_store.pl` are unmodified. The regenerated
checked-in crate is not committed.
