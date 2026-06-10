# WAM-Rust lowered dispatch: per-comparison atom allocation (and where T6 wins)

This note records a perf bottleneck found in the Rust lowered emitter, the fix,
the measured wins, and — for later — the conditions under which **T6
first-argument indexing (native `match`)** would add a further win on top.

## The bottleneck

Every `get_constant` (the hot head-match instruction, used by single-clause
facts, T3/T4 multi-clause head matching, and T5 clause-chain remainders) was
emitted as:

```rust
let _a = vm.get_reg("A1").unwrap_or(Value::Uninit);   // (1) clones the stored Value::Atom(String)
if _a.is_unbound() { /* bind */ }
else if _a != Value::Atom("alice".to_string()) { return false; }   // (2) allocates a String to compare
```

So a bound atom head-match paid **two heap allocations per comparison** — the
`get_reg` clone of the stored `Value::Atom(String)`, and the freshly-built
`Value::Atom("alice".to_string())` right-hand side. The T5 dispatch had the
same shape (`t5a1 == Value::Atom("...")` per guard). `get_integer` was already
allocation-free (`Value::Integer` is `Copy`).

## The fix

A borrowing, allocation-free comparison on `WamState`
(`templates/targets/rust_wam/state.rs.mustache`):

```rust
pub fn get_reg_ref(&self, name: &str) -> Option<&Value>      // borrow the register, no clone
pub fn match_reg_atom(&self, name: &str, atom: &str) -> Option<bool>
//   Some(true)  = bound & equal      (clause continues)
//   Some(false) = bound & not equal  (clause fails)
//   None        = unbound            (clause binds)
```

`match_reg_atom` derefs the register through the binding chain **by reference**
(mirroring `deref_var`) and compares the inner atom against the `&str` literal,
so neither side allocates on the bound path. The unbound path still builds the
`Value::Atom(String)` to bind (rare; only when actually binding).

The emitter (`wam_rust_lowered_emitter.pl`) now emits, for an atom `get_constant`
/ `get_nil` and for each T5 guard:

```rust
match vm.match_reg_atom("A1", "alice") {
    Some(true)  => {}
    Some(false) => return false,
    None        => { vm.trail_binding("A1"); vm.put_reg("A1", Value::Atom("alice".to_string())); }
}
```

Integer/float constants keep the (already allocation-free) `Value` comparison.

## Measured wins (release, lto, 5M iterations)

| Benchmark | before | after | speedup |
|-----------|-------:|------:|:-------:|
| single-clause `wide/20` (20 `get_constant`, all bound) | 7501 ms | 992 ms | **7.6×** |
| `d64/1` T5 dispatch, 64 distinct atoms, worst case (last clause) | 7381 ms | 1521 ms | **4.85×** |

A standalone micro-benchmark isolating the comparison agreed: a 64-way linear
scan comparing `Value::Atom(k.clone())` ran at 2899 ms vs 323 ms comparing
`&str` — i.e. **one heap allocation per comparison is the whole ~9× penalty.**
All Rust lowered tests (`t4`, `t5`, `ite_exec`, `save_regs`) still pass, so the
change is purely an allocation removal — no semantic change.

## Where T6 (first-arg indexing / native `match`) will win — for later

We benchmarked T6 (`match` on the first arg) **before** doing T6, to check it is
actually a win (compilers optimise dispatch aggressively; an earlier Go
array-dispatch experiment *lost* to the compiler). Findings, on the 64-key
micro-benchmark:

| dispatch (64 keys, worst case) | time |
|---|---:|
| linear, alloc per cmp (old generated) | 5802 ms |
| linear, `&str` (post-fix generated) | 679 ms |
| `match(&str)` (T6 native switch) | 118 ms |
| `HashMap` (array/map dispatch) | 110 ms |

Conclusions for a future T6:

1. **For few clauses (the typical 2–5), dispatch shape is noise** — the
   compiler flattens `match` / if-cascade / map to the same thing. T6 is **not**
   worth it there. (Matches the prior Go result; do **not** sweep blindly.)
2. **For many clauses, a native `match` is a real ~3–6× over the (now
   allocation-free) linear scan** — `match(&str)` 118 ms vs linear `&str`
   679 ms at 64 keys. So T6 has a genuine niche: **first-arg-indexed predicates
   with many clauses.** It must be **gated to a clause-count threshold** and
   re-benchmarked per target (string-atom targets get a comparison-tree `match`;
   int-interned-atom targets get a true jump table → bigger win).
3. **Map/array dispatch is not a win** (110 ms ≈ `match` 118 ms but with more
   machinery) — consistent with the earlier Go experience. Prefer the host's
   native `switch`/`match` over an explicit table.

So: revisit T6 only for the many-clause case, gated on clause count, verified
per target — the allocation fix above is the universal win and lands first.

## A second bottleneck (noted for later): T4 `lo_restore_clause`

A 64-clause **non-distinct** T4 (`multi_clause_n`) worst case ran at ~35 µs/call
even after the allocation fix, because `lo_clause_snapshot` / `lo_restore_clause`
clear and restore ~200 register slots **per failed clause** (≈64× per call).
That dominates the allocation cost for many-clause T4. It is a less common shape
(many same-first-arg clauses), but the restore cost scales with
`clauses × reg-file-size` and would be the thing to optimise (e.g. restore only
the trail + the actually-written A-registers) before pushing T4 to wide
predicates.
