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

## Where T6 (first-arg indexing / native `match`) wins — now implemented (gated) for Rust

We benchmarked T6 (`match` on the first arg) **before** building it, to check it
is actually a win (compilers optimise dispatch aggressively; an earlier Go
array-dispatch experiment *lost* to the compiler). The initial 64-key
micro-benchmark:

| dispatch (64 keys, worst case) | time |
|---|---:|
| linear, alloc per cmp (old generated) | 5802 ms |
| linear, `&str` (post-fix generated) | 679 ms |
| `match(&str)` (T6 native switch) | 118 ms |
| `HashMap` (array/map dispatch) | 110 ms |

T6 is now **implemented for the Rust lowered emitter, behind a clause-count
gate** (`emit_clause_chain_match_rust`). When the `wam_clause_chain` guards are
**all atoms** and there are at least `t6_min_clauses` of them (option, **default
8**), the emitter replaces the T5 if-cascade with a native **two-stage `match`**:
a string `match` mapping the first arg's atom to a `Copy` selector index (the
immutable borrow of the register ends there, via `WamState::match_reg_atom_str`,
so the clause body can take `&mut vm`), then an integer `match` — a jump table —
dispatching to the clause remainder. Below the gate it stays the T5 cascade.

### Measured on the *generated* code (release + lto, 5M dispatches, keys rotated over the whole key space)

| N clauses | T5 cascade | T6 `match` | speedup |
|---:|---:|---:|:-:|
| 4 (gate declines → stays T5) | 126 ms | 129 ms | 1.0× (tie) |
| 8 (gate threshold) | 157 ms | 101 ms | **1.55×** |
| 16 | 260 ms | 101 ms | **2.6×** |
| 64 | 876 ms | 153 ms | **5.7×** |
| 256 | 2991 ms | 236 ms | **12.7×** |

This is the *generated* code (not a micro-benchmark): the T5 cascade is O(N)
(average ≈ N/2 string compares), while T6 is roughly flat (a string-match
decision tree → integer jump table). The crossover is below the gate — at N=8
T6 already wins 1.55×, and the win grows with clause count.

Conclusions (validated):

1. **For few clauses (the typical 2–5), dispatch shape is noise** — the
   compiler flattens `match` / if-cascade / map to the same thing (confirmed:
   N=4 is a tie). T6 is **not** worth it there, so the gate declines and emits
   the T5 cascade. (Matches the prior Go result; do **not** sweep blindly.)
2. **For many clauses, a native `match` is a real 1.5–12.7× over the (now
   allocation-free) linear scan**, growing with N. So T6's niche is
   **first-arg-indexed predicates with many atom clauses**, and it is gated to
   the clause-count threshold accordingly.
3. **Map/array dispatch is not a win** (110 ms ≈ `match` 118 ms but with more
   machinery) — consistent with the earlier Go experience. We use the host's
   native `match`, not an explicit table.

### Porting to other targets

The gate + front-end are reusable; only the back-end emit differs. **String-atom
targets** (cpp) get the same string-`match` decision tree. **Int-interned-atom
targets** (go, llvm, haskell, scala, lua) get a *true* single-stage jump table on
the interned id — a bigger win and simpler emit — but each must be re-benchmarked
because, on ints, the compiler more readily converts even the T5 if-cascade into
a switch (the "lost to the compiler" risk is highest there). Gate on clause
count and verify per target.

## A second bottleneck (noted for later): T4 `lo_restore_clause`

A 64-clause **non-distinct** T4 (`multi_clause_n`) worst case ran at ~35 µs/call
even after the allocation fix, because `lo_clause_snapshot` / `lo_restore_clause`
clear and restore ~200 register slots **per failed clause** (≈64× per call).
That dominates the allocation cost for many-clause T4. It is a less common shape
(many same-first-arg clauses), but the restore cost scales with
`clauses × reg-file-size` and would be the thing to optimise (e.g. restore only
the trail + the actually-written A-registers) before pushing T4 to wide
predicates.

## Sweep to other targets

The per-comparison atom allocation is **representation-specific**. Survey of the
lowered targets:

| target | atom repr | per-cmp alloc? | action |
|---|---|---|---|
| rust | `Value::Atom(String)` | yes (get_reg clone + RHS temp) | fixed (`match_reg_atom`) |
| cpp  | `Value{std::string s}` | yes (get_reg copy + RHS temp) | fixed (`match_reg_atom`) |
| go   | interned `*Atom` (cached ref) | no | clean |
| llvm | interned i32 atom id | no | clean |
| haskell | interned `Atom Int` | no | clean |
| scala | interned int (atom table) | no | clean |
| lua  | interned (intern_table) | no | clean |
| clojure | interned via `normalize-literal-term` / `interned-equal?` | no (a hash lookup, not an alloc) | clean |
| fsharp | `Atom of string`, but `get_constant` **delegates to `step`** (.NET interns string literals) | not the inline hot-alloc shape | left as is |

### C++ result

Same fix as Rust (`WamState::match_reg_atom` reads the register cell in place —
no `Value` copy, no temporary `Value::Atom`; `deref` is a no-op in C++'s
cell-mutates-in-place model). Measured (g++ -O2, 5M iters):

| benchmark | before | after | speedup |
|---|---:|---:|:-:|
| single-clause `wide/20` | 5821 ms | 3351 ms | **1.74×** |
| `d64/1` T5 dispatch, 64 atoms, worst case | 8446 ms | 6266 ms | **1.35×** |

The win is smaller than Rust's because **C++ stores the register file as
`std::unordered_map<std::string, CellPtr>`** — every register access also pays a
string-keyed hash lookup that the allocation fix does not touch. That hash-map
register file (vs Rust/Go's flat `Vec`/array indexed by an int) is the **next**
C++ bottleneck and a bigger structural change; noted here for later.

### C++ register file: hash-map -> flat array (the "next bottleneck", now done)

Followed up on the C++ register-file finding above. `regs` changed from
`std::unordered_map<std::string, CellPtr>` to a flat `std::vector<CellPtr>`
indexed by `reg_index(name)` (A1->0.., X1->100..; Y registers still live in env
frames). The six `saved_regs` snapshots (ChoicePoint + catcher/negation/conj/
disj/ifthen frames) became vectors too; every snapshot/restore site is a
whole-container copy/move, so no backtracking logic changed.

Measured (g++ -O2, 5M iters, both **on top of** the allocation fix):

| benchmark | map | flat array | speedup |
|---|---:|---:|:-:|
| single-clause `wide/20` (20 reg accesses/call) | 3360 ms | 1609 ms | **2.09×** |
| `d64/1` T5 dispatch (1 reg access, 64 atom compares) | 6310 ms | 5975 ms | 1.06× |

The win **scales with register accesses per call**: `wide/20` does 20 head
matches and gains 2.09×; `d64/1` reads only A1 per guard (the cost is the 64
atom comparisons, which indexing does not touch) so it barely moves. Typical
clauses (a few head args + body temporaries) land in between. Combined with the
allocation fix the total over the original map+`Value::Atom` baseline is **3.6×**
on `wide/20`.

Correctness: `test_wam_cpp_lowered_{t4,t5,ite_exec}` plus ~20 diverse
backtracking/control e2e subtests (member enumeration, findall/bagof/setof,
maplist, recursive arithmetic, cut, append, reverse) all pass. The full
`test_wam_cpp_generator` suite is simply slow (dozens of programs compiled at
-O0), not changed by this; it needs a longer timeout than the default.
