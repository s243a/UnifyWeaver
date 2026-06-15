# rust T9 — fact-table inline (design)

**Status:** IMPLEMENTED (opt-in). Step 1 (detection/extraction), Step 2
(emission: `emit_fact_table_rust/4` + `rust_term_to_value_literal/2`), Step 3
(classification seam in `classify_predicates`, gated by `fact_table_inline(true)`;
`fact_table` arm in `generate_predicate_codes`), and Step 4 (query-mode exec
matrix + emission tests + matrix doc) are landed. Enumeration uses a generic
`fact_table_attempt` runtime method + a `"fact_table"` `resume_builtin` arm. The
table is built once via `OnceLock` with a **first-argument hash index**
(`Value::fact_index_key` keys both rows and the query arg): a bound atomic first
arg selects its bucket in O(1), otherwise a full scan. Within-bucket order is
source order, so the solution sequence matches the T4/WAM path.
**Call-site integration is done:** a WAM `call`/`execute` of a fact-table
predicate is routed (by the Call/Execute instruction handlers) to a generated
crate-level `fact_table_call(vm, pred, cont_pc)` dispatcher, which reads the A
registers and drives the enumerator with the right continuation (`pc+1` for
`call`, the saved `cp` for a tail-call `execute`). So an ordinary predicate can
call a fact-table predicate and backtrack through it (incl. backtracking into the
fact-table choice point from a downstream goal). No remaining T9 follow-ons.

Original design / spec / implementation plan below. Survey of the reference
targets (R / Haskell / Lua) + the Rust target's current fact handling is folded in.

---

## 1. Philosophy

**What T9 is.** Ground unit-clause predicates (`color(red). color(green). …`,
`edge(a,b). edge(b,c). …`) compiled to a native data table + an inlined,
indexed lookup, instead of going through the bytecode WAM interpreter.

**Crucial framing: T9 is an *optimization over the already-correct T4*, not a
correctness gap.** Rust T4 (`multi_clause_n`) already lowers a fact predicate —
each fact is just a clause alternative (`get_constant…/proceed`) with an
iter-style retry choice point — so `color(X)` already enumerates correctly today.
T9's win is **for large fact sets**: N facts under T4 = N clause branches (linear
dispatch, large generated code); under T9 = one compact table + an index
(first-argument `HashMap`), giving O(1)-ish lookup and small code. So T9 should be
**threshold-gated** (only beat T4 above ~N rows) and must **preserve T4's exact
observable behaviour** (all query modes, full backtracking enumeration, order).

**Design principles:**

1. **Don't regress correctness.** T9 must produce the same solutions in the same
   order as the current (T4/WAM) path for every mode: all-ground membership
   check, partially-bound, fully-unbound enumeration.
2. **Prolog-level detection.** Unlike R/Haskell/Lua (which parse WAM *bytecode*
   segments because their pipelines start from WAM text), the Rust target already
   has the source clauses (it uses `clause/2` for the parallel-aggregate
   injection). Detecting "all clauses are ground unit clauses" and extracting the
   tuples is far simpler from source than from bytecode — reuse
   `wam_fact_table.pl:is_fact_predicate/2` + a tuple extractor.
3. **Reuse the enumeration protocol.** The generated lookup must integrate with
   the WAM choice-point/backtrack protocol the same way T4 does (first solution on
   call; `backtrack()` yields the next), so existing callers and the
   par_collect/aggregate machinery keep working unchanged.
4. **Threshold + gate.** Only apply above a row threshold (default ~64–128, like
   scala's auto-inline cutoff), overridable (`t9_min_rows(N)`); below it, leave T4
   to handle it. Optionally behind an opt-in flag initially.
5. **First-argument index.** The common query mode binds A1; index tuples by the
   first argument's principal functor/value (a `HashMap<Key, Vec<RowIdx>>`), with
   a full scan fallback when A1 is unbound.

---

## 2. Specification

### 2.1 Detection
A predicate `p/n` is T9-eligible iff every clause is a **ground unit clause**
(body `true`, all head args ground). Extract `rows = [[c11,…,c1n], …]` (one
ground arg-tuple per clause, in source order). `n ≥ 1`. Eligible only when
`length(rows) ≥ t9_min_rows` (else decline → T4 handles it).

### 2.2 Emission (native function + table)
For `p/n`, emit:
- a `static`/`OnceLock` table of the rows (interned atom ids / integers), plus a
  first-argument index `HashMap<Key, Vec<usize>>`;
- a `pub fn p_n(vm, a1..an) -> bool` that:
  - **A1 bound** → look up candidate rows in the index, unify each with `a1..an`;
  - **A1 unbound** → iterate all rows;
  - on the first unifying row, bind the args and **push a choice point** that, on
    `backtrack()`, resumes the scan at the next candidate row (mirroring T4's
    retry CP — likely a `builtin_state`-style frame with the predicate name + the
    next row index, handled in `resume_builtin`);
  - return `true`/`false` accordingly.
- Order: rows are tried in source order ⇒ same solution order as T4/WAM.

### 2.3 Query-mode correctness (the test matrix)
For `edge(a,b). edge(a,c). edge(b,d).`:
- `edge(a, X)` → X = b, c (in order), via backtrack.
- `edge(X, Y)` → all three pairs, in order.
- `edge(a, c)` → true; `edge(a, z)` → false.
- `edge(X, d)` → X = b (A1 unbound, A2 bound — full scan + unify).
Each must equal the current path's result.

### 2.4 Gating / decline
- Off unless enabled (`fact_table_inline(true)` initially, or always-on above the
  threshold once proven); below `t9_min_rows`, decline.
- Decline for non-ground or non-unit clauses (→ existing path).
- Default output unchanged when disabled.

---

## 3. Implementation plan

In `wam_rust_target.pl`, with the lowered emitter (`wam_rust_lowered_emitter.pl`)
and the fact detector (`wam_fact_table.pl`).

### Step 1 — detection + extraction (pure, tested first)
`rust_fact_table_classify(+Module:Pred/Arity, +Options, -fact_info(Arity, Rows))`:
reuse `is_fact_predicate/2`; collect ground arg-tuples via `clause/2`; enforce
`t9_min_rows`. Unit-test: recognises a fact predicate + extracts rows in order;
declines non-ground / rule-bearing / below-threshold predicates. **No codegen.**

### Step 2 — emission
`emit_fact_table_rust(+Pred/Arity, +fact_info, +Options, -RustCode)` →
`static` table + first-arg index + `pub fn p_n` with CP-based enumeration. Model
the table/index on R's `emit_fact_table/7`; model the choice-point enumeration on
the Rust T4 retry CP (`wam_rust_lowered_emitter.pl` multi_clause_n) +
`resume_builtin`. Interning: reuse the machine's `atom_intern`.

### Step 3 — classification seam
In `classify_predicates/3` (`wam_rust_target.pl` ~line 5560), add the T9 check
**before** `wam_rust_lowerable` so it takes precedence when eligible:
```
( option(fact_table_inline(true), Options),
  rust_fact_table_classify(Module:Pred/Arity, Options, FInfo)
-> emit_fact_table_rust(...), Entry = classified(..., fact_table, ...)
;  ... existing lowered / wam path ... )
```
Add a `fact_table` arm to `generate_predicate_codes/4`.

### Step 4 — exec + regression
`tests/test_wam_rust_fact_table_exec.pl` (cargo): the §2.3 query-mode matrix,
asserting each mode == the current path. Full `test_wam_rust_target.pl` stays
green (disabled ⇒ unchanged). Update the matrix: rust T9 `✗ → ✓` (or `~` if
threshold-gated/opt-in).

### Risk register
- **Backtracking integration** is the crux — get the CP/`resume_builtin`
  enumeration exactly right, else later solutions are lost or order changes. The
  §2.3 matrix is the guard.
- **Atom interning consistency** between the table and query args.
- **Index vs scan** for partially-bound modes — when A1 is unbound, must full-scan
  (don't rely on the A1 index).

### Order of work (each committable)
1. Step 1 detection/extraction (pure + unit tests).
2. Step 2+3 emission + seam (cargo-check a generated project).
3. Step 4 exec query-mode matrix + regression + matrix doc.
