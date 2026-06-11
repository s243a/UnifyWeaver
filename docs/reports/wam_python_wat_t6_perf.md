# T6 first-argument indexing for the VM-dispatched targets (Python, WAT)

The atom-keyed targets (Rust/C++/Go/F#) get a host string `switch`, and the
int-interned targets (Haskell/Scala/Lua) get a host `case`/`match`/hash table.
**Python and WAT are different**: their lowered path does not dispatch through a
host `switch` at all — it goes through a *runtime* mechanism (Python's bytecode
interpreter, WAT's wasm function). This report records how T6 was implemented
for each and the triage benchmark that justified it.

Measurements: average dispatch cost over all N keys + one miss; lower is better.

## Python — runtime `switch_on_constant`

Python's T4 hybrid already lowers every clause to a native `pred_*_cK` function,
but dispatch runs through the interpreter's `try_me_else`/`retry_me_else` chain
(O(n)) and the compiler's first-arg index is **dropped** (left as a
`# SKIP: switch_on_constant …` comment). The Python runtime, however, *already
implements* the `switch_on_constant` instruction: for a bound first argument it
jumps O(1) to the matching clause body; for an unbound one it skips to the
try/retry chain (so enumeration still works); for a bound no-match it fails.

T6 is therefore **emitter-only**: stop dropping the index and emit the real
`("switch_on_constant", {key: label})`, inserting a fresh clause-1 body label
(the index's "default" slot) so every key resolves to a real label — required
because the runtime fails, rather than falls through, on a bound key that is
absent from the table.

Triage (CPython, `Atom.name` string dict vs the linear `isinstance(x,Atom) and
x.name=="…"` chain):

| N   | chain (ns) | dict (ns) | speedup |
|-----|-----------:|----------:|--------:|
| 8   |      198.8 |     109.4 | 1.82×  |
| 64  |     1040.8 |     112.2 | 9.27×  |
| 256 |     3862.8 |     108.5 | 35.59× |

Interpreted Python has no optimiser to recover the dict, so the chain stays
linear — the biggest win of any target.

## WAT — binary search on the atom hash

WAT atoms are keyed by a **sparse i64 hash** (`atom_hash_i64`, in `[0,2³¹-1]`),
so a dense `br_table` does not apply. The lowered T5 path is a linear
`do_get_constant` cascade (O(n)). T6 replaces it with a **binary search**
(O(log n)) on the sorted clause hashes: the lowered function loads A1's
`val_payload` once and navigates an `i64.eq`/`i64.lt_u` tree. Each matched leaf
still runs `do_get_constant` before its body, so a hash collision or a non-atom
A1 whose payload lands on a node is verified and falls through to the 0 return —
semantics identical to the linear cascade.

Triage (wat2wasm + node/V8, linear `i64.eq` chain vs binary search):

| N   | linear (ns) | bsearch (ns) | speedup |
|-----|------------:|-------------:|--------:|
| 8   |         9.1 |          7.8 | 1.16×  |
| 64  |        32.9 |         18.4 | 1.79×  |
| 256 |        97.6 |         14.5 | 6.72×  |

V8 does **not** flatten the linear i64 chain (it stays ~O(n)), so the search is
a real, growing win — modest at the gate (8 clauses) and large for wide
predicates. Hence the same `t6_min_clauses` gate (default 8) as the other
targets: below it, the simpler linear cascade is kept.

## Verdict

Both implemented (gated). With these, **the T6 column is closed for every T5
target** — atom-keyed, int-interned, and VM-dispatched — except LLVM, which
declines on the measured "the optimiser already converts the chain to a switch"
basis (`docs/reports/wam_int_interned_t6_perf.md`).

## Note — deeper arithmetic bug found while testing Python T6 (interpreter fixed)

The Python T6 parity battery surfaced a bug **independent of T6 dispatch**, whose
root cause turned out to be in the **bytecode interpreter**, not just the lowered
emit. `put_structure F, Ai` (and `put_list`) unconditionally bound whatever var
the target register previously held. For `X is Expr` the compiler emits

```
get_variable X1, A2     % save the result var R (head arg) into X1
put_value    X1, A1     % A1 = R   (is/2 arg 1, the result target)
put_structure +/2, A2   % A2 = the expression structure
set_constant ...        % fill it
builtin_call is/2, 2
```

so A1 and A2 both point at R when `put_structure` runs; binding R to the freshly
built expression clobbered A1, and **every `X is Expr` with an unbound X failed**
in the interpreter (it only "worked" for a ground check like `3 is 1+2`).

**Fix** (`WamRuntime.py`): bind the old var only for an **X (temporary) register**
(`reg > _A_MAX`) — a nested sub-term slot created by an earlier
`set_variable`/`unify_variable`, where the bind links the inner term into its
parent (`error(type_error(..),..)`). An **A (argument) register** is a call-output
slot whose prior contents are dead and may alias a live argument, so it is
overwritten without binding. Regression test: `tests/test_wam_python_is_binding.pl`
(unbound `is`, a list-recursive `is`-accumulator, `X = Term` unify, nested
construction, ground check) — all green; the full `test_wam_python_target` suite
is unchanged.

**Still open (separate follow-up):** the *lowered* Python emit mishandles
in-clause structure construction — `parse_wam_text_py` drops `set_*`, and its
write-mode model is heap-consecutive rather than the runtime's `.args`/write-ctx —
so arithmetic-rule clauses remain wrong under `emit_mode(lowered)`. The T6 Python
test therefore exercises fact predicates (`shade/1`) and 2-arg fact remainders
(`tone/2`), which lower correctly, and avoids arithmetic rule bodies. Reconciling
the lowered emit with the runtime's read/write-ctx structure model is tracked
separately.
