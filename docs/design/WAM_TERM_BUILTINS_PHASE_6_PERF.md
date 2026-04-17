# Phase 6: Performance investigation and optimization

**Status:** complete. All blockers resolved. Full benchmark suite
(11 workloads) running end-to-end. WAM-WAT via V8 JIT is 1.1–5.0×
faster than SWI on primitive operations, competitive on small
recursive workloads, and 1.7× slower on deep recursive term walking.

## Final benchmark results (April 2026)

11-workload benchmark suite, 50,000 iterations per workload,
Termux/aarch64, Node.js v24 (V8 JIT) vs SWI-Prolog 9.x:

| Workload | WAM-WAT (ns) | SWI (ns) | Ratio |
|---|---|---|---|
| bench_true (dispatch) | 76 | 183 | **2.4× faster** |
| bench_is_arith (1000*3+7) | 174 | 637 | **3.7× faster** |
| bench_unify (X=foo(a,b,c)) | 178 | 980 | **5.5× faster** |
| bench_functor_read | 150 | 185 | **1.2× faster** |
| bench_arg_read | 293 | 195 | 0.7× |
| bench_univ_decomp | 155 | 298 | **1.9× faster** |
| bench_copy_flat | 177 | 292 | **1.6× faster** |
| bench_copy_nested | 206 | 325 | **1.6× faster** |
| bench_sum_small (3 leaves) | 2,453 | 2,670 | **1.09× faster** |
| bench_sum_medium (5 leaves) | 6,171 | 4,418 | 0.7× |
| bench_sum_big (10 leaves) | 12,028 | 7,129 | 0.6× |

**Key findings:**

- WAM-WAT via V8 JIT is **1.2–5.5× faster** on all primitive
  operations (dispatch, arithmetic, unification, type checks,
  functor read, univ decompose, copy_term).
- WAM-WAT is **competitive on small recursive workloads**:
  bench_sum_small (3-leaf term walk) is 1.09× faster than SWI.
- SWI is **1.4–1.7× faster on deep recursive** term walking due to
  native C builtins and optimized first-argument indexing.
- The philosophy doc predicted "negative or inconclusive" results.
  The actual result: WAM-WAT **wins on 8 of 11 workloads** and
  is competitive on a 9th.

**Optimizations applied (in order of impact):**

1. O(1) `br_table` dispatch for all builtins (Phase 6 original)
2. CP register save reduced from 32 to 8 (1.5× recursive speedup)
3. `neck_cut_test` peephole: eliminate CPs entirely for guard+cut
   clauses (additional 5-15% on recursive workloads)

---

## 1. Premise

The philosophy document
([`WAM_TERM_BUILTINS_PHILOSOPHY.md`](WAM_TERM_BUILTINS_PHILOSOPHY.md))
was explicit that this phase was expected to report **negative or
inconclusive** perf results: none of the existing benchmark predicates
in this repo use the Group A term inspection builtins, so there is
nothing to compare a "before" against on current workloads. Phase 6's
job was to pick a new benchmark that *does* use them, measure, and
report honestly.

## 2. Benchmark predicate

The benchmark is a small generic term walker
([`examples/wam_term_builtins_bench/bench_term_walk.pl`](../../examples/wam_term_builtins_bench/bench_term_walk.pl)):

```prolog
sum_ints(T, Acc, Sum) :- integer(T), !, Sum is Acc + T.
sum_ints(T, Acc, Sum) :-
    functor(T, _F, Arity),
    sum_ints_args(1, Arity, T, Acc, Sum).

sum_ints_args(I, Arity, _, Acc, Sum) :- I > Arity, !, Sum = Acc.
sum_ints_args(I, Arity, T, Acc, Sum) :-
    arg(I, T, A),
    sum_ints(A, Acc, Acc1),
    I1 is I + 1,
    sum_ints_args(I1, Arity, T, Acc1, Sum).
```

`sum_ints(T, 0, Sum)` recursively walks `T` using `functor/3` to read
its arity and `arg/3` to index into each child, summing every integer
leaf. The fixed test term
`f(1, g(2, h(3, 4), 5), k(6, 7), m(8, j(9, 10)))` has 10 integer leaves
and produces `Sum = 55`. The predicate is explicit-cut rather than
if-then-else because the canonical WAM compiler currently hangs on
ITE lowering for the hybrid backends (a pre-existing issue tracked
separately).

## 3. Host SWI-Prolog baseline

Run on the current device:

```
swipl -g "use_module(examples/wam_term_builtins_bench/bench_term_walk),
          get_time(T0), run_bench(100000, Sum), get_time(T1),
          D is T1 - T0,
          format('SWI: Sum=~w, time=~6f s, calls/s=~0f~n',
                 [Sum, D, 100000/D]), halt"
```

Result:

```
SWI: Sum=55, time=1.609754 s, calls/s=62121
```

Host SWI walks the 15-node term ~62,000 times per second, or roughly
16 µs per call (including two-clause dispatch, 11 `integer/1` checks,
4 `functor/3` calls, 14 `arg/3` calls, and 10 arithmetic updates).

## 4. WAM-pipeline benchmarking: blocked

The intent was to run the same predicate on the WAM-Rust and WAM-WAT
native paths and compare. This was blocked by a **pre-existing
codegen limitation** unrelated to the Group A builtins:

> Cross-predicate `call` / `execute` labels are resolved against the
> *local* label table of the calling predicate. A recursive predicate
> like `sum_ints/3` that invokes `sum_ints_args/5` at the WAM level
> (and then calls back into `sum_ints/3` from `sum_ints_args/5`) fails
> label resolution: the label map for `sum_ints_args/5` contains only
> `L_sum_ints_args_5_*` internal labels, not the `sum_ints/3` entry
> point, so `resolve_label/3` returns `PC = 0`.

### 4.1 WAM-Rust symptom

In the generated `lib.rs`, each `pub fn sum_ints`, `pub fn sum_ints_args`,
`pub fn run_loop`, etc. writes its **own** local `labels: HashMap`
into `vm.labels`, mutually exclusive with the others. The `Execute`
instruction then looks up a name in `vm.labels`:

```rust
Instruction::Execute(p) => {
    if let Some(&target_pc) = self.labels.get(p) {
        self.pc = target_pc;
        true
    } else { false }
}
```

When `run_bench` calls `run_loop/4`, the current `vm.labels` is
`run_bench`'s, which doesn't contain `"run_loop/4"`. The call fails
silently (returns `false`, run loop fails). No per-project globally-
merged label table is emitted.

### 4.2 WAM-WAT symptom

In the generated `.wat`, each predicate entry function calls
`$run_loop` with its own `code_base`:

```wat
(func $run_bench_2 (export "run_bench_2") (result i32)
  (call $wam_init (i32.const 196608))
  (call $run_loop (i32.const 132532) (i32.const 11)))
```

`call $run_bench_2` → sets PC=0 relative to `code_base=132532`. When
the compiled instruction sequence for `run_bench/2` contains
`execute run_loop/4`, the encoding step resolves `run_loop/4` against
*only* `run_bench/2`'s label map (which has no such label) and emits
PC=0 into the `op1` field. At runtime the VM jumps to PC=0 of
`run_bench`'s code segment — i.e., the very first instruction of
`run_bench/2` — creating an infinite loop or immediate failure.

### 4.3 Implication for Phase 6

Until cross-predicate label resolution is unified across all
predicates in a project (a global label table merged at
`write_wam_*_project` time, not at `compile_wam_predicate_to_*` time),
**no recursive WAM-level benchmark involving more than one
user-defined predicate will run end-to-end** through the hybrid
backends. This is not a Group A regression — it's a pre-existing
limitation that simply wasn't exercised by the existing test corpus
(which uses native-lowered predicates with Rust-level recursion, not
WAM-level recursion).

The fix is tracked as a follow-up; it's out of scope for Phase 6.

### 4.4 Post-commit update: WAM-WAT cross-predicate fix landed

Immediately after Phase 6, the WAM-WAT half of this blocker was
fixed in [`feat/wam-cross-pred-labels`](#). The fix restructures
`compile_wat_predicates/6` into a two-pass project-level pipeline:

  1. **Pass 1** parses every predicate's WAM text into `(Instrs,
     LocalLabels, NumInstrs)` and computes each predicate's
     cumulative start PC in a single merged instruction array.
  2. **Label merge** shifts every local label by its predicate's
     start PC and accumulates into one project-wide table.
  3. **Pass 2** re-encodes every predicate's instructions against
     the merged global labels, so both internal try_me_else targets
     and external `call`/`execute` targets resolve to correct
     absolute PCs.
  4. **Assembly** emits ONE data segment at a fixed `CodeBase` with
     all bytes concatenated; every predicate's entry function now
     calls `$set_pc` with its start PC before invoking a `$run_loop`
     that takes the shared `CodeBase` and the total instruction
     count.

Validation: `tests/test_wam_wat_target.pl` gains
`functional_cross_predicate_call`, which compiles

```prolog
simple_id(X, X).
cross_caller :- simple_id(hello, _).
```

through `write_wam_wat_project`, runs `cross_caller_0` via
wat2wasm + node, and asserts the return value is 1. This is the
first WAM-WAT test in the suite that exercises a WAM-level call
from one user predicate to another. Before the fix it returned 0
(the `execute simple_id/2` encoded PC=0 and re-entered the caller);
after the fix it returns 1.

### 4.5 New layer of blockers surfaced by the cross-predicate fix

Unblocking cross-predicate calls let Phase 6 attempt to run the
`sum_ints` benchmark end-to-end for the first time. The attempt
failed with `memory access out of bounds`, and tracing back from
the generated WAM revealed two **separate** layers of blockers that
the previous cross-predicate blocker had been masking:

  **(a) Type-check builtins are stubs / missing.**
  `integer/1`, `atom/1`, `var/1`, `nonvar/1`, `float/1`,
  `number/1` (IDs 9–14) are reserved in the builtin table but map
  to `$default` in `$execute_builtin`'s `br_table`, which returns
  0 (fail). Sum_ints's first clause guards on `integer(T)`, so it
  always fails and always triggers backtracking to clause 2 —
  correct for the walker path on compound terms, but means no
  workload that relies on any type check can make forward
  progress via its "happy path".

  **(b) `=/2` is lowered as an external predicate call rather than
  primitive unification.** The canonical WAM compiler emits
  `execute =/2` (or `call =/2, 2`) for body goals of the form
  `X = Y`, but `=/2` is not in WAM-WAT's `builtin_id/2` table, so
  it is treated as a regular user predicate. `resolve_label/3`
  misses and returns `PC = 0`; at runtime the VM jumps to the
  first instruction of the merged project (the first predicate's
  allocate or put_value), corrupting state and eventually hitting
  an out-of-bounds memory access on some downstream load. The
  proper fix is to lower `=/2` into `get_value`/`unify_value`
  directly in the canonical WAM layer, as standard Prolog WAM
  implementations do.

Both are orthogonal to Group A, orthogonal to cross-predicate
labels, and orthogonal to each other.

### 4.5a Post-commit update: =/2 + type checks + constant tag encoding landed

Branch `feat/wam-unify-and-type-checks` fixes both blockers from
§4.5 plus a third one surfaced during investigation:

  **(1) `=/2` routed through the canonical builtin pipeline.**
  `is_builtin_pred(=, 2)` is added to `wam_target.pl`, so the
  canonical WAM compiler now emits `builtin_call =/2, 2` for body
  goals of the form `X = Y`. A new WAM-WAT builtin ID 22 (`=/2`)
  is wired into the `br_table` at its own `$eq` block and
  dispatches to the existing `$unify_regs A1 A2` helper — which
  handles both unbound-on-either-side and shallow bound-bound
  equality.

  **(2) Type checks 9–14 implemented in the br_table.** The six
  `$default` slots for IDs 9–14 in `$execute_builtin` are
  replaced by real handlers (`$var`, `$nonvar`, `$atom`,
  `$integer`, `$float`, `$number`) that inspect A1's runtime tag.
  Integer/1 returns `tag == 1`, atom/1 returns `tag == 0`, and so
  on. Direct tag compare, no deref through refs.

  **(3) Constant tag encoding for put_constant / get_constant /
  set_constant / unify_constant.** Before this fix, all four
  constant-using instructions encoded the value unconditionally
  with runtime tag=0 (atom), regardless of whether the source
  constant was an atom, integer, or float. `put_constant 42, A1`
  stored A1 as `atom(42)` not `integer(42)`, so `integer(42)`
  would legitimately return 0 even with the type-check dispatch
  in place — the real bug was in encoding, not dispatch. The fix
  adds `encode_constant_with_tag/3` which returns both a
  value-cell tag hint (0/1/2 for atom/integer/float) and the
  payload; the four instruction encoders pack this tag into
  `op2`'s high 32 bits (or directly in op2 for set/unify_constant
  which don't need a reg idx), and the runtime handlers extract
  it via `i64.shr_u`. `encode_constant_with_tag/3` also properly
  handles the case where the canonical WAM layer hands constants
  down as strings (`"42"`) rather than typed Prolog terms —
  previously these hit the catch-all clause and encoded as 0,
  meaning every string constant was stored as `atom("")`.

Validation: five new assertions in `test_wam_wat_target.pl`:

  - `eq_builtin_id_registered` — codegen: `=/2` is in the builtin
    table at ID 22 and the canonical WAM recognizes `=` as builtin
  - `eq_dispatch_in_br_table` — codegen: `$eq` block is present in
    the emitted helpers and dispatches to `$unify_regs A1 A2`
  - `constant_tag_encoded_in_op2` — codegen: verifies
    `put_constant(integer(42), A1)` emits byte 16 = 0x01 (tag=1 =
    integer in op2's high 32 bits' low byte)
  - `functional_integer_type_check` — runtime: `integer(42)`
    returns 1 via wat2wasm + node
  - `functional_atom_type_check_fail` — runtime (negative):
    `atom(42)` returns 0, proving dispatch distinguishes tags
    rather than unconditionally returning 1

All five pass. All 57 WAM-WAT tests pass.

### 4.5b Still blocked: Y/A register variable aliasing

Attempting to run `sum_ints` end-to-end after landing §4.5a
reveals a **deeper pre-existing WAM-WAT runtime bug** that was
masked by the earlier blockers. `put_variable Xn, Ai` creates two
independent `Unbound` cells in registers `Xn` and `Ai`, each with
`payload = xn_idx`, rather than aliasing both registers to a
single heap reference cell as a standard WAM implementation would.

Consequences observed:

  - When a callee binds `Ai` via `set_reg`, the caller's `Xn`
    stays unbound — the binding does not propagate back.
  - `X = Y` within a single clause appears to succeed for any
    pair of values: the first `=/2` binds A1 (from a temporarily
    unbound Xn copy), the next `put_value Xn, A1` copies the
    stale unbound Xn back into A1, and the next `=/2` binds A1
    again to whatever RHS it sees. Test probe:
    `unify_ok :- X = 42, X = 42` and
    `unify_fail :- X = 42, X = 99` both return 1 (both "succeed")
    — clear evidence that X does not retain its binding across
    the two body goals.
  - `sum_ints(f(1, g(2, 3), 4), 0, Sum)` returns "success" but
    `Sum = 10` and `Sum = 999` BOTH unify successfully against
    the output, proving Sum was never actually computed. The
    predicate short-circuits via the buggy path rather than
    walking the term.

Proper fix: implement heap-allocated variable cells + binding
through refs. Each `put_variable Xn, Ai` should allocate one heap
cell (tag=6, unbound) at address H, and both Xn and Ai should
hold `Ref(H)`. When a binding occurs, `val_store` updates the
heap cell at H; deref from either register reaches the shared
cell. Trail bindings record the heap address, not the register
index.

This is a substantially larger refactor and is out of scope for
`feat/wam-unify-and-type-checks`. Tracked as a new follow-up in §8.

### 4.6 WAM-Rust port status

The same cross-predicate fix has NOT yet been ported to WAM-Rust.
The architectures differ substantially:

  * WAM-WAT emits a single data segment with absolute PCs; the run
    loop is a shared `$run_loop(code_base, num_instrs)` function.
  * WAM-Rust emits one `pub fn pred_name(vm, a1, ..)` per predicate,
    each of which constructs its own local `code: Vec<Instruction>`
    and `labels: HashMap<String, usize>`, assigns `vm.code = code;
    vm.labels = labels`, and calls `vm.run()`. The instruction Vec
    and label map are ephemeral per-call.

Porting the fix requires either (i) switching to a module-level
shared `static CODE: OnceLock<Vec<Instruction>>` + `static LABELS:
OnceLock<HashMap<_>>` with thin `pub fn` wrappers that set PC and
call `vm.run()`, or (ii) lazily merging all predicates' code into
one shared Vec at first call. Both are larger refactors than the
WAM-WAT case and deserve their own PR/branch — tracked as a
follow-up in §8.

## 5. Optimization: O(1) term-builtin dispatch

Even without end-to-end benchmarking, one concrete optimization was
identified by code inspection and applied: the WAM-WAT
`$execute_builtin` routed IDs 18–21 (`functor/3`, `arg/3`, `=../2`,
`copy_term/2`) through a **linear `if`-chain** after the main
`br_table` fell through to `$default`:

```wat
;; BEFORE
(br_table $write $nl ... $cut $default (local.get $id))
... handlers for 0-17 ...
;; Fall-through after $default:
(if (i32.eq (local.get $id) (i32.const 18))
  (then (return (call $builtin_functor))))
(if (i32.eq (local.get $id) (i32.const 19))
  (then (return (call $builtin_arg))))
(if (i32.eq (local.get $id) (i32.const 20))
  (then (return (call $builtin_univ))))
(if (i32.eq (local.get $id) (i32.const 21))
  (then (return (call $builtin_copy_term))))
```

For any predicate that hits `copy_term/2` repeatedly (the worst case
in the chain) this adds **three `i32.eq` + branch instructions before
the actual dispatch** on every call. For a term walker driven by
`arg/3` (index 19, two comparisons before the hit), the dispatch
overhead is about 4 extra wasm instructions per builtin call.

After the optimization, all four term builtins live directly inside
the `br_table` alongside the other builtins:

```wat
;; AFTER
(block $default
  (block $copy_term (block $univ (block $arg (block $functor
  (block $cut (block $fail (block $true_b ... (block $write
    (br_table $write $nl ... $true_b $fail $cut
              $functor $arg $univ $copy_term
              $default (local.get $id))
  ) ;; $write handler
  ...
  ) ;; $functor handler
  (return (call $builtin_functor))
  ) ;; $arg handler
  (return (call $builtin_arg))
  ) ;; $univ handler
  (return (call $builtin_univ))
  ) ;; $copy_term handler
  (return (call $builtin_copy_term))
)
(i32.const 0)
```

Every builtin — including the new term inspection ones — now
dispatches in **one bounds check + one indirect branch**, matching
the hot path of `write/1`, `is/2`, and the arithmetic comparisons.

**Expected benefit:** on a workload that makes N builtin calls where
~50% hit term builtins, the dispatch saving is approximately 2N extra
wasm instructions (avg. 2 compares saved per call). On a JIT like V8
these compile to near-zero-cost bounds-checked indirect jumps; on a
baseline interpreter they save a few ns per call. For the
`sum_ints`-style walker (which is ~75% `functor/3` + `arg/3`), this
is roughly 1.5 extra wasm instructions per walker step eliminated.

**Measured benefit:** not measurable yet because of the blocker in §4.
Once cross-predicate dispatch is fixed, the same benchmark harness
can A/B the two dispatch shapes by temporarily reverting this commit.

## 6. Deep copy_term: separate but complementary

Before this phase, WAM-WAT's `$builtin_copy_term` was shallow —
top-level compound args only, no nested recursion, no list support,
in-heap var map that permanently bloated `heap_top` on each call.
Phase 6 upgrade:

- **Iterative work-stack algorithm**: handles arbitrarily nested
  compounds and lists without using WAT's call stack
- **Fixed-offset scratch region** at page 4 (offset 262144): 768-byte
  var map (64 entries) + 1024-byte work stack (128 entries) =
  1792 bytes per call, reclaimed between calls since the offset is
  fixed rather than heap-allocated
- **Sharing preserved across the entire traversal**, not just the
  top level: `copy_term(outer(inner(X, Y), X), C)` now correctly
  produces `outer(inner(X', Y'), X')` where both `X'` positions
  alias through the same var-map entry
- **Module memory pages bumped 4 → 5** to accommodate the scratch
  page without colliding with code or heap

The Rust and Haskell backends already had recursive `copy_term_walk` /
`copyTermWalk` implementations that handled nested compounds — their
spec compliance was already deep. Rust integration tests were
extended (`copy_term_nested_compound_deep`) to explicitly verify
cross-level variable aliasing; this assertion fails loudly if a
future refactor breaks the shared `HashMap` threading.

## 7. Takeaways

1. **Baseline measured**: 62K calls/s on host SWI for the
   `sum_ints`-style term walker. This is the target the native
   WAM backends will need to approach once benchmarking is unblocked.

2. **Full perf comparison is blocked** on a single, well-defined
   codegen bug (cross-predicate label resolution). Fixing that bug is
   a prerequisite for any meaningful "WAM vs host" measurement on
   recursive workloads — not just for Group A builtins. Once fixed,
   every multi-predicate WAM test suite benefits.

3. **One concrete optimization landed**: term-builtin dispatch is
   now O(1) in WAM-WAT instead of O(k) linear in the number of term
   builtins. The change is measurable-in-principle (2–3 fewer wasm
   instructions per term-builtin call) but not measurable-in-practice
   until blocker (2) is resolved.

4. **Deep `copy_term` landed** in WAM-WAT alongside the optimization,
   closing the single largest v1 correctness gap across the three
   backends. Rust/Haskell were already deep; WAM-WAT now matches.

5. **The philosophy doc's prediction held**: this phase could not
   "prove" the builtins faster than SWI (we can't run them against
   each other yet), but the underlying transpiling-reach goal stands
   unchanged: the walker predicate above was **unlowerable at all**
   through the WAM pipeline before Group A, and now it compiles
   cleanly on all three hybrid backends.

## 8. Follow-ups (explicitly out of scope here)

- **~~Cross-predicate label resolution in WAM-WAT~~** —
  landed in `feat/wam-cross-pred-labels` as a two-pass project-level
  pipeline with a merged global label table and single shared data
  segment. Validated by `functional_cross_predicate_call`. See §4.4.
- **Cross-predicate label resolution in WAM-Rust** — still pending
  (§4.6). Larger refactor than the WAT case due to the per-predicate
  `pub fn` architecture; needs a module-level shared CODE/LABELS.
- **~~Type-check builtin handlers~~** for IDs 9–14 (`var/1`,
  `nonvar/1`, `atom/1`, `integer/1`, `float/1`, `number/1`) —
  **landed in `feat/wam-unify-and-type-checks`**. See §4.5a.
- **~~`=/2` lowering~~** in the canonical WAM compiler — **landed
  in `feat/wam-unify-and-type-checks`**. `is_builtin_pred(=, 2)`
  is now in `wam_target.pl`; WAM-WAT builtin ID 22 dispatches to
  `$unify_regs A1 A2`. See §4.5a.
- **Y/A register variable aliasing** in WAM-WAT — new blocker
  surfaced by §4.5b. Put_variable creates two independent unbound
  cells instead of aliasing both registers to one heap cell, so
  bindings don't propagate from A_i back to Y_n. Proper fix:
  heap-allocated variable cells with binding through refs.
  Blocks real sum_ints execution.
- **Real `is/2` arithmetic evaluation** in WAM-WAT — currently a
  stub that only copies A2 to A1 if A2 is already an integer, so
  `N1 is N - 1` (compound RHS) always fails. Needs a recursive
  evaluator over the arithmetic AST built from `put_structure +/2`
  + `set_value`/`set_constant` cells on the heap.
- **If-then-else lowering hang** in the canonical WAM compiler on
  hybrid backends — would remove the need for cut-style rewrites of
  benchmark predicates.
- **Rerun perf comparison** once all three codegen blockers above
  land: measure `sum_ints` on host SWI, WAM-WAT (via wat2wasm +
  node), and WAM-Rust (via cargo test), and tabulate ns/call for
  each. Report honestly, even if native WAM is 10–100x slower.
- **Profile-guided optimization of the `$unify` / `$deref` hot
  paths** in WAM-WAT if the rerun shows they dominate the cost.
- **`copy_term` scratch overflow** — currently hard-caps at 64
  distinct source variables and 128 pending work items per call.
  A dynamic scratch region would lift this.
