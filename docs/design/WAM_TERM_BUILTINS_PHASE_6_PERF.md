# Phase 6: Performance investigation and optimization

**Status:** partial — baseline measured, one optimization landed,
full WAM-pipeline benchmarking blocked on a pre-existing codegen
limitation (cross-predicate label resolution). See §5 for what is and
is not possible on current main.

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

- **Cross-predicate label resolution** in both WAM-Rust and WAM-WAT
  (fix `resolve_label/3` + its callers to use a project-level label
  table built in `write_wam_*_project/3`).
- **If-then-else lowering hang** in the canonical WAM compiler on
  hybrid backends — would remove the need for cut-style rewrites of
  benchmark predicates.
- **Rerun perf comparison** once (1) lands: measure `sum_ints` on
  host SWI, WAM-WAT (via wat2wasm + node), and WAM-Rust (via cargo
  test), and tabulate ns/call for each. Report honestly, even if
  native WAM is 10–100x slower.
- **Profile-guided optimization of the `$unify` / `$deref` hot
  paths** in WAM-WAT if the rerun shows they dominate the cost.
