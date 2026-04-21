# WAM-LLVM Benchmark Suite (Phase 0)

Phase 0 of the WAM-LLVM performance roadmap (see
`docs/design/WAM_LLVM_LESSONS_FROM_WAT.md`). Establishes a baseline
measurement infrastructure for the WAM-LLVM target — mirroring the
WAT-side bench at `examples/wam_term_builtins_bench/` — so Phase 1 profiling
and Phase 2 peephole work have numbers to compare against.

## Scope

Phase 0 does **not** aim for 13/13 workload parity with the WAT suite.
It covers the 8 single-predicate workloads that compile through the current
WAM-LLVM fallback path. The 5 multi-predicate workloads (`bench_sum_*`,
`bench_term_depth`, `bench_fib10`) surface pre-existing WAM-LLVM gaps that
are out of scope here and tracked below.

| Workload             | Status | Notes                                         |
|----------------------|--------|-----------------------------------------------|
| `bench_true`         | OK     | baseline dispatch overhead                    |
| `bench_is_arith`     | OK     | `is/2`                                        |
| `bench_unify`        | OK     | `X = foo(a,b,c), X = foo(a,b,c)`              |
| `bench_functor_read` | OK     | `functor/3` read mode — landed in this PR     |
| `bench_arg_read`     | OK     | `arg/3` — landed in this PR                   |
| `bench_univ_decomp`  | FAIL   | `=../2` — needs a list representation         |
| `bench_copy_flat`    | FAIL   | `copy_term/2` — needs term-walking allocator  |
| `bench_copy_nested`  | FAIL   | `copy_term/2` — same                          |
| `bench_sum_*`        | —      | excluded: cross-pred label bug (see below)    |
| `bench_term_depth`   | —      | excluded: if-then-else lowering gap           |
| `bench_fib10`        | —      | excluded: cross-pred + if-then-else           |

The FAIL rows still produce ns/call timings (the bench harness just records
the returned `0`). Those numbers are still meaningful as dispatch-cost
baselines; they'll get revisited when the underlying correctness bugs are
fixed and the workloads start returning 1.

## Usage

```bash
# From project root:
bash examples/wam_llvm_term_builtins_bench/build_bench.sh
examples/wam_llvm_term_builtins_bench/bench_suite 10000
```

Pipeline:
1. `generate_llvm_bench.pl` produces `bench_suite.ll` via
   `write_wam_llvm_project/3`, then appends C-ABI wrappers.
2. `llc --relocation-model=pic` produces `bench_suite.s`.
3. `clang -O2 -fPIE -pie` links the assembly with `run_bench.c`.

Output goes to stdout plus `bench_suite_results.json` (mirrors the WAT
side's JSON schema so downstream compare tooling can treat them alike).

## Toolchain (verified 2026-04 on Termux aarch64-android)

- `swipl` (SWI-Prolog)
- `llc` (LLVM 21.1.8)
- `clang`

## Blockers discovered during Phase 0

Phase 0 doesn't try to fix these — it documents them so Phase 1 has a clear
punch list.

### 1. Cross-predicate label resolution (silent wrong-answer)

`compile_wam_predicate_to_llvm/4` emits per-predicate `@<pred>_code` and
`@<pred>_labels` globals. An instruction like `call sum_ints/3` inside
`bench_sum_big` looks up `"sum_ints/3"` in `bench_sum_big`'s local label
map, doesn't find it, falls back to index 0 (with a warning), which at
runtime is `bench_sum_big`'s first instruction — silent self-recursion.

Same bug WAT had before PR #1476's project-level label merge. Fix is a
structural refactor: concatenate all predicate instruction arrays into a
single module-level array and resolve labels across the merged table.

### 2. Unhandled `cut_ite` / `jump` in LLVM IR emission

`wam_line_to_llvm_literal/2` has no clause for `cut_ite` or `jump
L_<label>` — the shared `wam_target` emits these for if-then-else.
They fall through to a `; TODO:` comment, which counts toward the
`[N x %Instruction]` array size computed by `length(ResolvedLiterals, _)`
but produces a blank literal. llc rejects the module with
`got type [N-k x %Instruction] but expected [N x %Instruction]`.

Even if the size were correct, the semantics would be wrong — without
real instruction translations, if-then-else would not dispatch correctly.

Fix: add `wam_line_to_llvm_literal` clauses for `cut_ite` and
`jump` that emit proper LLVM instruction literals, analogous to what
`wam_wat_target.pl` emits.

### 3. WASM-variant state-struct mismatch

`write_wam_llvm_wasm_project/3` loads the shared
`templates/targets/llvm_wam/state.ll.mustache` — which assumes 23-field
`%WamState` with pointer types — but WASM's `templates/targets/llvm_wam_wasm/
types.ll.mustache` defines a 20-field struct with i32 offsets. llc rejects
with `invalid getelementptr indices`.

This was the originally-planned Phase 0 path (reuse Node.js harness from
WAT side). It turns out the WASM variant has never been end-to-end
validated. Fix is either: (a) duplicate state.ll.mustache for WASM with
i32-offset semantics, or (b) extend WASM types.ll.mustache with the 3
missing agg_* fields and change state.ll.mustache to be WASM-compatible.
Phase 0 sidestepped by targeting native aarch64-android directly.

### 4. Functor-name collision (fixed here)

Pre-fix: `sanitize_functor_for_llvm/2` mapped every non-alphanumeric char
to `_`, so `+` and `*` both produced the LLVM global `@.fn__` →
`redefinition of global`. Fixed in this PR by switching to a bijective
hex-escape encoding (`_HH`). See `src/unifyweaver/targets/wam_llvm_target.pl`.

### 5. WASM export wrappers ignored predicate instruction array (fixed here)

Pre-fix: `generate_wasm_exports/2` emitted wrappers that called
`run_loop` on a freshly-constructed state with a null instruction array,
not the predicate's actual code. Fixed by delegating to `@<pred>()`
directly. See `src/unifyweaver/targets/wam_llvm_target.pl`.

### 6. Arena growth not reset across bench iterations

The bench wrapper calls `@wam_cleanup()` after each iteration, which is
supposed to rewind the arena. 1000-iteration runs complete; a
10000-iteration run on bench_suite aborts mid-workload. This points to
`wam_cleanup` not fully resetting whatever allocator state accumulates
(a malloc-backed structure not on the arena, or a bump pointer that
doesn't rewind). Out of scope for Phase 0 — for now, use `1000`
iterations, which is enough for order-of-magnitude baseline signal.

## Per-workload correctness failures — root cause identified

The FAIL rows come from the same underlying gap: WAM-LLVM's
`execute_builtin` switch (`compile_execute_builtin_to_llvm/1` in
`wam_llvm_target.pl`) historically handled opcodes 0–25 only — `is/2`,
comparisons, type checks, `=/2`. Anything else hit the `unknown:` label
and returned `ret i1 false` unconditionally.

| Workload             | Builtin the bench uses   | LLVM status          |
|----------------------|--------------------------|----------------------|
| `bench_functor_read` | `functor/3`              | implemented in this PR (opcode 26; read mode only) |
| `bench_arg_read`     | `arg/3`                  | implemented in this PR (opcode 27) |
| `bench_univ_decomp`  | `=../2`                  | still missing — requires list repr |
| `bench_copy_flat`    | `copy_term/2`            | still missing — requires term-walking allocator |
| `bench_copy_nested`  | `copy_term/2`            | same                 |

The WAT target has all four — `$builtin_functor`, `$builtin_arg`,
`$builtin_univ`, `$builtin_copy_term` in `wam_wat_target.pl`. Adding
`functor/3` and `arg/3` to LLVM was mechanical; the remaining two are
larger because they touch runtime data structures that aren't fully
worked out on the LLVM side yet (cons-cell list layout; term-walking
allocator that interacts with the arena's `wam_cleanup` rewind).

### `put_constant` tag fix (landed in this PR)

While implementing `arg/3` we found that `wam_llvm_case('put_constant', …)`
was storing every put_constant operand with tag 0 (Atom) regardless of
source type, because the WAM-text parser dropped the tag info before
reaching the runtime. Fixed by encoding the tag in the upper 16 bits of
op2 (reg_idx fits in the lower 16) on both the parser and runtime sides.
Without this fix `arg/3` had to accept either tag for A1 and trust the
payload; after the fix the strict `tag == 1` (Integer) check works.

The remaining FAIL timings are still meaningful as a *dispatch* cost
baseline — they measure what a builtin call round-trip costs when the
builtin body is `ret i1 false`. Once the bodies land, the delta tells us
the builtin's own cost.

## Relationship to the WAT suite

The WAT suite at `examples/wam_term_builtins_bench/` is the reference —
same `bench_suite.pl` + `bench_term_walk.pl` Prolog source is imported
here, same 13 workload set, matching JSON schema. Once Phase 0 follow-ups
close the blocker list, this directory should report 13/13 OK with
timings directly comparable to the WAT results.
