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
| `bench_functor_read` | OK     | `functor/3` read mode                         |
| `bench_arg_read`     | OK     | `arg/3`                                       |
| `bench_univ_decomp`  | OK     | `=../2` — cons-cell list; this PR             |
| `bench_copy_flat`    | FAIL   | `copy_term/2` — needs term-walking allocator  |
| `bench_copy_nested`  | FAIL   | `copy_term/2` — same                          |
| `bench_sum_small`    | OK     | cross-pred (merged-labels)                    |
| `bench_sum_medium`   | OK     | cross-pred (merged-labels)                    |
| `bench_sum_big`      | OK     | cross-pred (merged-labels)                    |
| `bench_term_depth`   | FAIL   | needs separate ITE-interaction fix (below)    |
| `bench_fib10`        | OK     | cut_ite/jump (this PR)                        |

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

### 1. Cross-predicate label resolution (fixed)

Previously, `compile_wam_predicate_to_llvm/4` emitted per-predicate
`@<pred>_code` and `@<pred>_labels` globals. An instruction like
`call sum_ints/3` inside `bench_sum_big` looked up `"sum_ints/3"` in
`bench_sum_big`'s local label map, didn't find it, and defaulted to
index 0 — silent self-recursion at runtime.

**Fixed** (same architecture WAT adopted in PR #1476):
`compile_predicates_for_llvm/4` now does a two-pass merged-label
compile — all wam-fallback predicates share a single `@module_code`
instruction array and a single `@module_labels` label-PC table built
from every predicate's locals shifted by its cumulative start PC.
Cross-predicate `call` / `execute` now resolve to the right global PC.
Each `@<pred>()` entry function points at the shared globals and calls
`@wam_set_pc(vm, <pred>_start_pc)` before `@run_loop`. For tests that
build their own driver VM, a `@<pred>_start_pc` constant is also
emitted so the driver can seed the PC.

### 2. Unhandled `cut_ite` / `jump` in LLVM IR emission (fixed)

`wam_line_to_llvm_literal/2` had no clause for `cut_ite` or `jump
L_<label>` — the shared `wam_target` emits these for if-then-else.
They used to fall through to a `; TODO:` comment, which counted toward
the `[N x %Instruction]` array size but produced a blank literal. llc
rejected the module with a size mismatch.

**Fixed in this PR's branch.** New instruction tags 31 (`cut_ite`) and
32 (`jump`) with proper LLVM case bodies: `cut_ite` decrements
`cp_count` by 1 (soft cut, preserving outer CPs); `jump` calls
`wam_label_pc` and sets PC. Enables `bench_fib10` (previously excluded).
`bench_term_depth` no longer errors at llc time but still returns 0 —
separate bug below.

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

| Workload             | Builtin the bench uses   | LLVM status                                    |
|----------------------|--------------------------|------------------------------------------------|
| `bench_functor_read` | `functor/3`              | implemented (opcode 26; read mode only)        |
| `bench_arg_read`     | `arg/3`                  | implemented (opcode 27)                        |
| `bench_univ_decomp`  | `=../2`                  | implemented (opcode 28; decompose only)        |
| `bench_copy_flat`    | `copy_term/2`            | still missing — term-walking allocator needed  |
| `bench_copy_nested`  | `copy_term/2`            | same                                           |

### `bench_term_depth` FAIL — separate ITE + register-aliasing issue

With cut_ite/jump in place, `term_depth/2` compiles cleanly through
llc but the runtime returns 0. Probably `put_variable Xn, Ai` in the
LLVM target doesn't create a SHARED heap cell between Xn and Ai —
it puts two independent `Unbound` Value structs in the two register
slots. When a cross-pred `call` binds Ai to the callee's result, Xn
remains unbound, so subsequent reads (e.g. the `>/2` guard in
`term_depth_args`'s ITE) compare against an unbound payload.

`bench_fib10` exercises the same shape but passes — its wrapper only
checks whether `@run_loop` returned `i1 true`, not the computed result.
`bench_term_depth`'s WAM path probably fails a guard that a correct
impl would pass, causing `run_loop → false`. Separate follow-up;
orthogonal to this PR's scope.

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
