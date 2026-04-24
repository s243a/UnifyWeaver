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
| `bench_univ_decomp`  | OK     | `=../2` decompose                             |
| `bench_copy_flat`    | OK     | `copy_term/2` — this PR                       |
| `bench_copy_nested`  | OK     | `copy_term/2` deep copy — this PR             |
| `bench_sum_small`    | OK     | cross-pred (merged-labels)                    |
| `bench_sum_medium`   | OK     | cross-pred (merged-labels)                    |
| `bench_sum_big`      | OK     | cross-pred (merged-labels)                    |
| `bench_term_depth`   | FAIL   | `put_variable` register-aliasing bug (below)  |
| `bench_fib10`        | OK     | cut_ite/jump                                  |

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

### 6. VM-state leak across bench iterations (fixed)

Pre-fix: each call to an `@<pred>()` entry function allocated a fresh
`%WamState` via `wam_state_new`, which malloc'd ~85 KB of supporting
buffers (stack 6 KB + heap 64 KB + trail 6 KB + choice points ~1.5 KB
+ agg_accum 1 KB + state struct). `wam_cleanup` only destroyed the
arena, so the 85 KB leaked per iteration. At 10 000 iterations that
is ~850 MB, which aborts on memory-constrained devices.

Fixed by adding a `@wam_state_free` helper in
`templates/targets/llvm_wam/state.ll.mustache` that frees every
malloc'd buffer plus the state struct itself, and having each
per-predicate entry function emitted by `emit_one_entry_func` call it
after `run_loop` returns. Tests that build their own driver VM
(bypassing the entry function) keep working because they never call
`@<pred>()` — they manage their own state directly.

Two effects of this fix:
  - 10 000-iteration bench runs now complete.
  - ns/call drops ~10-50× across the board, because the previous
    numbers were dominated by malloc pressure, not WAM work. E.g.
    `bench_true` went from ~50 000 ns/call at 1 k iter to ~1 100
    ns/call at 10 k iter. The 10 k numbers are a much more faithful
    "cost of a minimal WAM call" baseline for Phase 1 profiling.

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
| `bench_univ_decomp`  | `=../2`                  | implemented (opcode 28; decompose + compose)   |
| `bench_copy_flat`    | `copy_term/2`            | implemented (opcode 29) — this PR              |
| `bench_copy_nested`  | `copy_term/2`            | same                                           |

### `bench_term_depth` FAIL — put_variable register-aliasing + structural unification

`term_depth/2` compiles cleanly through llc but returns 0 at runtime.
Root cause is `put_variable Xn, Ai` in the LLVM target: it stores two
independent `Unbound` Value structs in Xn and Ai rather than Refs to
a shared heap cell. When a cross-pred `call` binds Ai to the callee's
result, Xn remains unbound, and the next `>/2` guard in
`term_depth_args`'s ITE compares against an unbound payload.

`bench_fib10` exercises the same shape but passes — its wrapper only
checks whether `@run_loop` returned `i1 true`, not the computed result.
`bench_term_depth` has a conditional path that fails a guard under the
broken aliasing, so `run_loop → false`.

**Progress on the fix** (this PR):

  1. `value_equals` is now structural on compounds: same functor
     pointer (functor globals are interned, so pointer equality = name
     equality), same arity, recursively equal args. List cons-cells
     fall out naturally. **Landed.**

  2. Infrastructure helpers added in `state.ll.mustache`:
     `@wam_heap_get`, `@wam_heap_set`, `@wam_deref_value` (follow Ref
     chain), `@wam_get_reg_deref` (get + deref),
     `@wam_bind_reg` (depth-1 Ref write-through). **Landed** but
     currently unused by the main target code — available for future
     Ref-based work.

  3. `%ChoicePoint` extended with `saved_heap_top` (field 11), and
     every CP-push site (`try_me_else`, `begin_aggregate`,
     `wam_foreign_iter_init`) now saves it. `backtrack` rewinds
     `heap_top` to the saved value after `unwind_trail`, so
     put_variable heap pushes from a failed alternative do not leak
     into the next clause's heap region. **Landed.**

  4. Trail initial capacity raised from 256 to 16 384 entries
     (~384 KB per VM). Previously enough for the current Phase 0
     bench corpus, but the Ref-based put_variable landing will double
     trail traffic (one reg entry from the caller plus one heap entry
     from `wam_bind_reg`), so the extra headroom removes trail
     overflow as a blocker. Still no bounds check in
     `wam_trail_binding` / `wam_trail_heap_binding`; if 16 384 turns
     out to be insufficient a realloc path will be needed. **Landed.**

  5. Ref-based `put_variable` + binding-site migration: **attempted
     three times and reverted each time.** Third attempt (with all
     four prerequisites above in place) finally diagnosed the root
     cause, which is **not** in the Ref migration itself but in a
     pre-existing WAM-LLVM register layout issue:

     **Y-registers and X-registers share the same physical slots.**
     `src/unifyweaver/bindings/llvm_wam_bindings.pl:73-85` maps both
     `Yn` and `Xn` to index `n + 15` in the 32-slot `[32 x %Value]`
     register array. Canonical WAM keeps Y-regs in per-call env
     frames; this target flattened them into the X-reg space. As
     long as Y-regs were treated as "just temporaries that happen to
     survive allocate/deallocate", the naive put_variable scheme
     (two independent Unbound structs in both regs) papered over the
     issue — any inner-call trashing of outer's Y-regs got masked by
     the aliasing that wasn't there.

     With Ref-based put_variable, the Y-reg collisions become
     visible: when an inner predicate does `get_variable Y3, A1`, it
     writes the outer caller's Y3 slot. When the outer resumes after
     the call and does `set_value Y3` (e.g. to pass the current
     index to an arithmetic compound), it reads the trashed value.
     `bench_sum_small` works by coincidence because the arg values
     happen to equal the iteration indices (`f(1,2,3)` — arg at
     position I equals I). `bench_sum_medium` breaks because
     `g(2,3)` as an arg makes `arg_value != I`.

     **Proper fix** needs Y-regs to live in per-call env frames,
     with `allocate` reserving Y-slots, `deallocate` popping them,
     and `get_variable Yn` / `set_value Yn` reading/writing the top
     env frame's Y-array. That is a substantial refactor touching
     `allocate`, `deallocate`, every Y-reg access in
     `wam_line_to_llvm_literal`, and probably backtrack's register
     restore. Out of scope for Phase 0 / initial Ref work.

The structural-equals, Ref-aware helpers, CP-heap-top rewinding, and
trail headroom remain useful — they will all be needed when the
Y-reg env-frame refactor lands, after which the Ref-based
put_variable migration can finally succeed.

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
