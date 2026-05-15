# WAM C Target - Status And Next Steps

Status date: 2026-05-14

Base verified locally:

- `swipl -q -g run_tests -t halt tests/test_wam_c_target.pl`
- `main` at `137a693f` (`Merge pull request #2096 from s243a/feat/wam-c-lowered-helper-expanded-body-calls`)
- `swipl -q -g run_tests -t halt tests/test_wam_c_effective_distance_benchmark.pl`
- `python3 tests/test_benchmark_target_matrix.py`
- `python3 -m py_compile examples/benchmark/benchmark_effective_distance_matrix.py examples/benchmark/benchmark_target_matrix.py examples/benchmark/benchmark_common.py`
- `python3 examples/benchmark/benchmark_effective_distance_matrix.py --scales dev --target-sets c-wam-lowered-helper --repetitions 1 --baseline-target c-wam-lowered-helper-interpreted`
- `python3 examples/benchmark/benchmark_effective_distance_matrix.py --scales dev --targets prolog-accumulated,c-wam-accumulated,c-wam-accumulated-no-kernels --repetitions 1`

Active branch:

- `test/wam-c-generated-project-multi-predicate-setup`

This file replaces the older implementation plan. The four original C follow-up
items are now complete on `main`; the remaining work is feature parity with the
more mature hybrid WAM targets, especially Haskell and Rust.

## Completed C Follow-Ups

| Item | Status | Evidence |
|---|---:|---|
| `VAL_LIST` O(1) dispatch via `list_target_pc` | Done | `InstructionPayload.switch_index.list_target_pc`, `INSTR_SWITCH_ON_TERM` list branch, `test_switch_on_term_list_dispatch`, `test_list_target_pc_emission` |
| O(1) predicate resolution | Done | `PredEntry pred_hash`, `wam_register_predicate_hash`, `resolve_predicate_hash`, `test_predicate_hash_registration` |
| Dynamic atom interning | Done | `AtomEntry`, `wam_intern_atom`, `wam_free_state`, executable smoke interns duplicate runtime atom |
| Runtime helpers | Done | `wam_state_init`, `wam_free_state`, `wam_run_predicate`, `test_helpers_generation` |
| Unsupported instruction fail-fast | Done | `unsupported_instruction`, `unsupported_instruction_tokens`, no `(Instruction){0}` fallback |
| Foreign predicate dispatch foundation | Done | `INSTR_CALL_FOREIGN`, `WamForeignHandler`, `wam_register_foreign_predicate`, `wam_execute_foreign_predicate`, executable smoke |
| Native category ancestor kernel foundation | Done | `wam_register_category_parent`, `wam_register_category_ancestor_kernel`, `wam_category_ancestor_handler`, direct and recursive executable smoke |
| File-backed FactSource foundation | Done | `WamFactSource`, `wam_fact_source_load_tsv`, `wam_fact_source_lookup_arg1`, `wam_register_category_parent_fact_source`, executable smoke |
| Streaming integer foreign-result foundation | Done | `WamIntResults`, `wam_int_results_push`, `wam_collect_category_ancestor_hops`, multi-path executable smoke |
| Effective-distance benchmark generator | Done | `generate_wam_c_effective_distance_benchmark.pl`, TSV facts, `kernels_on`/`kernels_off`, generated C runner smoke |
| LMDB-backed FactSource foundation | Done | Optional `WAM_C_ENABLE_LMDB`, `wam_fact_source_load_lmdb`, duplicate-key LMDB smoke, existing lookup/kernel registration reuse |
| Shared kernel detector setup | Done | `detect_kernels/2`, `generate_setup_detected_kernels_c/2`, detected `category_ancestor/4` foreign trampoline project smoke |
| Effective-distance matrix wiring | Done | `c-wam-accumulated`, `c-wam-accumulated-no-kernels`, C kernel-pair delta, `dev` parity smoke against Prolog |
| Effective-distance LMDB fact-storage wiring | Done | `facts_lmdb` generator mode, LMDB seeder/validator, `c-wam-accumulated-lmdb` matrix targets, `dev` parity smoke against Prolog |
| Classic recursive program e2e | Done | Generated Prolog-to-WAM-C Fibonacci smoke, `set_*` instruction support, dereferenced constants/results, call-base choicepoint pruning |
| Packed instruction layout | Done | `InstructionPayload` union keyed by `WamInstrTag`, typed generated initializers, typed runtime dispatch payload reads, switch-table cleanup guarded by switch tags |
| ASAN memory lifecycle smoke | Done | ASAN-generated executable smoke covers switch table setup replacement, indexed clauses, FactSource loading, native foreign kernel dispatch, and repeated top-level calls |
| Lowered fact-only helper prototype | Done | `lowered_helpers(true)` emits native C foreign handlers for constant fact-only predicates plus a `call_foreign` trampoline |
| ASAN availability probe hardening | Done | `asan_available/0` requires a trivial sanitized executable to compile and run before enabling the optional ASAN lifecycle smoke |
| Lowered helper planner metadata | Done | Explicit lowered/interpreted/rejected helper plans, detected-kernel exclusion, generated `lib.c` plan comments, and `report_lowered_helpers(true)` |
| Lowered helper benchmark wiring | Done | `c-wam-lowered-helper` and `c-wam-lowered-helper-interpreted` matrix targets compare a tiny fact-helper benchmark |
| Lowered helper body-call shape | Done | Deterministic alias-to-fact body calls lower through direct native fact-helper dispatch |
| Lowered helper filtered-fact shape | Done | Constant-guarded fact projections lower to native fact rows and the tiny lowered-helper matrix exercises the filtered helper in lowered mode |
| Lowered helper filter rejection metadata | Done | Planner reports explicit rejection reasons for non-constant filter arguments, unsupported comparison guards, multi-goal bodies, and unsupported filter callees |
| Lowered helper comparison-filter shape | Done | Fact-only callee plus one integer comparison guard lowers into native fact rows and the tiny lowered-helper matrix exercises it |
| Lowered helper comparison-filter rejection metadata | Done | Planner reports explicit rejection reasons for unbound comparison guard variables, unsupported expressions, non-integer ordering rows, and no matching rows |
| Lowered helper repeated-variable filters | Done | Constant and comparison filtered helpers preserve repeated callee-variable row constraints with planner and executable coverage |
| Lowered helper empty-result rejections | Done | Planner reports `no_matching_filter_rows` for supported constant filtered helpers whose constraints produce no rows, alongside existing comparison no-match metadata |
| Lowered helper body-call rejection metadata | Done | Planner reports explicit unavailable and non-lowerable callee reasons for exact user-predicate body-call shapes |
| Lowered helper projected body calls | Done | Variable-only reordered body-call projections lower through native fact-helper dispatch with executable coverage |
| Generated multi-predicate setup | In progress | Active branch appends generated predicate code at per-setup base PCs and covers multiple setup functions in one executable |

## Current C Target Baseline

The C target is now a credible small WAM backend:

- Emits a C runtime and predicate setup functions.
- Supports core head/body WAM instructions for constants, variables,
  structures, lists, and unification.
- Supports `set_variable`, `set_value`, and `set_constant` for generated body
  term construction.
- Supports `call`, `execute`, `proceed`, `allocate`, `deallocate`, and
  choice-point instructions.
- Supports `switch_on_constant`, `switch_on_structure`, `switch_on_term`,
  second-argument constant dispatch, and direct list dispatch.
- Uses a tagged `InstructionPayload` union instead of a single wide instruction
  struct.
- Supports basic `builtin_call` delegation for tested builtins:
  `atom/1`, `integer/1`, `is_list/1`, `=/2`, and `is/2`.
- Supports deterministic `call_foreign` dispatch through a C handler registry.
- Can opt into a prototype lowered-helper path for constant fact-only
  predicates, plus same-arity alias bodies that call those native fact helpers,
  emitted as native C foreign handlers behind `call_foreign` trampolines.
- Supports a deterministic native `category_ancestor/4` handler over an
  in-memory category-parent edge table.
- Supports loading category-parent facts from TSV through a small
  `WamFactSource` interface.
- Supports collecting all integer hop results for native `category_ancestor/4`
  through `WamIntResults`, while preserving deterministic first-result handler
  behavior.
- Can generate a small effective-distance C benchmark runner from Prolog facts,
  with both native-kernel and reference DFS modes over TSV category-parent
  facts.
- Can eagerly load category-parent facts from LMDB into the same `WamFactSource`
  edge storage used by TSV, behind the optional `WAM_C_ENABLE_LMDB` build flag.
- Can detect the shared `category_ancestor/4` recursive-kernel shape and emit
  C startup registration plus a `call_foreign` trampoline for generated
  projects.
- Is visible in the shared effective-distance benchmark matrix as TSV and LMDB
  `kernels_on` / `kernels_off` accumulated target pairs.
- Has executable smokes for generated runtime, cross-predicate calls,
  foreign calls, native category ancestor, file-backed facts, streaming native
  results, real multi-clause predicates, structure indexing, `is_list/1`, and
  `=/2`.
- Has an executable smoke for a generated multi-recursive Fibonacci-style
  arithmetic program.

The main limitation is not core WAM execution anymore. It is hybrid-WAM
infrastructure: C lacks lowered/native helper integration and full benchmark
wiring that Haskell and Rust already have.

## Feature Parity Matrix

Legend: `Done` = covered by code/tests; `Partial` = basic support exists but is
missing important target features; `Missing` = no comparable C path yet.

| Capability | C | Rust | Haskell | C gap / next step |
|---|---:|---:|---:|---|
| Project generator | Partial | Done | Done | C writes runtime/lib files, but lacks Cargo/Cabal-like driver/test scaffolding depth. |
| Core WAM interpreter | Done | Done | Done | Keep C focused on executable correctness and memory safety. |
| Structures and lists | Done | Done | Done | C has executable smokes; broaden classic program coverage. |
| Choice points and backtracking | Done | Done | Done | Add deeper classic programs to stress retry/trust paths. |
| First-arg indexing | Done | Done | Done | C has constants, structures, mixed term, and list dispatch. |
| Second-arg indexing | Partial | Partial/Done | Partial/Done | C has constant A2 dispatch; broaden tests if this becomes hot. |
| Predicate dispatch map | Done | Done | Done | C now uses open-addressing hash table. |
| Builtin calls | Partial | Broader | Broader | C has a small builtin set. Add `functor/3`, `arg/3`, `atom_concat/3`, arithmetic comparisons as needed. |
| Aggregates (`findall`/`bagof`/`setof`) | Missing | Present in hybrid/lowered paths | Present in interpreter/lowered paths | Add only after C has enough runtime term-copy and list construction coverage. |
| Negation / control builtins | Partial | Broader | Broader | C likely needs explicit tests for `\+/1`, cut interactions, and if-then-else lowering. |
| Foreign predicate instruction (`CallForeign`) | Partial/Done | Done | Done | C has deterministic handler dispatch plus integer result collection for native kernels. |
| Native recursive kernels | Partial/Done | Done | Done | C has detected `category_ancestor/4` setup and all-hop collection; add more kernel kinds only after runtime support exists. |
| Shared kernel detector integration | Partial | Done | Done | C reuses `recursive_kernel_detection.pl` for `category_ancestor/4`; broaden as more native C kernels land. |
| Lowered/native helper functions | Partial/Done | Done | Done | C has constant fact-only native helpers, planner metadata, interpreted-vs-lowered matrix wiring, body-call helpers, filtered-fact helpers, comparison-filter helpers, rejection metadata, repeated-variable filter hardening, empty-result rejection metadata, and projected body-call helper expansion. |
| FactSource abstraction | Partial | Partial/less central | Done | C has TSV category-parent loading; generalize beyond category edges as needed. |
| LMDB-backed facts | Partial/Done | Not primary | Done | C has optional eager LMDB loading for UTF-8 key/value category-parent facts and generated effective-distance LMDB wiring; larger artifact layout support remains. |
| Effective-distance benchmark harness | Partial/Done | Done | Done | C is wired into the shared matrix for TSV and LMDB `kernels_on`/`kernels_off`; next gap is larger artifact layouts. |
| Classic-program e2e suite | Partial/Done | Partial/Done | Partial/Done | C now covers generated Fibonacci-style recursion with arithmetic; add Ackermann-style depth only if routine runtime stays acceptable. |
| Memory lifecycle | Partial/Done | Runtime-managed | Runtime-managed | Active branch adds an ASAN lifecycle smoke and fixes stale top-level choicepoints plus indexed `retry_me_else` without an active choicepoint. |
| Instruction layout efficiency | Done | N/A | N/A | C now packs instruction fields into tag-specific payload arms; benchmark larger generated programs if layout becomes performance-sensitive. |

## Recommended Next Branches

### 1. `test/wam-c-generated-project-multi-predicate-setup`

Goal: make generated-project smoke coverage less sensitive to setup-function
ordering.

Scope:

- Cover multiple generated WAM predicate trampolines in one executable without
  relying on the last `setup_*` call.
- Preserve existing lowered-helper foreign registration behavior.
- Prefer a focused runtime/setup smoke before changing generated setup layout.

Status: active; generated setup now appends predicate code using `base_pc`
relative labels and a focused executable smoke covers multiple setup functions
in one process.

### 2. `feat/wam-c-lowered-helper-nonreordered-projections`

Goal: decide whether body-call helper lowering should support projection shapes
that bind only a subset of callee variables.

Scope:

- Start with planner metadata for repeated or omitted head/callee variables.
- Avoid runtime support until the binding contract for ignored callee outputs is
  explicit.
- Keep constant filters on the existing materialized `filtered_fact` path.

## Suggested Immediate Next Step

Continue validating `test/wam-c-generated-project-multi-predicate-setup`.

The active branch makes generated predicate setup append code instead of
replacing prior predicate trampolines, and covers the behavior with a focused
multi-predicate executable smoke.
