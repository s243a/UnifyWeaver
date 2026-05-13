# WAM C Target - Status And Next Steps

Status date: 2026-05-12

Base verified locally:

- `swipl -q -g run_tests -t halt tests/test_wam_c_target.pl`
- `main` at `55b898f9` (`Merge pull request #2046 from s243a/feat/wam-c-lowered-helper-planner`)
- `swipl -q -g run_tests -t halt tests/test_wam_c_effective_distance_benchmark.pl`
- `python3 examples/benchmark/benchmark_effective_distance_matrix.py --scales dev --targets prolog-accumulated,c-wam-accumulated,c-wam-accumulated-no-kernels --repetitions 1`

Active branch:

- `feat/wam-c-lowered-helper-benchmark-wiring`

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
| Lowered helper benchmark wiring | In progress | Active branch adds `c-wam-lowered-helper` and `c-wam-lowered-helper-interpreted` matrix targets over a tiny fact-helper benchmark |

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
  predicates, emitted as native C foreign handlers behind `call_foreign`
  trampolines.
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
| Lowered/native helper functions | Partial/Active | Done | Done | C has constant fact-only native helpers, planner metadata, and active interpreted-vs-lowered matrix wiring. |
| FactSource abstraction | Partial | Partial/less central | Done | C has TSV category-parent loading; generalize beyond category edges as needed. |
| LMDB-backed facts | Partial/Done | Not primary | Done | C has optional eager LMDB loading for UTF-8 key/value category-parent facts and generated effective-distance LMDB wiring; larger artifact layout support remains. |
| Effective-distance benchmark harness | Partial/Done | Done | Done | C is wired into the shared matrix for TSV and LMDB `kernels_on`/`kernels_off`; next gap is larger artifact layouts. |
| Classic-program e2e suite | Partial/Done | Partial/Done | Partial/Done | C now covers generated Fibonacci-style recursion with arithmetic; add Ackermann-style depth only if routine runtime stays acceptable. |
| Memory lifecycle | Partial/Done | Runtime-managed | Runtime-managed | Active branch adds an ASAN lifecycle smoke and fixes stale top-level choicepoints plus indexed `retry_me_else` without an active choicepoint. |
| Instruction layout efficiency | Done | N/A | N/A | C now packs instruction fields into tag-specific payload arms; benchmark larger generated programs if layout becomes performance-sensitive. |

## Recommended Next Branches

### 1. `feat/wam-c-lowered-helper-benchmark-wiring`

Goal: prove the lowered-helper path on a small benchmark surface before
attempting broader helper classes.

Scope:

- Add a narrow generated benchmark or matrix target that exercises C lowered
  fact helpers.
- Keep the comparison against interpreted C and Prolog small enough for routine
  local validation.
- Record lowered/interpreted planning output with the benchmark artifacts.

Status: active; a tiny fact-helper matrix target pair is implemented for
interpreted-vs-lowered WAM-C comparison.

### 2. `feat/wam-c-lowered-helper-body-calls`

Goal: broaden C lowered helpers past fact-only predicates into one small
deterministic body-call shape.

Scope:

- Pick one simple deterministic helper body that Haskell/Rust already lower.
- Keep fallback/interpreter routing explicit through the planner metadata.
- Add executable smoke coverage before expanding to aggregates or multi-result
  helpers.

## Suggested Immediate Next Step

Continue validating `feat/wam-c-lowered-helper-benchmark-wiring`.

The active branch wires a tiny fact-helper benchmark into the existing matrix
surface as separate interpreted and lowered WAM-C targets. The next check is
final validation, then the following branch can broaden lowered helper shapes.
