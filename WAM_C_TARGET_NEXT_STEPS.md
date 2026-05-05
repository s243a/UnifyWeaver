# WAM C Target - Status And Next Steps

Status date: 2026-05-04

Base verified locally:

- `main` at `83b34375` (`Merge pull request #1837 from s243a/feat/wam-c-streaming-foreign-results`)
- `swipl -q -g run_tests -t halt tests/test_wam_c_target.pl`

This file replaces the older implementation plan. The four original C follow-up
items are now complete on `main`; the remaining work is feature parity with the
more mature hybrid WAM targets, especially Haskell and Rust.

## Completed C Follow-Ups

| Item | Status | Evidence |
|---|---:|---|
| `VAL_LIST` O(1) dispatch via `list_target_pc` | Done | `Instruction.list_target_pc`, `INSTR_SWITCH_ON_TERM` list branch, `test_switch_on_term_list_dispatch`, `test_list_target_pc_emission` |
| O(1) predicate resolution | Done | `PredEntry pred_hash`, `wam_register_predicate_hash`, `resolve_predicate_hash`, `test_predicate_hash_registration` |
| Dynamic atom interning | Done | `AtomEntry`, `wam_intern_atom`, `wam_free_state`, executable smoke interns duplicate runtime atom |
| Runtime helpers | Done | `wam_state_init`, `wam_free_state`, `wam_run_predicate`, `test_helpers_generation` |
| Unsupported instruction fail-fast | Done | `unsupported_instruction`, `unsupported_instruction_tokens`, no `(Instruction){0}` fallback |
| Foreign predicate dispatch foundation | Done | `INSTR_CALL_FOREIGN`, `WamForeignHandler`, `wam_register_foreign_predicate`, `wam_execute_foreign_predicate`, executable smoke |
| Native category ancestor kernel foundation | Done | `wam_register_category_parent`, `wam_register_category_ancestor_kernel`, `wam_category_ancestor_handler`, direct and recursive executable smoke |
| File-backed FactSource foundation | Done | `WamFactSource`, `wam_fact_source_load_tsv`, `wam_fact_source_lookup_arg1`, `wam_register_category_parent_fact_source`, executable smoke |
| Streaming integer foreign-result foundation | Done | `WamIntResults`, `wam_int_results_push`, `wam_collect_category_ancestor_hops`, multi-path executable smoke |
| Effective-distance benchmark generator | Done | `generate_wam_c_effective_distance_benchmark.pl`, TSV facts, `kernels_on`/`kernels_off`, generated C runner smoke |

## Current C Target Baseline

The C target is now a credible small WAM backend:

- Emits a C runtime and predicate setup functions.
- Supports core head/body WAM instructions for constants, variables,
  structures, lists, and unification.
- Supports `call`, `execute`, `proceed`, `allocate`, `deallocate`, and
  choice-point instructions.
- Supports `switch_on_constant`, `switch_on_structure`, `switch_on_term`,
  second-argument constant dispatch, and direct list dispatch.
- Supports basic `builtin_call` delegation for tested builtins:
  `atom/1`, `integer/1`, `is_list/1`, `=/2`, and `is/2`.
- Supports deterministic `call_foreign` dispatch through a C handler registry.
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
- Has executable smokes for generated runtime, cross-predicate calls,
  foreign calls, native category ancestor, file-backed facts, streaming native
  results, real multi-clause predicates, structure indexing, `is_list/1`, and
  `=/2`.

The main limitation is not core WAM execution anymore. It is hybrid-WAM
infrastructure: C lacks lowered/native helper integration, LMDB, and benchmark
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
| Native recursive kernels | Partial | Done | Done | C has `category_ancestor/4` all-hop collection; add detector-driven setup later. |
| Shared kernel detector integration | Missing | Done | Done | Reuse `recursive_kernel_detection.pl`; generate C registration stubs. |
| Lowered/native helper functions | Missing | Done | Done | Consider after foreign kernels; C can lower simple fact-only or deterministic predicates. |
| FactSource abstraction | Partial | Partial/less central | Done | C has TSV category-parent loading; generalize beyond category edges as needed. |
| LMDB-backed facts | Missing | Not primary | Done | Haskell is the reference. C now has an ownership model to extend. |
| Effective-distance benchmark harness | Partial | Done | Done | C has a small generated TSV harness with `kernels_on`/`kernels_off`; next gap is full benchmark matrix integration and larger datasets. |
| Classic-program e2e suite | Partial | Partial/Done | Partial/Done | C has targeted smokes; add Fibonacci/Ackermann-style suite like Scala/Elixir. |
| Memory lifecycle | Partial | Runtime-managed | Runtime-managed | C has init/free; needs ASAN/Valgrind CI-style coverage for larger programs. |
| Instruction layout efficiency | TODO | N/A | N/A | Pack `Instruction` fields into a tagged union if runtime footprint matters. |

## Recommended Next Branches

### 1. `feat/wam-c-lmdb-fact-source`

Goal: add LMDB-backed category-parent facts once the file FactSource and
streaming result contracts are stable.

Scope:

- Mirror the TSV FactSource ownership model.
- Load/cursor category-parent pairs from LMDB.
- Keep this behind an optional build/runtime path.

### 2. `feat/wam-c-kernel-detector-setup`

Goal: connect the C target to the shared recursive-kernel detector.

Scope:

- Reuse `recursive_kernel_detection.pl`.
- Generate C registration stubs for `category_ancestor/4`.
- Keep the hand-registration runtime helpers as the low-level API.

### 3. `feat/wam-c-effective-distance-matrix`

Goal: make C visible in the standard effective-distance benchmark matrix.

Scope:

- Reuse the generated TSV harness.
- Add script wiring next to the Rust and Haskell effective-distance paths.
- Compare `kernels_on` and `kernels_off` outputs on shared small datasets
  before scaling up.

### 4. `perf/wam-c-pack-instruction`

Goal: reduce `Instruction` memory footprint.

Scope:

- Replace the current wide struct with a tagged union keyed by `WamInstrTag`.
- Keep setup emission readable.
- Add compile-time or smoke coverage to ensure every instruction arm reads the
  correct union member.

This is lower priority than hybrid parity. Do it only if generated program size
or cache behavior becomes a real bottleneck.

## Suggested Immediate Next Step

Start with `feat/wam-c-lmdb-fact-source`.

The C target now has a TSV-backed effective-distance harness, so the next
highest parity gap is durable fact storage. Use the Haskell LMDB path as the
reference design, keep the C API optional, and preserve the existing TSV
FactSource as the dependency-free fallback.
