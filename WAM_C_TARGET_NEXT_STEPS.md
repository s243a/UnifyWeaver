# WAM C Target - Status And Next Steps

Status date: 2026-05-04

Base verified locally:

- `main` at `00c2cd4e` (`Merge pull request #1829 from s243a/fix/wam-c-remove-zero-instruction-fallback`)
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
- Has executable smokes for generated runtime, cross-predicate calls,
  real multi-clause predicates, structure indexing, `is_list/1`, and `=/2`.

The main limitation is not core WAM execution anymore. It is hybrid-WAM
infrastructure: C lacks the native kernel, FactSource, lowered/native helper,
and benchmark wiring that Haskell and Rust already have.

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
| Foreign predicate instruction (`CallForeign`) | Missing | Done | Done | Highest-value parity gap: add C `INSTR_CALL_FOREIGN` plus registration/dispatch hooks. |
| Native recursive kernels | Missing | Done | Done | Start with one kernel, probably `category_ancestor/4` or `transitive_closure2`, after `CallForeign` exists. |
| Shared kernel detector integration | Missing | Done | Done | Reuse `recursive_kernel_detection.pl`; generate C registration stubs. |
| Lowered/native helper functions | Missing | Done | Done | Consider after foreign kernels; C can lower simple fact-only or deterministic predicates. |
| FactSource abstraction | Missing | Partial/less central | Done | Add simple file-backed FactSource before LMDB. |
| LMDB-backed facts | Missing | Not primary | Done | Haskell is the reference. C should first define FactSource API and ownership model. |
| Effective-distance benchmark harness | Missing | Done | Done | Add after foreign kernels and external fact sources. |
| Classic-program e2e suite | Partial | Partial/Done | Partial/Done | C has targeted smokes; add Fibonacci/Ackermann-style suite like Scala/Elixir. |
| Memory lifecycle | Partial | Runtime-managed | Runtime-managed | C has init/free; needs ASAN/Valgrind CI-style coverage for larger programs. |
| Instruction layout efficiency | TODO | N/A | N/A | Pack `Instruction` fields into a tagged union if runtime footprint matters. |

## Recommended Next Branches

### 1. `feat/wam-c-call-foreign`

Goal: add the minimum foreign predicate abstraction to the C target.

Scope:

- Add `INSTR_CALL_FOREIGN` or reuse an equivalent C instruction shape for
  compile-time-resolved foreign predicates.
- Add a C registry type for foreign handlers keyed by `pred/arity`.
- Add setup emission for a hand-registered foreign predicate.
- Add an executable smoke where a C foreign handler returns one or more
  solutions and the WAM runtime binds output registers.

Why first: Rust and Haskell hybrid WAMs both hinge on `CallForeign`; without it,
C cannot participate in native-kernel or external-source benchmarks.

### 2. `feat/wam-c-category-ancestor-kernel`

Goal: wire one shared recursive kernel into C after `CallForeign` exists.

Candidate kernel:

- `category_ancestor/4`, because it is the effective-distance workload's core
  recursive kernel and has mature Rust/Haskell examples.

Scope:

- Use `recursive_kernel_detection.pl` to detect the predicate.
- Emit C registration stubs and a native handler skeleton.
- Keep facts in a simple in-memory edge table initially.
- Add a small executable smoke with a tiny category graph.

### 3. `feat/wam-c-file-fact-source`

Goal: add the first C FactSource abstraction.

Scope:

- Define a small C interface for `scan`, `lookup_arg1`, and `close`.
- Implement TSV or grouped-by-first file loading.
- Wire a C foreign predicate to a FactSource lookup.
- Add tests that mirror the Scala/Haskell grouped fact-source behavior.

Why before LMDB: ownership, row encoding, and cursor semantics are easier to
settle with a file-backed source.

### 4. `feat/wam-c-effective-distance-bench`

Goal: make C visible in the benchmark matrix once kernels/facts exist.

Scope:

- Add a C effective-distance generator under `examples/benchmark/`.
- Support `kernels_on` / `kernels_off`.
- Start with small TSV/fact-source mode; add LMDB later.

### 5. `perf/wam-c-pack-instruction`

Goal: reduce `Instruction` memory footprint.

Scope:

- Replace the current wide struct with a tagged union keyed by `WamInstrTag`.
- Keep setup emission readable.
- Add compile-time or smoke coverage to ensure every instruction arm reads the
  correct union member.

This is lower priority than hybrid parity. Do it only if generated program size
or cache behavior becomes a real bottleneck.

## Suggested Immediate Next Step

Start with `feat/wam-c-call-foreign`.

It is the smallest feature that moves C toward Rust/Haskell parity. The work
should be split into two commits:

1. Runtime API and instruction support for C foreign handlers.
2. Transpiler emission plus one executable smoke.

Keep LMDB and effective-distance out of that first branch; they depend on this
foreign-call foundation.
