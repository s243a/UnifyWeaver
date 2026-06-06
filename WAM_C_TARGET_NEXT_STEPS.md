# WAM C Target - Status And Next Steps

Status date: 2026-06-05

Latest branch verification:

- `investigate/wam-c-candidate-filter-observability` based on `main` at
  `eb5f5353` (`Merge pull request #2801 from
  s243a/investigate/wam-c-candidate-filter-boundary-repeatability`)
- `python3 -m py_compile examples/benchmark/benchmark_wam_c_candidate_filter_threshold_sweep.py tests/test_wam_c_candidate_filter_threshold_sweep.py`
- `python3 tests/test_wam_c_candidate_filter_threshold_sweep.py`
- `python3 examples/benchmark/benchmark_wam_c_candidate_filter_threshold_sweep.py --scales dev --profiles low --thresholds auto,off --repetitions 1 --run-timeout-seconds 120`
- `git diff --check`

Active branch:

- `investigate/wam-c-candidate-filter-observability`

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
| Generated multi-predicate setup | Done | Generated setup appends predicate code at per-setup base PCs and covers multiple setup functions in one executable |
| Lowered helper non-reordered projection metadata | Done | Planner reports explicit rejection reasons for omitted head variables, repeated head variables, and unbound callee variables |
| Lowered helper ignored-output projections | Done | Projected body-call helpers pass singleton callee-local variables as fresh unbound arguments and ignore their returned bindings |
| Lowered helper projection row constraints | Done | Repeated caller-variable projections lower through materialized native fact rows while preserving availability checks |
| Lowered helper projection benchmark | Done | Tiny lowered-helper benchmark covers direct, reordered, ignored-output, and row-constrained projection shapes |
| Lowered helper scaled benchmark workload | Done | `dev` emits 16 normalized rows and `10x` emits 160 rows for the same projection-shape workload, with interpreted and lowered output hashes matching per scale |
| Lowered helper scale regression coverage | Done | `tests/test_wam_c_lowered_helper_scale_regression.py` pins `dev`/`10x` row counts and interpreted/lowered hash parity |
| Lowered helper larger-scale calibration | Done | Self-contained lowered-helper scales no longer require matching `data/benchmark/<scale>` directories; `25x`, `100x`, and `1k` preserve output parity |
| Lowered helper indexed row dispatch | Done | First-argument hash-bucket dispatch preserves unbound-argument fallback and gives `1k` lowered-helper runtime around 5.8x faster than interpreted in local calibration |
| Lowered helper compile/code-size compaction | Done | Compact static row tables plus bucket row-index arrays reduce `1k` generated `lib.c` from 12.8 MB to 2.6 MB and compile time from 151.75s to 0.98s |
| Lowered helper larger-scale regression | Done | `tests/test_wam_c_lowered_helper_scale_regression.py` now promotes `100x` into the focused local lowered-helper parity regression while leaving `1k` as calibration-only because it costs about 31s end to end |
| Lowered helper next-shape selection | Done | Selected term-shape builtin parity as the next narrow helper-adjacent surface and added C `functor/3` runtime support because Haskell and Rust already cover richer structure inspection paths |
| `arg/3` builtin support | Done | Deterministic positive-index argument extraction for structures and lists has direct executable smoke coverage |
| `atom_concat/3` builtin support | Done | Explicit deterministic atom concatenation and split modes have direct executable smoke coverage |
| Generated Prolog builtin parity smoke | Done | Generated WAM-to-C executable smoke exercises `functor/3`, `arg/3`, and `atom_concat/3` together |
| Benchmark-demand scan | Done | `dev`/`10x` accumulated TSV+LMDB C targets and `dev`/`10x` lowered-helper targets preserve output parity; accumulated C kernel path shows no meaningful speedup over no-kernels and is much slower than optimized Prolog at `10x` |
| `transitive_closure2` native kernel | Done | C shared-kernel detector accepts `transitive_closure2`, emits detected setup, registers a native arity-2 foreign handler, and covers direct plus detected-project executable smokes |
| `transitive_distance3` native kernel | Done | C shared-kernel detector accepts `transitive_distance3`, emits detected setup, registers a native arity-3 foreign handler, and covers direct plus detected-project executable smokes |
| `transitive_parent_distance4` native kernel | Done | C shared-kernel detector accepts `transitive_parent_distance4`, emits detected setup, registers a native arity-4 foreign handler, and covers direct plus detected-project executable smokes |
| `transitive_step_parent_distance5` native kernel | Done | C shared-kernel detector accepts `transitive_step_parent_distance5`, emits detected setup, registers a native arity-5 foreign handler, and covers direct plus detected-project executable smokes |
| `weighted_shortest_path3` native kernel | Done | C shared-kernel detector accepts `weighted_shortest_path3`, emits detected setup, registers a native arity-3 foreign handler, and covers direct plus detected-project executable smokes over weighted edges |
| `astar_shortest_path4` native kernel | Done | C shared-kernel detector accepts `astar_shortest_path4`, emits detected setup, registers a native arity-4 foreign handler, and covers direct plus detected-project executable smokes over weighted and direct-distance edges |
| Accumulated runtime edge-index fix | Done | Lazy child indexes for `WamState` and `WamFactSource` remove repeated full-edge scans; `10x` accumulated C targets now run around 0.065-0.071s with output parity versus Prolog at 0.202s |
| Native weighted-kernel float output | Done | C runtime has `VAL_FLOAT`, numeric unification, double weighted/direct edge storage, and executable weighted/A* smokes for fractional 1.5 results while preserving exact-integer outputs as `VAL_INT` |

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
- Supports `builtin_call` delegation for tested builtins:
  `atom/1`, `integer/1`, `number/1`, `var/1`, `nonvar/1`,
  `compound/1`, `is_list/1`, `=/2`, `is/2`, arithmetic comparisons,
  `functor/3`, `arg/3`, and `atom_concat/3`.
- Supports deterministic `call_foreign` dispatch through a C handler registry.
- Can opt into a prototype lowered-helper path for constant fact-only
  predicates, plus same-arity alias bodies that call those native fact helpers,
  emitted as native C foreign handlers behind `call_foreign` trampolines.
- Supports deterministic native `category_ancestor/4`, `transitive_closure2`,
  `transitive_distance3`, `transitive_parent_distance4`, and
  `transitive_step_parent_distance5` handlers over an in-memory edge table,
  plus `weighted_shortest_path3` and `astar_shortest_path4` over in-memory
  weighted edges with exact-integer results kept as `VAL_INT` and fractional
  results returned as `VAL_FLOAT`.
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
| Builtin calls | Partial | Broader | Broader | C has a growing builtin set, including generated-Prolog coverage over `functor/3`, `arg/3`, and `atom_concat/3`; next builtin gaps should be chosen from concrete benchmark demand. |
| Aggregates (`findall`/`bagof`/`setof`) | Missing | Present in hybrid/lowered paths | Present in interpreter/lowered paths | Add only after C has enough runtime term-copy and list construction coverage. |
| Negation / control builtins | Partial | Broader | Broader | C likely needs explicit tests for `\+/1`, cut interactions, and if-then-else lowering. |
| Foreign predicate instruction (`CallForeign`) | Partial/Done | Done | Done | C has deterministic handler dispatch plus integer result collection for native kernels. |
| Native recursive kernels | Partial/Done | Done | Done | C has detected `category_ancestor/4` setup, all-hop collection for that kernel, native transitive closure/distance/parent-distance/step-parent-distance handlers, weighted shortest path, and A* shortest path with integer and fractional result coverage; remaining parity gaps are broader integration details. |
| Shared kernel detector integration | Partial/Done | Done | Done | C reuses `recursive_kernel_detection.pl` for `category_ancestor/4`, `transitive_closure2`, `transitive_distance3`, `transitive_parent_distance4`, `transitive_step_parent_distance5`, `weighted_shortest_path3`, and `astar_shortest_path4`; Haskell and Rust still have broader wrapper/fact-layout integration. |
| Lowered/native helper functions | Partial/Done | Done | Done | C has constant fact-only native helpers, planner metadata, interpreted-vs-lowered matrix wiring, body-call helpers, filtered-fact helpers, comparison-filter helpers, rejection metadata, repeated-variable filter hardening, empty-result rejection metadata, and projected body-call helper expansion. |
| FactSource abstraction | Partial | Partial/less central | Done | C has TSV category-parent loading; generalize beyond category edges as needed. |
| LMDB-backed facts | Partial/Done | Not primary | Done | C has optional eager LMDB loading for UTF-8 key/value category-parent facts, generated effective-distance LMDB wiring, and a matrix path for LMDB-offset reverse CSR row lookup. |
| Effective-distance benchmark harness | Partial/Done | Done | Done | C is wired into the shared matrix for TSV and LMDB parent-only targets plus bounded child-search scan, sorted-array CSR, buffered-pread-drop CSR, and LMDB-offset CSR layout targets. |
| Classic-program e2e suite | Partial/Done | Partial/Done | Partial/Done | C now covers generated Fibonacci-style recursion with arithmetic; add Ackermann-style depth only if routine runtime stays acceptable. |
| Memory lifecycle | Partial/Done | Runtime-managed | Runtime-managed | C has ASAN lifecycle smoke coverage for repeated setup, indexed clauses, fact-source loading, native kernel dispatch, and repeated top-level calls. |
| Instruction layout efficiency | Done | N/A | N/A | C now packs instruction fields into tag-specific payload arms; benchmark larger generated programs if layout becomes performance-sensitive. |

## Recommended Next Branches

### Completed: `feat/wam-c-native-kernel-float-output`

Goal: close the output-type parity gap for weighted/A* native kernels by adding
a C runtime representation for floating-point results.

Evidence:

- `VAL_FLOAT`, `val_float`, and `val_number_from_double` are available in
  the C runtime.
- Weighted and direct-distance edges store `double` weights.
- Weighted shortest path and A* emit exact integers as `VAL_INT` and fractional
  weights as `VAL_FLOAT`.
- Executable smokes cover fractional weighted and A* results.

Reason:

- The accumulated effective-distance runtime blocker is resolved at `10x`; the
  dominant cost was repeated full-edge scans in ancestor traversal, not WAM
  dispatch or native-kernel call overhead.
- Weighted/A* native kernels previously covered only integer results, while
  Haskell and Rust had broader numeric-result surfaces.

### Completed Investigation: `investigate/wam-c-child-search-runtime-sweep`

Goal: measure whether the newer parent-only LMDB and reverse CSR artifact
layouts remain the right defaults at larger category scales, without forcing
full WAM-C query-runner generation and C compilation when only artifact bytes
or narrow lookup timings are needed, then confirm the end-to-end bounded
child-search query cost before promoting child expansion beyond smoke scale.

Implemented so far:

- Reused the existing effective-distance matrix instead of adding a separate
  benchmark script.
- Added WAM-C generator layout profiles for bounded child search over loaded
  parent facts, sorted-array reverse CSR, buffered-pread-drop reverse CSR, and
  LMDB-offset reverse CSR.
- Added `c-wam-child-search-layouts` for scan-vs-CSR smoke comparisons and
  `c-wam-child-csr-layouts` for larger CSR-only comparisons.
- Compile-only matrix rows now report WAM-C artifact byte sizes for TSV, LMDB,
  CSR index/values, and LMDB-offset stores.
- Added `benchmark_wam_c_child_csr_scale_sweep.py` as the routine compile-only
  scale-sweep wrapper for CSR layout artifacts.
- Added an `--artifact-only` path to that wrapper. It reads the benchmark TSVs,
  assigns the same sorted category IDs as the WAM-C generator, writes the
  parent-sorted reverse CSR index/value files, and optionally writes the
  LMDB-offset lookup store. This keeps `50k_cats` and `100k_cats` artifact
  measurements out of the expensive generated-C compile path.
- `dev` layout smoke shows output parity across scan, sorted-array CSR,
  buffered-pread-drop CSR, and LMDB-offset CSR; runtimes were about
  `0.130-0.145s` for the four variants.
- `10x` CSR compile-only smoke builds all three CSR layouts in about
  `0.735-0.811s`; the generated parent TSV is `167,698` bytes, the reverse
  CSR index is `22,528` bytes, reverse CSR values are `15,728` bytes, and the
  LMDB-offset store adds `77,824` bytes.
- The default compile-only scale sweep now covers `10x,1k,5k,10k` and finishes
  locally in roughly half a minute. At `10k`, parent TSV is `1,266,946` bytes,
  reverse CSR index is `118,672` bytes, reverse CSR values are `100,908` bytes,
  and the LMDB-offset store adds `327,680` bytes.
- The artifact-only path reproduces the `10k` generated parent TSV, CSR index,
  and CSR value byte counts without compiling C. Its Python-created LMDB-offset
  directory reports `335,872` bytes including `lock.mdb`, while the data file
  alone is consistent with the prior `327,680` byte observation.
- `50k_cats` and `100k_cats` artifact-only rows currently share the same local
  category-parent graph: generated parent TSV `10,126,909` bytes, reverse CSR
  index `678,752` bytes, reverse CSR values `787,600` bytes, LMDB-offset store
  `1,687,552` bytes, `42,422` parent rows, `196,900` child-parent edges, and
  `84,136` category IDs. Build time was about `0.11s` for sorted-array rows
  and about `0.15s` for LMDB-offset rows.
- Added `benchmark_wam_c_reverse_csr_lookup.py`, which builds the same TSV
  artifacts, generates the shared WAM-C runtime C file, compiles a tiny harness,
  and times the actual `wam_reverse_csr_lookup_children` API without running
  effective distance.
- WAM-C CSR lookup microbenchmarks with `1,000` sampled parent rows and `5`
  timed iterations show sorted-array CSR ahead of LMDB-offset at the measured
  scales. At `10k`, sorted-array median is about `0.78us` per parent lookup,
  pread/drop sorted-array is about `1.08us`, LMDB-offset is about `1.00us`,
  and LMDB-offset pread/drop is about `1.33us`. At the shared
  `50k_cats`/`100k_cats` category graph, sorted-array is about `1.15-1.17us`,
  pread/drop sorted-array about `1.48us`, LMDB-offset about `1.39-1.44us`,
  and LMDB-offset pread/drop about `1.75us` per parent lookup.
- Added `benchmark_wam_c_child_search_runtime_sweep.py`, a small wrapper for
  the generated WAM-C end-to-end child-search target set. It defaults to `dev`,
  one repetition, and a `180s` per-invocation timeout, with parent-only WAM-C
  available as an explicit comparison row.
- Runtime sweep evidence: at `dev`, parent-only WAM-C finishes in `0.003s`
  and emits a different output hash, while all child-search layouts agree on
  `19` rows. Child scan and sorted-array CSR are essentially tied
  (`0.129s` versus `0.128s`); pread/drop CSR is `0.139s`, and LMDB-offset CSR
  is `0.142s`.
- At `10x`, parent-only WAM-C finishes in `0.071s` with `183` rows, but all
  four child-search layouts hit the `180s` timeout. This says the current
  bounded child expansion query shape, not reverse CSR lookup itself, is the
  blocker beyond smoke scale.

Branch conclusion before pruning follow-up:

- The sweep showed full child-search runtime rows should stay smoke-scale until
  query pruning changed.
- The next child-search improvement needed to focus on pruning/query policy,
  not in-memory CSR. The file-backed sorted-array lookup was already cheap in
  the narrow runtime benchmark.
- Parent-only TSV and LMDB targets remain the priority memory structures for
  current effective-distance workloads. Reverse CSR targets are now available
  for future child-path variants and artifact-layout measurements.

### Completed Investigation: `investigate/wam-c-child-search-root-distance-pruning`

Goal: bring WAM-C child-search pruning closer to the F# and Haskell hybrid WAM
kernels by adding the root-distance lower bound that makes bounded child path
exploration viable beyond smoke scale.

Implemented so far:

- Added a per-query `WamBidirectionalDistanceMap` to the generated C runtime.
  It is built by BFS from the requested root over child edges, using the
  attached reverse CSR artifact when available and falling back to the loaded
  parent-edge table when no CSR is attached.
- Parent and child frontier candidates now require
  `next_cost + min_parent_hops_to_root * parent_step_cost <= budget`. Missing
  min-distance entries are pruned, matching the F#/Haskell behavior for nodes
  that cannot route back to the requested root.
- Changed bounded child-search generator defaults from effectively unbounded
  child exploration to `parent_step_cost(1.0)`, `child_step_cost(3.0)`, and
  `child_search_budget(10.0)`. Explicit options still override these values.
- Updated CSR smokes so reverse CSR fixtures contain both the expansion row and
  the root-descendant row required by root-distance calibration.

Evidence:

- `swipl -q -g run_tests -t halt tests/test_wam_c_target.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_c_effective_distance_benchmark.pl`
- `dev` runtime sweep: parent-only and all child-search layouts now agree on
  `19` rows; child CSR, pread/drop CSR, LMDB-offset CSR, and scan complete in
  roughly `0.004-0.005s`.
- `10x` runtime sweep with a `60s` timeout: sorted-array CSR completes in
  `1.460s`, pread/drop CSR in `1.449s`, LMDB-offset CSR in `1.479s`, and
  scan fallback in `4.100s`. The prior child-search timeout is gone, and all
  child-search layouts agree on `187` rows. Parent-only remains faster
  (`0.066s`) and emits `183` rows, so the output mismatch is expected when
  child paths are enabled.

Branch conclusion before cache follow-up:

- The min-distance map was intentionally per-query in this branch. The next
  useful follow-up was root-keyed cache reuse because effective-distance runs
  query the same roots repeatedly.
- The CSR path is now fast enough to be useful at `10x`; the next scalability
  question is query policy and repeated-root reuse, not raw CSR lookup cost.

### Completed Investigation: `investigate/wam-c-root-distance-cache`

Goal: reuse root-distance calibration across repeated effective-distance
queries that share the same root, without adding a preprocessing artifact yet.

Implemented so far:

- Added an opaque `bidirectional_min_distance_cache` pointer to `WamState`.
- The generated C runtime now keeps a root-keyed linked list of
  `WamBidirectionalDistanceMap` entries. `wam_collect_bidirectional_ancestor_hops`
  reuses an existing root map when possible and builds a new one only on a
  cache miss.
- Cache entries are freed by `wam_free_state` and invalidated when category
  parent facts, category IDs, or the attached child CSR artifact change.
- Executable smokes assert that a successful bidirectional query populates the
  cache and that fact/CSR mutations clear it.

Evidence:

- `swipl -q -g run_tests -t halt tests/test_wam_c_target.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_c_effective_distance_benchmark.pl`
- `dev` runtime sweep: all layouts still agree on `19` rows; child-search
  layouts complete in roughly `0.004-0.005s`.
- `10x` runtime sweep with a `60s` timeout: sorted-array CSR completes in
  `0.110s`, pread/drop CSR in `0.101s`, LMDB-offset CSR in `0.102s`, and scan
  fallback in `0.128s`. This is down from about `1.45s` for CSR and `4.10s`
  for scan after root-distance pruning but before root-cache reuse.
- `1k` runtime sweep with a `60s` timeout: parent-only and all child-search
  layouts agree on `580` rows; sorted-array CSR completes in `0.584s`,
  pread/drop CSR in `0.598s`, LMDB-offset CSR in `0.567s`, and scan fallback
  in `0.708s`.
- `5k` runtime sweep with a `60s` timeout: parent-only and all child-search
  layouts agree on `3,224` rows; sorted-array CSR completes in `2.992s`,
  pread/drop CSR in `3.185s`, LMDB-offset CSR in `3.394s`, and scan fallback
  in `3.679s`.

Open measurement:

- Root-distance maps can now be reused within a generated query run, but they
  are still runtime-local. Persisting `min_distance(root,node)` in LMDB or a
  compact artifact could reduce cold-start work for workloads with stable roots,
  but it adds preprocessing time, storage, invalidation complexity, and planner
  surface area. That option should be gated by the cost analyzer rather than
  made a default.
- The next scalability check should run larger child-search sweeps and compare
  root-cache memory growth against the number of distinct roots.

### Completed Investigation: `investigate/wam-c-child-search-scale-ceiling`

Goal: find the next full-generated WAM-C child-search scale ceiling after
root-distance cache reuse, while reporting enough cache-input shape data to
decide whether persisted min-distance artifacts need cost-planner support.

Implemented so far:

- `benchmark_wam_c_child_search_runtime_sweep.py` now appends
  `wam_c_child_search_cache_inputs` rows for each requested scale. The row
  reports root count, distinct category IDs, parent-edge rows,
  article-category rows, maximum root-cache maps, and the worst-case
  `roots * category_ids` distance-entry upper bound.
- Added `--skip-cache-input-summary` for callers that need the wrapper to emit
  only matrix rows.
- Added focused unit coverage for the cache-input summary calculation on a
  tiny fixture with multiple roots, parent edges, and article-category rows.
- WAM-C effective-distance matrix compile-only rows now include build phase
  timings in their message field, so large generated builds can be attributed
  to Prolog generation, LMDB seeding, reverse-CSR offset seeding, or C
  compilation.
- Added `benchmark_wam_c_bidirectional_kernel.py`, a narrow WAM-C runtime
  harness that builds optional reverse-CSR artifacts from benchmark TSVs,
  loads parent edges and category IDs into `WamState`, and times
  `wam_collect_bidirectional_ancestor_hops` over sampled category/root queries
  without generating the full effective-distance WAM-C facts program. The
  output now splits setup into parent TSV loading, parent-edge registration,
  category-ID loading, query TSV loading, reverse-CSR artifact loading, CSR
  attachment, and kernel registration.
- WAM-C category IDs now maintain hash indexes for atom-to-ID and ID-to-atom
  lookup while preserving the existing stable category-ID array for iteration
  and fallback. This removes the duplicate-registration scan during setup and
  avoids scanning all category IDs when CSR child IDs are converted back to
  atoms.
- WAM-C FactSource parent-edge registration now bulk appends loaded edge
  arrays into `WamState`, reusing already-interned edge atoms when the source
  was loaded by the same state and invalidating child indexes/root-distance
  cache once per source instead of once per edge.
- WAM-C atom interning now uses a dynamic hash table that grows and rehashes
  as large benchmark datasets intern tens of thousands of category atoms,
  avoiding long chains in the old fixed 512-bucket table without inflating each
  stack-allocated `WamState`.
- WAM-C effective-distance reverse-CSR generation now builds an assoc from the
  category-ID map before converting category-parent edges to parent/child ID
  pairs. This removes the previous repeated `memberchk/2` scan over all
  category IDs for each edge.

Evidence:

- `10k` runtime sweep with parent-only comparison and a `120s` per-invocation
  timeout completed. Parent-only WAM-C emitted `5,192` rows in `6.058s`, while
  all child-search layouts agreed on `5,262` rows. Sorted-array CSR completed
  in `6.503s`, buffered-pread-drop CSR in `6.327s`, LMDB-offset CSR in
  `6.708s`, and scan fallback in `8.133s`.
- The same `10k` run reported
  `roots=1 category_ids=8247 parent_edges=25227 article_category_rows=10326 max_cache_maps=1 max_distance_entries_upper_bound=8247`.
- `50k_cats` and `100k_cats` currently share the same category-parent graph
  shape for this cache calculation:
  `roots=4054 category_ids=84136 parent_edges=196900 max_cache_maps=4054 max_distance_entries_upper_bound=341087344`.
  Their article-category rows differ: `50k_cats` has `50,000`, and
  `100k_cats` has `84,136`.
- Before assoc-backed category-ID lookup in CSR generation, `10k`
  compile-only for `c-wam-accumulated-child-csr` took `6.750s`, split into
  `generate_s=6.051` and `compile_s=0.699`. `50k_cats` compile-only took
  `437.955s`, split into `generate_s=428.912` and `compile_s=9.042`.
- After assoc-backed category-ID lookup in CSR generation, `10k`
  compile-only for `c-wam-accumulated-child-csr` took `2.032s`, split into
  `generate_s=1.128` and `compile_s=0.904`. `50k_cats` compile-only took
  `17.796s`, split into `generate_s=8.760` and `compile_s=9.036`. Artifact
  bytes stayed the same: `10k` has `category_parent_tsv_bytes=1266946`,
  `reverse_csr_index_bytes=118672`, and `reverse_csr_values_bytes=100908`;
  `50k_cats` has `category_parent_tsv_bytes=10126909`,
  `reverse_csr_index_bytes=678752`, and `reverse_csr_values_bytes=787600`.
- Before assoc-backed category-ID lookup in CSR generation, a constrained
  full-generated `50k_cats` matrix attempt over
  `c-wam-accumulated,c-wam-accumulated-child-csr` produced no matrix rows after
  several minutes. The old compile-only phase split above explains that missing
  row: project generation alone was multi-minute.
- After assoc-backed category-ID lookup in CSR generation, a bounded
  single-target generated `50k_cats` run for
  `c-wam-accumulated-child-csr` timed out at `300s` with no matrix row. Since
  compile-only for that target is now `17.796s`, the remaining full-generated
  ceiling is generated execution, result enumeration, query volume, or runner
  output timing rather than CSR file creation.
- After adding generated-runner root/query caps and stderr timing probes, a
  `50k_cats` run capped at `100` article/root queries completed in `0.172s`
  of generated runtime with `setup_ms=120.323`, `query_ms=38.815`, and no
  result rows. A one-article/all-root run completed `4054` queries in `1.509s`
  of generated runtime with `setup_ms=124.893`, `query_ms=1370.613`, one
  result row, and `first_result_ms=990.534`.
- After pre-indexing generated article-category rows by article, the same
  `100`-query cap completed in `0.165s` of generated runtime with
  `setup_ms=127.881` and `query_ms=21.676`. The same one-article/all-root run
  completed in `0.721s` of generated runtime with `setup_ms=132.850`,
  `query_ms=569.122`, one result row, and `first_result_ms=387.293`. An
  uncapped `10k` `c-wam-accumulated-child-csr` smoke produced the same `5262`
  rows in `5.406s`, down from the earlier `6.503s` sorted-array CSR row.
- After adding generated-runner article/root name filters and stride/offset
  sampling, a sampled `50k_cats` run with `article_stride=1000` and
  `root_stride=100` selected `50` articles and `41` roots. It completed
  `2050` generated queries in `1.549s`, with `setup_ms=130.866`,
  `query_ms=1403.952`, one result row, and `first_result_ms=1141.808`.
- On that same sampled `50k_cats` query set, parent-only WAM-C produced no
  rows with `query_ms=1425.065`. The child-search variants all produced the
  same one-row output: scan fallback took `query_ms=4988.927`, sorted CSR took
  `query_ms=1439.475`, buffered-drop CSR took `query_ms=1433.230`, and
  LMDB-offset CSR took `query_ms=1334.566`.
- A result-capped `50k_cats` child-search run without article/root sampling
  reached `50` result rows after `72,977` generated queries. The CSR storage
  variants produced matching output hashes: sorted CSR took `query_ms=11731.611`,
  buffered-drop CSR took `query_ms=11319.778`, and LMDB-offset CSR took
  `query_ms=10540.272`. The first standalone sorted-CSR result-capped run
  measured `query_ms=10407.918`, so this comparison is enough to show the
  remaining full-matrix cost is per-query traversal volume, not CSR storage
  disagreement.
- After adding generated-runner traversal counters, a sorted-CSR result-capped
  `50k_cats` run reached the same `50` rows after `72,977` queries with
  `query_ms=11049.112`. It visited `72,977` article-category slices, made
  `72,975` parent collector calls, found `626` parent path results, and spent
  `parent_collect_ms=8008.670`. It made `72,932` child collector calls, found
  only `11` child path results, and spent `child_collect_ms=3022.342`.
- After adding cached parent-reachability prefiltering, the same sorted-CSR
  result-capped run reached the same `50` rows after `72,977` queries with
  `query_ms=4718.503`. Parent reachability checks took
  `parent_reachability_ms=2668.182`, but they pruned `72,932` parent DFS calls:
  parent collector calls dropped from `72,975` to `43`, and
  `parent_collect_ms` dropped from `8008.670` to `9.889`. Child collection is
  now the largest remaining measured traversal cost, with `72,932` calls,
  `11` child path results, and `child_collect_ms=2024.048`. An uncapped `10k`
  `c-wam-accumulated-child-csr` smoke still produced `5262` rows with the same
  output hash and improved from `5.406s` to `3.983s`.
- After adding capped child-reachability prefiltering, the same sorted-CSR
  result-capped `50k_cats` run preserved the `8da5f8534aba` output hash and
  reached `50` rows after `72,977` queries with `query_ms=3122.296`.
  Child prefiltering checked the `72,932` parent-pruned pairs, pruned `72,927`
  of them, found `5` plausible child candidates, and took
  `child_prefilter_ms=494.637`. Child collector calls dropped from `72,932`
  to `5`, while preserving the same `11` child path results and reducing
  `child_collect_ms` from `2024.048` to `0.881`.
- After adding per-article candidate-root filtering with a default
  `candidate_filter_min_roots=256`, the same sorted-CSR result-capped
  `50k_cats` run again preserved the `8da5f8534aba` output hash and reached
  `50` rows after `72,977` ordered query pairs, but only entered category
  traversal for `50` pairs. Candidate filtering processed `19` articles,
  marked `52` candidate roots, skipped `72,927` impossible roots, and took
  `candidate_filter_ms=28.418`. End-to-end query time fell to
  `query_ms=541.447`. The default threshold keeps the filter off for one-root
  workloads: uncapped `10k` stayed at `5262` rows with hash `51be51c22aa7`,
  `candidate_filter_articles=0`, and `query_ms=3963.402`.
- After replacing the dense filtered-root loop with a sparse candidate-root
  schedule when no query cap is active, the same result-capped `50k_cats` run
  preserved the `8da5f8534aba` hash and the logical `72,977` query-pair count,
  scheduled `52` candidate roots across `19` articles, and reduced
  `query_ms` to `485.270`. The one-root `10k` guardrail stayed off as
  intended with hash `51be51c22aa7`, `candidate_schedule_articles=0`, and
  `query_ms=3674.656`. The matrix runner exposes
  `--wam-c-candidate-filter-min-roots` so sweeps can tune or disable the
  threshold without regenerating C.
- Broader sampled validation on `50k_cats` with `article_stride=1000` and
  `root_stride=100` selected `50` articles and `41` roots. Forced sparse
  scheduling and dense traversal both produced `1` row with hash
  `e2bde0c720fe`, but dense traversal was faster (`query_ms=25.390`) than
  sparse scheduling (`query_ms=100.436`) because candidate discovery cost
  dominated at only `41` roots. With the then-default
  `candidate_filter_min_roots=256`, the same sample stayed dense by default,
  preserved the hash, and reported `candidate_filter_articles=0` with
  `query_ms=26.435`. This is why the default root threshold is conservative and
  the CLI override exists.
- The generated runner now resolves the default candidate-root threshold through
  `cost_model:resolve_candidate_filter_min_roots/2`, so Prolog workload options
  can declare `candidate_filter_min_roots(N)`, `always`, `off`, or `auto`.
  `auto` currently preserves the measured dense-root ceiling by generating
  `512`, while `UW_WAM_C_EFFECTIVE_CANDIDATE_FILTER_MIN_ROOTS` remains a runtime
  override for sweeps.
- Before category-ID indexing, narrow runtime evidence bypassing full WAM-C
  project generation at `10k` showed `100` warm-cache sampled queries over one
  root taking `8,905.525ms` with sorted-array CSR. Runtime setup was
  `213.149ms`, split into `7.322ms` parent TSV load, `7.269ms` parent-edge
  registration, `198.187ms` category-ID load, `0.053ms` query TSV load, and
  `0.256ms` reverse-CSR load.
- After category-ID indexing, the same `10k` sorted-array CSR narrow row took
  `456.704ms` for `100` warm-cache sampled queries and produced the same
  `17,047` path results and checksum. Runtime setup dropped to `20.257ms`,
  with category-ID load down to `6.108ms`.
- After bulk parent-edge registration, the same `10k` sorted-array CSR narrow
  row took `450.202ms` for `100` warm-cache sampled queries. Runtime setup
  dropped to `13.125ms`, with parent-edge registration down to `0.129ms`.
- After dynamic atom-table rehashing, the same `10k` sorted-array CSR narrow
  row took `438.787ms` for `100` warm-cache sampled queries. Runtime setup was
  `13.080ms`, with parent TSV load at `7.001ms` and category-ID load at
  `5.549ms`.
- The same narrow runner at `50k_cats` with sorted-array CSR, `10` warm-cache
  sampled queries, and one sampled root completed without full source
  generation before category-ID indexing. Runtime setup/loading took
  `23,647.238ms`, split into
  `297.197ms` parent TSV load, `613.616ms` parent-edge registration,
  `22,734.748ms` category-ID load, `0.027ms` query TSV load, and `1.424ms`
  reverse-CSR load. The measured query loop took `0.348ms`, artifact build
  took `0.106s`, and the run produced `10` path results.
- After category-ID indexing, the `50k_cats` sorted-array CSR narrow row
  dropped to `1,150.795ms` setup, split into `309.277ms` parent TSV load,
  `637.056ms` parent-edge registration, `203.145ms` category-ID load,
  `0.024ms` query TSV load, and `1.015ms` reverse-CSR load. The measured query
  loop took `0.042ms` for the same `10` sampled queries and produced the same
  checksum.
- After bulk parent-edge registration, the `50k_cats` sorted-array CSR narrow
  row dropped to `507.700ms` setup, split into `285.517ms` parent TSV load,
  `1.270ms` parent-edge registration, `219.686ms` category-ID load, `0.024ms`
  query TSV load, and `0.977ms` reverse-CSR load. The measured query loop took
  `0.061ms` for the same `10` sampled queries and produced the same checksum.
- After dynamic atom-table rehashing, the `50k_cats` sorted-array CSR narrow
  row dropped to `144.934ms` setup, split into `65.613ms` parent TSV load,
  `1.749ms` parent-edge registration, `76.071ms` category-ID load, `0.019ms`
  query TSV load, and `1.012ms` reverse-CSR load. The measured query loop took
  `0.010ms` for the same `10` sampled queries and produced the same checksum.
- A `dev` LMDB-offset narrow smoke completed with `3` sampled queries,
  `setup_ms=0.172`, `reverse_csr_offsets_lmdb_bytes=32768`, and nonzero path
  results after dynamic atom-table rehashing.

Open measurement:

- Runtime setup is no longer the large `50k_cats` ceiling in the narrow
  runner, and full generated-project compile-only generation is no longer the
  multi-minute ceiling after assoc-backed CSR ID lookup. Generated article-row
  slicing removes the full `ARTICLE_CATEGORY_COUNT` scan from each query. The
  remaining full-runner ceiling is the Cartesian `ARTICLE_COUNT * ROOT_COUNT`
  query product plus per-query WAM/path collection cost; generated-runner
  sampling can now measure that cost without launching the full matrix.
- Evaluate whether a parent-edge artifact or LMDB-backed setup path can avoid
  copying every parent edge into `WamState` when the hot query path uses the
  sorted child CSR plus parent-child index.
- Compare observed root-cache entry counts against the worst-case
  `341,087,344` entry bound before adding persisted `min_distance(root,node)`.
  Persisted distance maps still need to be justified by the cost analyzer
  because they add preprocessing, storage, and invalidation work.

### Completed Investigation: `investigate/wam-c-next-benchmark-demand`

Goal: choose the next C parity gap from an actual benchmark or generated-program
failure instead of adding builtins speculatively.

Evidence:

| Surface | Command | Outcome |
|---|---|---|
| Target registry | `python3 examples/benchmark/benchmark_effective_distance_matrix.py --list-targets` | C exposes accumulated TSV, accumulated LMDB, and lowered-helper target sets. |
| Kernel-pair registry | `python3 examples/benchmark/benchmark_effective_distance_matrix.py --list-kernel-pairs` | C exposes accumulated TSV and LMDB kernels-on/no-kernels pairs. |
| Accumulated C dev parity | `python3 examples/benchmark/benchmark_effective_distance_matrix.py --scales dev --targets prolog-accumulated,c-wam-accumulated,c-wam-accumulated-no-kernels,c-wam-accumulated-lmdb,c-wam-accumulated-no-kernels-lmdb --repetitions 1 --baseline-target prolog-accumulated` | All outputs match across Prolog, TSV C, and LMDB C targets. |
| Lowered-helper dev/10x parity | `python3 examples/benchmark/benchmark_effective_distance_matrix.py --scales dev,10x --target-sets c-wam-lowered-helper --repetitions 1 --baseline-target c-wam-lowered-helper-interpreted` | Lowered and interpreted helper outputs match at `dev` and `10x`. |
| Accumulated C 10x parity and kernel delta | `python3 examples/benchmark/benchmark_effective_distance_matrix.py --scales 10x --targets prolog-accumulated,c-wam-accumulated,c-wam-accumulated-no-kernels,c-wam-accumulated-lmdb,c-wam-accumulated-no-kernels-lmdb --repetitions 1 --baseline-target prolog-accumulated` | All outputs match, but C WAM is about `0.03x` versus optimized Prolog and C kernels are only `1.004x` to `1.014x` versus no-kernels. |

### Completed Shared Kernel Slice: `feat/wam-c-transitive-closure-kernel`

Goal: add the next shared recursive kernel to C from the Haskell/Rust parity
surface.

Evidence:

| Surface | Coverage |
|---|---|
| Kernel support predicate | `wam_c_supported_kernel/1` accepts `recursive_kernel(transitive_closure2, ...)`. |
| Detected setup emission | `generate_setup_detected_kernels_c/2` emits `wam_register_transitive_closure_kernel`. |
| Runtime handler | `wam_transitive_closure_handler` supports bound-target reachability and first-solution unbound target mode over registered in-memory edges. |
| Direct executable smoke | `tc_ancestor/2` succeeds for direct and recursive edges, binds an unbound target, and fails for a reversed edge. |
| Detected-project smoke | Generated project detection lowers `tc_ancestor/2` to a `call_foreign` trampoline and runs through `setup_detected_wam_c_kernels`. |

Performance note:

- `docs/design/WAM_PERF_OPTIMIZATION_LOG.md` repeatedly points to native kernel
  dispatch as the meaningful hybrid-WAM speedup lever. For C, the immediate
  goal should stay on broadening native shared kernels before tuning the general
  WAM instruction dispatch loop.

### Completed Shared Kernel Slice: `feat/wam-c-transitive-distance-kernel`

Goal: add the next measured shared recursive kernel to C from the Haskell/Rust
parity surface.

Evidence:

| Surface | Coverage |
|---|---|
| Kernel support predicate | `wam_c_supported_kernel/1` accepts `recursive_kernel(transitive_distance3, ...)`. |
| Detected setup emission | `generate_setup_detected_kernels_c/2` emits `wam_register_transitive_distance_kernel`. |
| Runtime handler | `wam_transitive_distance_handler` supports bound-target reachability, first-solution unbound target mode, and integer distance binding over registered in-memory edges. |
| Direct executable smoke | `tc_distance/3` succeeds for direct and recursive edges, binds an unbound target and distance, accepts a bound distance, and fails for a reversed edge. |
| Detected-project smoke | Generated project detection lowers `tc_distance/3` to a `call_foreign` trampoline and runs through `setup_detected_wam_c_kernels`. |

### Completed Shared Kernel Slice: `feat/wam-c-transitive-parent-distance-kernel`

Goal: add the next measured wide-output shared recursive kernel to C from the
Haskell/Rust parity surface.

Evidence:

| Surface | Coverage |
|---|---|
| Kernel support predicate | `wam_c_supported_kernel/1` accepts `recursive_kernel(transitive_parent_distance4, ...)`. |
| Detected setup emission | `generate_setup_detected_kernels_c/2` emits `wam_register_transitive_parent_distance_kernel`. |
| Runtime handler | `wam_transitive_parent_distance_handler` supports bound-target reachability, first-solution unbound target mode, parent binding, and integer distance binding over registered in-memory edges. |
| Direct executable smoke | `tc_parent_distance/4` succeeds for direct and recursive edges, binds unbound target/parent/distance, accepts bound parent/distance, and fails for a reversed edge. |
| Detected-project smoke | Generated project detection lowers `tc_parent_distance/4` to a `call_foreign` trampoline and runs through `setup_detected_wam_c_kernels`. |

### Completed Shared Kernel Slice: `feat/wam-c-transitive-step-parent-distance-kernel`

Goal: add the final measured unweighted wide-output shared recursive kernel to
C from the Haskell/Rust parity surface.

Evidence:

| Surface | Coverage |
|---|---|
| Kernel support predicate | `wam_c_supported_kernel/1` accepts `recursive_kernel(transitive_step_parent_distance5, ...)`. |
| Detected setup emission | `generate_setup_detected_kernels_c/2` emits `wam_register_transitive_step_parent_distance_kernel`. |
| Runtime handler | `wam_transitive_step_parent_distance_handler` supports bound-target reachability, first-solution unbound target mode, first-step binding, parent binding, and integer distance binding over registered in-memory edges. |
| Direct executable smoke | `tc_step_parent_distance/5` succeeds for direct and recursive edges, binds unbound target/step/parent/distance, accepts bound step/parent/distance, and fails for a reversed edge. |
| Detected-project smoke | Generated project detection lowers `tc_step_parent_distance/5` to a `call_foreign` trampoline and runs through `setup_detected_wam_c_kernels`. |

### Completed Shared Kernel Slice: `feat/wam-c-weighted-shortest-path-kernel`

Goal: add the first measured weighted shared recursive kernel to C from the
Haskell/Rust parity surface.

Evidence:

| Surface | Coverage |
|---|---|
| Kernel support predicate | `wam_c_supported_kernel/1` accepts `recursive_kernel(weighted_shortest_path3, ...)`. |
| Detected setup emission | `generate_setup_detected_kernels_c/2` emits `wam_register_weighted_shortest_path_kernel`. |
| Runtime handler | `wam_weighted_shortest_path_handler` supports bound-target reachability, first-solution unbound target mode, and integer shortest-weight binding over registered in-memory weighted edges. |
| Direct executable smoke | `weighted_path/3` chooses the lower-cost two-hop route over a higher-cost direct-looking alternative, accepts a bound weight, binds an unbound target/weight, and fails for a reversed edge. |
| Detected-project smoke | Generated project detection lowers `weighted_path/3` to a `call_foreign` trampoline and runs through `setup_detected_wam_c_kernels`. |

### Completed Shared Kernel Slice: `feat/wam-c-astar-shortest-path-kernel`

Goal: add the goal-directed weighted shared recursive kernel to C from the
Haskell/Rust parity surface.

Evidence:

| Surface | Coverage |
|---|---|
| Kernel support predicate | `wam_c_supported_kernel/1` accepts `recursive_kernel(astar_shortest_path4, ...)`. |
| Detected setup emission | `generate_setup_detected_kernels_c/2` emits `wam_register_astar_shortest_path_kernel`. |
| Runtime handler | `wam_astar_shortest_path_handler` requires bound source, target, and integer dimensionality, then binds an integer shortest weight using registered weighted edges and optional direct-distance heuristic edges. |
| Direct executable smoke | `astar_path/4` chooses the lower-cost two-hop route, accepts a bound weight, rejects an unbound dimensionality argument, and fails for a reversed edge. |
| Detected-project smoke | Generated project detection lowers `astar_path/4` to a `call_foreign` trampoline and runs through `setup_detected_wam_c_kernels`. |

## Completed Atom Concat Builtin

The merged `feat/wam-c-atom-concat-builtin` branch added deterministic
`atom_concat/3` composition and split modes before this generated-Prolog smoke
branch.

## Completed Arg Builtin

The merged `feat/wam-c-arg-builtin` branch added deterministic `arg/3`
extraction for structures and lists before this `atom_concat/3` branch.

## Completed Next-Shape Selection

The merged `feat/wam-c-lowered-helper-next-shape-selection` branch selected
term-shape builtin parity and added `functor/3` support before this `arg/3`
branch.

## Completed Larger-Scale Selection

Routine-scale selection from the merged larger-scale regression branch:

| Scale | Normal matrix wall-clock | Rows | Decision |
|---|---:|---:|---|
| `100x` | 8.53s | 1600 | Promoted to local regression |
| `1k` | 30.84s | 4000 | Keep as calibration-only |

## Completed Calibration

Compile/code-size baseline after indexed row dispatch but before compaction:

| Scale | `lib.c` size | Compile real time |
|---|---:|---:|
| `100x` | 5,120,321 bytes | 33.77s |
| `1k` | 12,750,113 bytes | 151.75s |

Compile/code-size result with compact row tables:

| Scale | `lib.c` size | Compile real time |
|---|---:|---:|
| `100x` | 1,123,743 bytes | 0.51s |
| `1k` | 2,601,727 bytes | 0.98s |

Runtime calibration after compaction:

`python3 examples/benchmark/benchmark_effective_distance_matrix.py --scales 25x,100x,1k --target-sets c-wam-lowered-helper --repetitions 3 --baseline-target c-wam-lowered-helper-interpreted`

| Scale | Rows | Output parity | Lowered median | Interpreted median | Lowered speedup vs interpreted |
|---|---:|---:|---:|---:|---:|
| `25x` | 400 | match | 0.002s | 0.003s | 1.25x |
| `100x` | 1600 | match | 0.003s | 0.007s | 2.46x |
| `1k` | 4000 | match | 0.005s | 0.030s | 5.89x |

The compact representation preserves the indexed runtime win while making `1k`
compilation routine-scale again.

## Historical Calibration

Before indexed row dispatch:

`python3 examples/benchmark/benchmark_effective_distance_matrix.py --scales 25x,100x,1k --target-sets c-wam-lowered-helper --repetitions 3 --baseline-target c-wam-lowered-helper-interpreted`

| Scale | Rows | Output parity | Lowered median | Interpreted median | Lowered speedup vs interpreted |
|---|---:|---:|---:|---:|---:|
| `25x` | 400 | match | 0.003s | 0.002s | 0.72x |
| `100x` | 1600 | match | 0.026s | 0.008s | 0.30x |
| `1k` | 4000 | match | 0.179s | 0.030s | 0.17x |

After hash-bucket row dispatch but before compact row tables:

| Scale | Rows | Output parity | Lowered median | Interpreted median | Lowered speedup vs interpreted |
|---|---:|---:|---:|---:|---:|
| `25x` | 400 | match | 0.002s | 0.002s | 1.09x |
| `100x` | 1600 | match | 0.004s | 0.008s | 1.81x |
| `1k` | 4000 | match | 0.005s | 0.031s | 5.79x |

### Completed Investigation: `investigate/wam-c-candidate-filter-calibration`

Goal: make the candidate-root filter threshold calibration repeatable enough to
decide whether `auto` should stay at the current dense-root ceiling or consume
measured workload costs.

Implemented so far:

- Added `benchmark_wam_c_candidate_filter_threshold_sweep.py`, a thin wrapper
  around the effective-distance matrix runner. It sweeps root-count profiles
  (`low`, `medium`, `high-capped`) against threshold policies (`auto`,
  `always`, `off`, or explicit integers) and emits TSV or Markdown summaries.
- The summary extracts generated-runner counters from the matrix message field:
  selected article/root counts, output hash, dense hash agreement, query time,
  candidate-filter time, schedule roots, skipped roots, and category visits.
- Added unit coverage for threshold alias parsing, matrix command construction,
  message parsing, dense-baseline selection, and TSV rendering.
- The generated effective-distance runner now uses `CLOCK_MONOTONIC` for setup
  and query counters, matching the narrower WAM-C timing harnesses and avoiding
  wall-clock adjustments in calibration rows.

Evidence:

- Dry-run over `50k_cats` low/high-capped profiles prints the expected matrix
  commands, with `auto` omitting the runtime override, `always` passing
  `--wam-c-candidate-filter-min-roots 1`, and `off` passing a large threshold.
- Live `dev` low-profile smoke over `auto,always,off` preserved the same
  `e94e9c7a70e3` output hash for all three policies. `auto` and `off` stayed
  dense, while `always` used sparse scheduling and reported
  `candidate_schedule_roots=1`.
- Full `50k_cats` threshold sweep over `low,medium,high-capped` profiles and
  `auto,always,16,64,256,1024,off` preserved output hashes for every row.
  Low selected `50` articles and `41` roots; `auto` stayed dense
  (`query_ms=25.338`) while forced sparse rows were much slower
  (`query_ms=108.324` for `always`, `114.917` for `16`). High-capped selected
  all `4,054` roots and reached the same `50` rows with hash `8da5f8534aba`;
  `auto` stayed sparse with `query_ms=483.756`, while `off` dense traversal
  took `query_ms=4546.297`.
- The medium profile selected `50` articles and `406` roots with hash
  `226c7fdad57d`. One full sweep put sparse `auto` at `query_ms=119.097` and
  dense `off` at `105.482`; a focused 3-repetition medium rerun put `auto` at
  `106.188`, `256` at `108.198`, `512` dense at `100.884`, `1024` dense at
  `110.546`, and `off` at `130.339`. Treat this as the noisy boundary region,
  not as a reason to move the default yet.

### Completed Investigation: `investigate/wam-c-candidate-filter-boundary-repeatability`

Goal: pin the noisy candidate-root filter crossover with repeatable boundary
profiles and change the generated `auto` threshold only when boundary evidence
supports it.

Implemented so far:

- Added boundary calibration profiles around the crossover band:
  `boundary-250`, `boundary-500`, and `boundary-800`.
- Changed the cost-model `auto` dense-root ceiling from `255` to `511`, so the
  generated default `candidate_filter_min_roots` becomes `512`.
- Kept `candidate_filter_min_roots(N)`, `always`, `off`, and the environment
  override available for workloads whose measured crossover differs.

Evidence:

- Dry-run coverage confirms the boundary profiles pass the expected root
  strides into the existing effective-distance matrix runner.
- A 3-repetition `50k_cats` boundary sweep over
  `boundary-250,medium,boundary-500,boundary-800` and `auto,256,512,off`
  preserved output hashes for every row.
- Explicit `512` is the simulated new `auto` threshold on the old build:
  `254` selected roots stay dense, `406` roots stay in the noisy dense-favored
  boundary band, `507` roots are near parity, and `811` roots choose sparse
  scheduling with much lower query time than dense traversal.
- After changing the generated default to `512`, a focused sanity sweep showed
  `medium` (`406` selected roots) staying dense with matching hash
  `226c7fdad57d`, while `boundary-800` (`811` selected roots) used sparse
  scheduling with matching hash `92be2a9b5ac1`.

### Active: `investigate/wam-c-candidate-filter-observability`

Goal: make threshold sweep output self-describing enough that future
calibration rows show both the requested policy and the resolved runtime
threshold.

Implemented so far:

- The candidate-filter sweep summary now fills `threshold_min_roots` for
  `auto` rows from the generated runner's `candidate_filter_min_roots` metric.
- Explicit threshold rows still report the requested override, even when the
  runner message also includes the resolved threshold metric.
- Unit coverage pins `auto -> 512` observability and explicit-threshold
  precedence in the summary renderer.

Evidence:

- The low-profile `dev` sweep over `auto,off` is enough to verify the rendered
  `auto` row reports `threshold_min_roots=512` without rerunning the larger
  boundary matrix.

## Suggested Immediate Next Step

Result-capped and sampled runs now confirm child-CSR variants agree, and parent
plus child reachability prefilters remove most avoidable traversal work inside
each visited article/root pair. Per-article candidate-root filtering plus sparse
candidate-root scheduling now avoids most impossible root traversals when many
roots are selected, while staying off for low-root workloads by default and
preserving dense semantics when an explicit query cap is active. The threshold
default now has a cost-model resolver, a Prolog option surface, and a repeatable
calibration wrapper. The boundary-repeatability sweep supports moving `auto` to
`512`, and the observability follow-up makes future sweep rows display that
resolved value directly. The next useful work is feeding measured query/artifact
costs into the resolver only if future datasets show a sharper crossover or if
manual threshold overrides become common.
Keep `benchmark_wam_c_child_csr_scale_sweep.py --artifact-only` for large
category-graph artifact bytes, and use `benchmark_wam_c_reverse_csr_lookup.py`
only when changing CSR lookup storage. Do not persist root-distance maps to
LMDB or a separate artifact by default; add that only behind a cost-analyzer
decision that can justify the extra preprocessing and invalidation surface.
