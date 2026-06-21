# Rust ⇄ F# Parity Campaign — Brief & Session State

Written 2026-06-11 at the end of the M137–M152 correctness session
(session: claude/jolly-ride-8ccdh7), to survive context compaction.
Read together with `wam_correctness_campaign_handoff.md` (bug classes,
probe battery, methodology, container toolchain map — do not duplicate
that knowledge here).

## Mission (owner's words, distilled)

1. Inventory what the **F# hybrid WAM target** has that the **Rust
   hybrid WAM target** lacks — *especially kernels and graph-search
   algorithms* — and port the valuable parts to Rust.
2. Once Rust reaches parity with F#, look at what *else* should be
   implemented in Rust (graph-search related) based on the latest
   commits, theory, and design documentation.
3. Non-graph-search feature gaps between the two targets are also in
   scope.

## Starting pointers (verified to exist as of writing)

- `docs/WAM_TARGET_ROADMAP.md` — per-target feature matrix. Known rows:
  Rust = "hand-tuned FFI kernel route; effective-distance matrix FFI
  kernel; port the kernel pattern to other targets; add layout
  policies / FactSource generalisation; LMDB via lmdb-zero (R1)".
  Check the F# row freshly — recent F# PRs added ISO arith-compare
  sweep, succ/2 family, reverse-append builtins, lowered-emitter
  Phase-I coverage, query smoke work.
- `docs/handoff/wam_rust_simplewiki_blocker.md` — a documented Rust
  graph-workload blocker; read FIRST, it may gate everything.
- `docs/handoff/wam_haskell_enwiki_benchmark_handoff.md` — benchmark
  shapes used for large-graph work (Haskell, but the data pipeline and
  workload definitions transfer).
- Graph-search theory/design recency: `docs/reports/` (reverse_csr_*,
  bidirectional_kernel_simplewiki_benchmark, depth_likeness_budget,
  topical_geometric_regime, distribution_cache_simplewiki_depth_grid
  2026-06-10), `docs/design/DISTRIBUTION_CACHE_BENCHMARK_PLAN.md`,
  `docs/design/QUERY_PLAN_RUNTIME_PHILOSOPHY.md`.
  Canonical simplewiki data: `examples/benchmark/parse_simplewiki_dump.py`
  → SQLite → `data/benchmark/simplewiki_articles/category_parent.tsv`
  (+ `root_categories.tsv`); sampling via
  `scripts/sample_distribution_cache_subtree.py`.
- Rust already has SOME kernels (don't re-port): the M151 e2e exercises
  `tc_ancestor/2`, `tc_distance/3`, `tc_parent_distance/4`,
  `tri_sum/2`, `weighted_path/3`, `astar_weighted_path/4`,
  `min_semantic_dist` variants (FFI-kernel classified, wrappers added
  in M151). F#'s comparative advantage is likely in builtins coverage,
  intra-query machinery, and reverse-index/CSR work — verify.
- Method that worked all session: fan out read-only inventory agents
  (one for F# features, one for Rust features, one for theory-docs
  recency), then diff, then implement in priority order with
  per-feature exec tests + full-suite gates.

## Session state at handoff

- **Open PR**: #3015 (M152 R lowered-dispatch dynamic guard) — awaiting
  owner merge. This handoff doc rides on the same branch.
- **Flag for the T-tier agent / next session**:
  `test_wam_r_generator:negation_meta_call_e2e_rscript` fails on
  current main (`\+` meta-call e2e prints "true", expects "false") —
  pre-existing before M152, verified by parent-commit run; likely from
  a recent T-tier merge. Bisect per the campaign handoff discipline.
- **Sub-agent quota**: exhausted around 2026-06-11 01:00 UTC; resets
  4am UTC. Launch fan-out agents after that.
- **Owner's standing preferences** (granted during this session):
  create PRs without asking when work is verified; hold merge only
  until full-suite gates are posted; Kotlin target = low investment
  ("easy fixes only"); JVM ground-up + ILasm compound terms + WAT
  read-mode are deliberately deferred (campaign-scoped); the
  ILasm compound-terms campaign is "mine to resume" if chosen.
- **Reviews delivered this session** (conclusions, no need to redo):
  Tn lowering does NOT interfere with external fact sources (LMDB/CSV)
  in Scala/R/Rust — safe by construction; the one real hazard
  (R lowered_dispatch shadowing runtime assertz) is fixed by M152.
  The graph-search agent's simplewiki work correctly used the
  canonical category_parent.tsv artifact chain.
