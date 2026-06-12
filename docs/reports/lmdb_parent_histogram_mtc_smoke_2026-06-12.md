# Enwiki MTC Parent Histogram Smoke Benchmark

Date: 2026-06-12

Branch: codex/mtc-ancestor-cone-histograms

## Scope

This smoke benchmark moves from parent-degree moments to actual bounded parent-path histograms over the title-resolved enwiki Main_topic_classifications artifact:

    artifact=/home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_cats_correct/lmdb_resident
    root_title=Main_topic_classifications
    root_id=7345184

The benchmark counts simple parent paths only: a path is rejected if it would revisit a node already present on the current path. This is important for enwiki, where the parent graph is cyclic. A node-only memoized histogram is exact for DAGs, but not for cyclic simple-path semantics, because the admissible continuation from a node depends on the current visited set.

## Command

    python3 scripts/lmdb_parent_histogram_benchmark.py --lmdb-dir /home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_cats_correct/lmdb_resident --root 7345184 --graph-name enwiki_mtc_parent_histogram_smoke --child-depths 2,3 --children-per-node 128 --frontier-limit 2000 --targets-per-depth 20 --budgets 4,6,8 --path-cap 50000 --expansion-cap 100000 --seed enwiki-mtc-hist-smoke-v1 --output-dir /mnt/c/Users/johnc/Scratch/full-parent-branching-diagnostic

Generated outputs:

    /mnt/c/Users/johnc/Scratch/full-parent-branching-diagnostic/lmdb_parent_histogram_benchmark_enwiki_mtc_parent_histogram_smoke_20260612T005353Z.jsonl
    /mnt/c/Users/johnc/Scratch/full-parent-branching-diagnostic/lmdb_parent_histogram_benchmark_summary_enwiki_mtc_parent_histogram_smoke_20260612T005353Z.md

## Selection

| child_depth | sampled_frontier_nodes |
|-------------|------------------------|
| 0 | 1 |
| 1 | 35 |
| 2 | 1023 |
| 3 | 2000 |

## Summary

| target_budget_rows | reachable_rows | capped_rows |
|--------------------|----------------|-------------|
| 120 | 113 | 39 |

## Histogram Cost By Budget

| budget | rows | reachable | mean_paths | p95_paths | max_paths | mean_bins | p95_bins | max_bins | mean_nodes_expanded | mean_cycle_skips | capped_rows |
|--------|------|-----------|------------|-----------|-----------|-----------|----------|----------|---------------------|------------------|-------------|
| 4 | 40 | 40 | 3.900 | 8.000 | 8 | 2.400 | 3.000 | 3 | 626.9 | 43.2 | 0 |
| 6 | 40 | 40 | 21.950 | 44.000 | 65 | 4.550 | 5.000 | 5 | 15733.3 | 3583.8 | 0 |
| 8 | 40 | 33 | 32.515 | 82.000 | 149 | 5.000 | 7.000 | 7 | 99199.6 | 41348.1 | 39 |

Budget 8 is already near the expansion cap for almost every row, so those histograms should be treated as capped observations. The cycle-skip counts are high enough that simple-path validity is not a minor edge case; it is part of the cost model.

## Parent Degree Signal For Reachable Rows

| budget | rows | mean_full_p | b_full | mean_root_p | b_root |
|--------|------|-------------|--------|-------------|--------|
| 4 | 40 | 3.700 | 4.405405 | 2.125 | 2.435294 |
| 6 | 40 | 3.700 | 4.405405 | 2.825 | 3.194690 |
| 8 | 33 | 3.333 | 3.781818 | 2.758 | 3.043956 |

Root-reaching parent branching remains above 2 in every budget slice. This is much closer to the expected enwiki behavior than the earlier sampled-TSV local degree estimates.

## Fit Error By Model

| model | rows | mean_l1 | p95_l1 | max_l1 | mean_cdf_error |
|-------|------|---------|--------|--------|----------------|
| binomial_fit | 113 | 0.348106 | 1.000000 | 1.500000 | 0.101057 |
| shifted_gamma_fit | 113 | 0.813835 | 1.216973 | 1.362897 | 0.381731 |

Binomial fits remain better than shifted Gamma on this smoke sample, but the errors are no longer tiny. These are real multi-bin distributions, not the one-bin SimpleWiki-like histograms from the earlier pilot. Because many budget-8 rows are capped, the fit numbers are best read as smoke-test diagnostics, not final accuracy estimates.

## Implication

The next optimizer validation should compare three modes on the same selected target set:

    1. exact simple-path histograms for budgets that finish uncapped;
    2. capped exact searches with explicit cap metadata;
    3. approximation or cache-boundary policies initialized from root-reaching parent-degree priors.

The key semantic constraint is that parent paths are simple paths. Any memoized histogram cache must either be used only where the graph is acyclic under the active cone, or include enough visited/boundary state to remain exact. Node-only histogram memoization is a performance shortcut, not a valid exact semantics for cyclic enwiki cones.

## Validation

- python3 -m unittest tests.test_lmdb_parent_histogram_benchmark tests.test_lmdb_parent_branching_diagnostic
- python3 scripts/lmdb_parent_histogram_benchmark.py ... --graph-name enwiki_mtc_parent_histogram_smoke
- git diff --check

## Cache Scheduling Note

Cycles are expected to be rare relative to ordinary parent branching, so the per-path visited set is primarily a correctness guard. The more important optimization signal is the maximum parent distance to root. If a child node has a lower known L_max than the current search horizon, its histogram can be recomputed or refreshed first and then used as a boundary value for deeper searches.

This suggests an optional root-distance ordered cache schedule. It should be treated as a refinement rather than the default path, because it adds scheduler state and only pays off when enough downstream searches can reuse the refreshed child histograms:

    1. compute scalar L_min/L_max bounds for candidate nodes;
    2. refresh exact histograms for children whose L_max is below the active budget;
    3. use those refreshed child histograms as boundary states when evaluating deeper or wider ancestor cones;
    4. fall back to capped search or distributional priors when L_max is high enough that exact refresh is too expensive.

This is different from a naive child-depth traversal. The useful admission question is not how far the node is from the sampled child frontier; it is whether the node has a small enough parent-distance support to root that its histogram is cheap and reusable. The baseline implementation can ignore this refinement and rely on capped exact searches plus priors; the scheduler should enable it only when reuse counters or workload shape show that the extra bookkeeping is worthwhile.
