# Enwiki MTC Boundary Cache Estimator Validation

Date: 2026-06-13

This report checks the current precompute-depth estimator against measured
boundary-cache behavior on the enwiki Main_topic_classifications LMDB artifact.

The estimator now separates:

- `boundary_depth`: where the cached distribution is materialized;
- `target_depth`: where the query starts;
- `suffix_hops = target_depth - boundary_depth`: the skipped parent-search
  suffix that should scale like `b^m`.

## Setup

Root:

```text
Category:Main_topic_classifications = 7345184
```

Benchmark shape:

```text
target_depth = 8
boundary_depths = 2,3,4,5,6,7
budget = 8
targets_per_depth = 2
boundary_builder = recurrence
include_target_ancestor_boundaries = true
path_cap = 3000
expansion_cap = 20000
```

Target-ancestor boundaries are included so the benchmark measures cache behavior
on nodes actually encountered by the sampled target searches, rather than random
frontier overlap.

## Measured Grid

| boundary_depth | suffix_hops | boundary_nodes | added_target_ancestor_boundaries | cached_nodes | mean_hist_hits | mean_time_ratio | mean_full_time_ns | mean_cached_time_ns | mean_l1 | full_capped | cached_capped |
|---------------:|------------:|---------------:|---------------------------------:|-------------:|---------------:|----------------:|------------------:|--------------------:|--------:|------------:|--------------:|
| 2 | 6 | 81 | 80 | 81 | 30.000 | 1.146 | 10887088.0 | 12397237.0 | 0.000000 | 2 | 2 |
| 3 | 5 | 81 | 80 | 81 | 13.500 | 1.117 | 10806888.5 | 12067137.0 | 0.000000 | 2 | 2 |
| 4 | 4 | 81 | 80 | 81 | 13.000 | 1.139 | 10699788.5 | 12179937.5 | 0.000000 | 2 | 2 |
| 5 | 3 | 81 | 80 | 81 | 16.000 | 1.089 | 10286039.0 | 11204138.0 | 0.242820 | 2 | 2 |
| 6 | 2 | 9 | 8 | 9 | 1.000 | 1.060 | 10890238.5 | 11538138.5 | 0.000000 | 2 | 2 |
| 7 | 1 | 50 | 49 | 50 | 0.000 | 1.005 | 11730854.5 | 11713305.5 | 0.000000 | 2 | 2 |

The latest raw benchmark outputs are:

```text
docs/reports/lmdb_parent_boundary_cache_benchmark_enwiki_mtc_boundary_cache_estimator_validation_b2_recurrence_20260613T164208Z.jsonl
docs/reports/lmdb_parent_boundary_cache_benchmark_enwiki_mtc_boundary_cache_estimator_validation_b3_recurrence_20260613T164209Z.jsonl
docs/reports/lmdb_parent_boundary_cache_benchmark_enwiki_mtc_boundary_cache_estimator_validation_b4_recurrence_20260613T164210Z.jsonl
docs/reports/lmdb_parent_boundary_cache_benchmark_enwiki_mtc_boundary_cache_estimator_validation_b5_recurrence_20260613T164210Z.jsonl
docs/reports/lmdb_parent_boundary_cache_benchmark_enwiki_mtc_boundary_cache_estimator_validation_b6_recurrence_20260613T164211Z.jsonl
docs/reports/lmdb_parent_boundary_cache_benchmark_enwiki_mtc_boundary_cache_estimator_validation_b7_recurrence_20260613T164212Z.jsonl
```

## Interpretation

The hit pattern broadly agrees with the estimator's structure: useful cache
interactions are concentrated when `suffix_hops` is still at least a few hops.
At boundary depth 7, `suffix_hops = 1` and the measured mean cache hit count is
zero for this sample.

The timing result does not validate the current cost constants.  Every measured
time ratio is approximately `1.0` or worse, even when many cache hits occur. The
main caveat is that every full and cached target row hit a cap. In this capped
regime, both searches stop early, so the benchmark is not measuring the full
uncached suffix cost that the estimator assumes. The cached path also pays
payload decode and histogram splice overhead before hitting the same cap.

Depth 5 shows non-zero distribution error (`mean_l1 = 0.242820`) because cached
recurrence states are node-local and do not know the current simple-path visited
set. This is the previously documented cycle/visited-state approximation.

## Consequences For The Estimator

The estimator needs one more term before it can predict this benchmark:

```text
effective_saved_suffix_cost =
    min(b^suffix_hops, cap_limited_uncached_work)
    - cached_eval_cost
    - decode_cost
    - splice_cost
```

The current model is closer to an uncapped suffix-cost planner. That is still
useful for deciding where full searches would become expensive, but capped
search workloads need a different calibration path.

The next validation step should either:

1. raise caps enough that full search work is not truncated, on a much smaller
   target sample; or
2. teach the estimator to use observed cap-limited work as the suffix-cost
   ceiling when the operational query also has `path_cap` or `expansion_cap`.

The second path is likely more realistic for production, because bounded search
queries will normally keep caps.
