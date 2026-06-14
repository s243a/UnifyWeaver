# Enwiki MTC Boundary Cache Hit Geometry

This run adds cache-hit geometry metrics to `scripts/lmdb_parent_boundary_cache_benchmark.py`. The prior boundary-descendant target selection proved that targets can be sampled with known selected-boundary ancestors, but cached search still expanded about as many nodes as full search. The new metrics explain whether cache hits happen early enough to skip meaningful suffix work.

## Metrics

Each `boundary_cache_comparison` row now records:

- `cache_hit_depth_sum`
- `cache_hit_remaining_budget_sum`
- `cache_hit_suffix_path_count_sum`
- `mean_cache_hit_depth`
- `mean_cache_hit_remaining_budget`
- `mean_cache_hit_suffix_path_count`
- `cache_hits_by_depth`

The markdown summary includes a `Cache Hit Geometry` section aggregated by budget.

## Smoke Runs

Both runs used enwiki MTC root `7345184`, `--target-selection boundary-descendants`, boundary depths `2,3`, recurrence boundary construction, budget `8`, path cap `50000`, expansion cap `100000`, and baseline admission.

Artifacts:

- `lmdb_parent_boundary_cache_benchmark_enwiki_mtc_boundary_descendant_hit_geometry_smoke_20260614T030022Z.jsonl`
- `lmdb_parent_boundary_cache_benchmark_summary_enwiki_mtc_boundary_descendant_hit_geometry_smoke_20260614T030022Z.md`
- `lmdb_parent_boundary_cache_benchmark_enwiki_mtc_boundary_descendant_hit_geometry_depth7_smoke_20260614T030036Z.jsonl`
- `lmdb_parent_boundary_cache_benchmark_summary_enwiki_mtc_boundary_descendant_hit_geometry_depth7_smoke_20260614T030036Z.md`

## Result

| run | target_depth | rows | mean_time_ratio | mean_node_ratio | mean_cache_hits | mean_hit_depth | mean_remaining_budget | mean_suffix_path_count | hit_depth_histogram |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| boundary-descendant geometry | 6 | 8 | 1.133 | 0.994 | 20.750 | 7.157 | 0.843 | 0.620 | `{"3": 5, "4": 4, "5": 6, "6": 18, "7": 45, "8": 88}` |
| boundary-descendant geometry depth7 | 7 | 6 | 1.224 | 1.000 | 35.667 | 7.383 | 0.617 | 0.140 | `{"4": 1, "5": 9, "6": 14, "7": 73, "8": 117}` |

The cache is being hit, but the hits are mostly too late to skip useful work. In the depth-6 smoke, average remaining budget at hit time is less than one edge. In the depth-7 smoke it is even lower, and most hits occur at depths 7 or 8. That explains why node expansion stays near `1.0` even when boundary-descendant sampling improves accuracy.

## Interpretation

The next experiment should not be another decode optimization. The benchmark needs a boundary-hit shape where selected boundaries are reached earlier in the parent DFS. Two likely routes are:

- sample targets below shallower cached boundaries and run with larger target budgets, so a boundary hit leaves several remaining parent hops; or
- add a target filter/admission diagnostic that selects only targets whose first cache hit occurs before a minimum depth or with at least `k` remaining budget.

Until the mean remaining budget at cache hit time is materially above zero, cached search cannot be expected to reduce expansion much.
