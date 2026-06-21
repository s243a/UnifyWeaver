# Enwiki MTC Boundary-Descendant Hit Shape

This run adds a benchmark target-selection mode for boundary-cache validation: `--target-selection boundary-descendants`. Instead of independently sampling targets by child depth, the benchmark first samples boundary candidates and then samples targets from descendants of those selected boundaries at requested absolute target depths. The intent is to test cache economics when sampled targets are known to have selected boundary ancestors.

## Implementation

The new mode uses the LMDB child index to walk down from selected boundary nodes. For each boundary node with sampled child depth `d_b` and each requested target depth `d_t`, it samples descendants at suffix depth `d_t - d_b` when that value is positive. It records `target_selection` in the selection row and the markdown summary.

## Smoke Runs

Both runs used enwiki MTC root `7345184`, boundary depths `2,3`, recurrence boundary construction, budget `8`, path cap `50000`, expansion cap `100000`, and baseline admission.

Artifacts:

- `lmdb_parent_boundary_cache_benchmark_enwiki_mtc_boundary_descendant_hit_shape_smoke_20260614T025146Z.jsonl`
- `lmdb_parent_boundary_cache_benchmark_summary_enwiki_mtc_boundary_descendant_hit_shape_smoke_20260614T025146Z.md`
- `lmdb_parent_boundary_cache_benchmark_enwiki_mtc_boundary_descendant_hit_shape_depth7_smoke_20260614T025206Z.jsonl`
- `lmdb_parent_boundary_cache_benchmark_summary_enwiki_mtc_boundary_descendant_hit_shape_depth7_smoke_20260614T025206Z.md`

## Result

| run | target_depth | targets | descendant_candidates | mean_l1 | mean_path_count_relative_error | mean_node_ratio | mean_time_ratio | mean_full_time_ns | mean_cached_time_ns | mean_hist_hits | full_capped | cached_capped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boundary-descendant smoke | 6 | 8 | 18,835 | 0.007757 | 0.032182 | 0.994 | 1.150 | 58,757,694.6 | 67,323,865.4 | 20.750 | 7 | 7 |
| boundary-descendant depth7 smoke | 7 | 6 | 16,665 | 0.000000 | 0.000000 | 1.000 | 1.149 | 58,548,164.0 | 67,230,831.3 | 35.667 | 6 | 6 |

The new hit shape does what it was intended to do: it gives targets known selected-boundary ancestors, and the depth-7 run is exact under the measured caps. However, runtime is still slower because the cached and full searches expand nearly the same number of nodes. Both runs are also capped in nearly every row, so the current shape is not yet demonstrating skipped suffix work as reduced expansion.

## Interpretation

The earlier concern that random target sampling might miss useful cache boundaries was valid enough to test, but it is not the whole explanation for slower cached rows. With boundary-descendant targets, cache hits are present and accuracy improves, yet node expansion does not fall. The next benchmark refinement should measure where the cached hit occurs in the DFS and how much remaining suffix work it actually skips. A useful next field would be per-row hit depth and remaining budget at hit time, aggregated as `mean_cache_hit_depth` and `mean_cache_hit_remaining_budget`.
