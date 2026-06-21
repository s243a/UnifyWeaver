# Enwiki MTC Boundary Cache Early-Hit Budget Probe

Date: 2026-06-14

Terminology note: `path_count_cap` is the maximum number of root-reaching paths enumerated before stopping a row; it is not the maximum parent-hop length. The path-length budget is the `budget`/`path_length_budget` value.

This probe adds cache-hit geometry to the boundary-cache benchmark so a run can distinguish "the search hit a cached boundary" from "the search hit it early enough for the splice to matter."  The new row-level fields record the first cached hit depth, first remaining budget, maximum remaining budget, and a histogram of cache hits by remaining budget.

## Smoke Runs

| graph | target_depth | targets | budget | path_count_cap | mean_cache_hits | mean_remaining_budget | mean_first_remaining_budget | mean_max_remaining_budget | hits_rem_ge_2 | hits_rem_ge_4 | hits_rem_ge_6 | full_capped | cached_capped | mean_time_ratio |
|-------|-------------:|--------:|-------:|---------:|----------------:|----------------------:|----------------------------:|--------------------------:|--------------:|--------------:|--------------:|------------:|--------------:|----------------:|
| enwiki_mtc_boundary_descendant_early_hit_budget8_smoke | 7 | 4 | 8 | 50000 | 17.000 | 0.956 | 2.667 | 3.333 | 20 | 4 | 0 | 4 | 4 | 1.089 |
| enwiki_mtc_boundary_descendant_early_hit_budget12_smoke | 7 | 4 | 12 | 50000 | 0.000 | n/a | n/a | n/a | 0 | 0 | 0 | 4 | 4 | 1.171 |
| enwiki_mtc_boundary_descendant_early_hit_budget12_single_probe | 5 | 1 | 12 | 1000000 | 0.000 | n/a | n/a | n/a | 0 | 0 | 0 | 1 | 1 | 1.189 |

The budget-8 run confirms the original concern: cache hits are common, but most arrive with little budget left.  Its remaining-budget histogram was `{0: 35, 1: 13, 2: 12, 3: 4, 4: 4}`, so only 20 of 68 hits had at least two hops remaining and none had at least six.  Mean suffix path count was 0.662 paths per hit, which means many boundary hits splice little or no useful suffix mass under the current path-length constraint.

The budget-12 probes show a second failure mode.  Raising the starting search budget increases the theoretical value of a boundary splice, but these sampled rows hit the path-count cap before the selected boundary appeared in parent traversal order.  In that regime a larger budget does not help unless the search also reaches the cached boundary before the cap or expansion limit fires.  Because capped runs only observe an enumerated prefix, full-run cache impact still requires an estimate of the unvisited path mass or an explicit assumption that the unvisited paths have similar boundary-hit statistics.

## Interpretation

The useful decision variable is not just boundary depth.  It is the distribution of remaining budget at the first cache hit, plus the path mass available behind that hit.  A boundary with many hits at remaining budget 0 or 1 is mostly acting as a late accounting shortcut.  A boundary with hits at remaining budget 4 or more can replace a substantially larger suffix search.

For follow-up policy work, the cache planner should prefer boundaries that are both in the target ancestor cone and likely to be reached before the query cap.  If parent traversal order can enumerate many root-reaching paths before a cached boundary, then admission alone is insufficient; the runtime needs boundary-aware traversal priority or an ancestor-cone query plan that only recurses through relevant ancestors.

## Artifacts

- `docs/reports/lmdb_parent_boundary_cache_benchmark_summary_enwiki_mtc_boundary_descendant_early_hit_budget8_smoke_20260614T034456Z.md`
- `docs/reports/lmdb_parent_boundary_cache_benchmark_enwiki_mtc_boundary_descendant_early_hit_budget8_smoke_20260614T034456Z.jsonl`
- `docs/reports/lmdb_parent_boundary_cache_benchmark_summary_enwiki_mtc_boundary_descendant_early_hit_budget12_smoke_20260614T034517Z.md`
- `docs/reports/lmdb_parent_boundary_cache_benchmark_enwiki_mtc_boundary_descendant_early_hit_budget12_smoke_20260614T034517Z.jsonl`
- `docs/reports/lmdb_parent_boundary_cache_benchmark_summary_enwiki_mtc_boundary_descendant_early_hit_budget12_single_probe_20260614T034631Z.md`
- `docs/reports/lmdb_parent_boundary_cache_benchmark_enwiki_mtc_boundary_descendant_early_hit_budget12_single_probe_20260614T034631Z.jsonl`
