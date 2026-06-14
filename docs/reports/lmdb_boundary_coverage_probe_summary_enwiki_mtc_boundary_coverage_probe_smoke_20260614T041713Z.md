# LMDB Boundary Coverage Probe

Graph: `enwiki_mtc_boundary_coverage_probe_smoke`

Root: `7345184`

Target selection: `boundary-descendants`

Boundary nodes: `6`

Targets: `2`

Path length budgets: `4`

Exact mode enumerates simple parent-prefixes until root, boundary, or the path-length budget. Sample mode uses branch-product weighted random walks; its path-count totals are estimates, not direct counts.

## Selection

| role | child_depth | sampled_frontier_nodes |
|------|-------------|------------------------|
| boundary | 0 | 1 |
| boundary | 1 | 12 |
| boundary | 2 | 100 |
| target | 4 | 309 |

## Coverage Summary

For sample mode, these are observed random-walk outcomes. Use `Sampled Estimates` for branch-product weighted path-space estimates.

| mode | path_length_budget | targets | completed_targets | observed_terminal_prefixes | observed_root_paths | observed_boundary_hit_prefixes | observed_boundary_hit_fraction | observed_budget_exhausted_prefixes | mean_boundary_suffix_path_mass | path_count_cap_hit_targets | expansion_cap_hit_targets | cycle_skips |
|------|-------------------:|--------:|------------------:|------------------:|-----------:|----------------------:|----------------------:|--------------------------:|-------------------------------:|---------------------------:|--------------------------:|------------:|
| exact | 4 | 2 | 2 | 1656 | 9 | 23 | 0.013889 | 1624 | 0.130 | 0 | 0 | 75 |
| sample | 4 | 2 | 2 | 200 | 4 | 24 | 0.120000 | 172 | 0.750 | 0 | 0 | 30 |

## Sampled Estimates

Weighted estimates use the product of eligible parent choices along each sampled simple path. Confidence intervals are bootstrap intervals for the weighted boundary-hit fraction.

| path_length_budget | targets | samples_per_target | mean_estimated_terminal_prefixes | mean_estimated_boundary_hit_fraction | mean_ci95_low | mean_ci95_high |
|-------------------:|--------:|-------------------:|---------------------------------:|------------------------------------:|--------------:|---------------:|
| 4 | 2 | 100 | 759.510 | 0.010409 | 0.003776 | 0.019644 |

## Target Rows

| mode | target_node | path_length_budget | terminal_prefixes | root_paths | boundary_hit_prefixes | boundary_hit_fraction | mean_boundary_remaining_budget | completed | cycle_skips |
|------|------------:|-------------------:|------------------:|-----------:|----------------------:|----------------------:|-------------------------------:|----------:|------------:|
| exact | 1294488 | 4 | 1454 | 8 | 21 | 0.014443 | 0.524 | yes | 68 |
| sample | 1294488 | 4 | 100 | 3 | 12 | 0.120000 | 1.667 | yes | 19 |
| exact | 1903269 | 4 | 202 | 1 | 2 | 0.009901 | 1.500 | yes | 7 |
| sample | 1903269 | 4 | 100 | 1 | 12 | 0.120000 | 1.833 | yes | 11 |
