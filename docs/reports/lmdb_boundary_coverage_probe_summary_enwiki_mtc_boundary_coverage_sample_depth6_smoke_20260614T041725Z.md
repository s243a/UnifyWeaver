# LMDB Boundary Coverage Probe

Graph: `enwiki_mtc_boundary_coverage_sample_depth6_smoke`

Root: `7345184`

Target selection: `boundary-descendants`

Boundary nodes: `16`

Targets: `2`

Path length budgets: `8`

Exact mode enumerates simple parent-prefixes until root, boundary, or the path-length budget. Sample mode uses branch-product weighted random walks; its path-count totals are estimates, not direct counts.

## Selection

| role | child_depth | sampled_frontier_nodes |
|------|-------------|------------------------|
| boundary | 0 | 1 |
| boundary | 1 | 16 |
| boundary | 2 | 150 |
| boundary | 3 | 150 |
| target | 6 | 1354 |

## Coverage Summary

For sample mode, these are observed random-walk outcomes. Use `Sampled Estimates` for branch-product weighted path-space estimates.

| mode | path_length_budget | targets | completed_targets | observed_terminal_prefixes | observed_root_paths | observed_boundary_hit_prefixes | observed_boundary_hit_fraction | observed_budget_exhausted_prefixes | mean_boundary_suffix_path_mass | path_count_cap_hit_targets | expansion_cap_hit_targets | cycle_skips |
|------|-------------------:|--------:|------------------:|------------------:|-----------:|----------------------:|----------------------:|--------------------------:|-------------------------------:|---------------------------:|--------------------------:|------------:|
| sample | 8 | 2 | 2 | 500 | 10 | 4 | 0.008000 | 486 | 6.000 | 0 | 0 | 1151 |

## Sampled Estimates

Weighted estimates use the product of eligible parent choices along each sampled simple path. Confidence intervals are bootstrap intervals for the weighted boundary-hit fraction.

| path_length_budget | targets | samples_per_target | mean_estimated_terminal_prefixes | mean_estimated_boundary_hit_fraction | mean_ci95_low | mean_ci95_high |
|-------------------:|--------:|-------------------:|---------------------------------:|------------------------------------:|--------------:|---------------:|
| 8 | 2 | 250 | 96190.394 | 0.000009 | 0.000000 | 0.000022 |

## Target Rows

| mode | target_node | path_length_budget | terminal_prefixes | root_paths | boundary_hit_prefixes | boundary_hit_fraction | mean_boundary_remaining_budget | completed | cycle_skips |
|------|------------:|-------------------:|------------------:|-----------:|----------------------:|----------------------:|-------------------------------:|----------:|------------:|
| sample | 36820502 | 8 | 250 | 4 | 2 | 0.008000 | 4.000 | yes | 708 |
| sample | 40624228 | 8 | 250 | 6 | 2 | 0.008000 | 4.000 | yes | 443 |
