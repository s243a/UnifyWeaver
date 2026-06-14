# LMDB Boundary Coverage Probe

Graph: `enwiki_mtc_boundary_coverage_root_reachable_b10_20_two_targets`

Root: `7345184`

Target selection: `boundary-descendants`

Parent filter: `root-reachable`

Boundary suffix mass measured: `False`

Boundary nodes: `6`

Targets: `2`

Path length budgets: `10,20`

Exact mode enumerates simple parent-prefixes until root, boundary, or the path-length budget. Sample mode uses branch-product weighted random walks; its path-count totals are estimates, not direct counts. With `root-reachable`, parent expansion keeps only parent nodes that can still reach the selected root within the remaining path budget.

## Selection

| role | child_depth | sampled_frontier_nodes |
|------|-------------|------------------------|
| boundary | 0 | 1 |
| boundary | 1 | 12 |
| boundary | 2 | 100 |
| target | 4 | 412 |

## Coverage Summary

For sample mode, these are observed random-walk outcomes. Use `Sampled Estimates` for branch-product weighted path-space estimates.

| mode | path_length_budget | targets | completed_targets | observed_terminal_prefixes | observed_root_paths | observed_boundary_hit_prefixes | observed_boundary_hit_fraction | observed_budget_exhausted_prefixes | observed_filtered_dead_ends | mean_boundary_suffix_path_mass | path_count_cap_hit_targets | expansion_cap_hit_targets | cycle_skips | root_unreachable_parent_skips |
|------|-------------------:|--------:|------------------:|------------------:|-----------:|----------------------:|----------------------:|--------------------------:|----------------------------:|-------------------------------:|---------------------------:|--------------------------:|------------:|------------------------------:|
| exact | 10 | 2 | 2 | 1355 | 1309 | 15 | 0.011070 | 0 | 31 | n/a | 0 | 0 | 112 | 7129 |
| exact | 20 | 2 | 1 | 31943 | 31589 | 81 | 0.002536 | 0 | 273 | n/a | 0 | 1 | 5041 | 153695 |

## Target Rows

| mode | target_node | path_length_budget | terminal_prefixes | root_paths | boundary_hit_prefixes | boundary_hit_fraction | budget_exhausted_prefixes | filtered_dead_end_prefixes | mean_boundary_remaining_budget | completed | cycle_skips | root_unreachable_parent_skips |
|------|------------:|-------------------:|------------------:|-----------:|----------------------:|----------------------:|--------------------------:|---------------------------:|-------------------------------:|----------:|------------:|------------------------------:|
| exact | 698261 | 10 | 1318 | 1274 | 13 | 0.009863 | 0 | 31 | 3.385 | yes | 106 | 6960 |
| exact | 698261 | 20 | 31637 | 31289 | 79 | 0.002497 | 0 | 269 | 4.468 | no | 5021 | 152151 |
| exact | 40402468 | 10 | 37 | 35 | 2 | 0.054054 | 0 | 0 | 8.000 | yes | 6 | 169 |
| exact | 40402468 | 20 | 306 | 300 | 2 | 0.006536 | 0 | 4 | 18.000 | yes | 20 | 1544 |
