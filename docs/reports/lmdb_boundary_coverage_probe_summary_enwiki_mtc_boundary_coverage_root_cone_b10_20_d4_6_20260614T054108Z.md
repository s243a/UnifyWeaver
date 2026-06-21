# LMDB Boundary Coverage Probe

Graph: `enwiki_mtc_boundary_coverage_root_cone_b10_20_d4_6`

Root: `7345184`

Target selection: `root-cone-child-depth`

Selection source: `root-cone`

Parent filter: `root-cone`

Root cone depth: `20`

Root cone nodes: `15159`

Boundary suffix mass measured: `False`

Boundary nodes: `24`

Targets: `4`

Path length budgets: `10,20`

Exact mode enumerates simple parent-prefixes until root, boundary, or the path-length budget. Sample mode uses branch-product weighted random walks; its path-count totals are estimates, not direct counts. With `root-reachable`, parent expansion checks finite-horizon reachability recursively. With `root-cone`, parent expansion uses a precomputed root cone and keeps only parent nodes whose cone depth fits within the remaining path budget.

## Selection

| role | child_depth | sampled_frontier_nodes |
|------|-------------|------------------------|
| boundary | 2 | 747 |
| boundary | 3 | 977 |
| target | 4 | 973 |
| target | 6 | 990 |

## Root Cone

| child_depth | new_nodes |
|------------:|----------:|
| 0 | 1 |
| 1 | 32 |
| 2 | 747 |
| 3 | 977 |
| 4 | 973 |
| 5 | 977 |
| 6 | 990 |
| 7 | 990 |
| 8 | 989 |
| 9 | 986 |
| 10 | 994 |
| 11 | 988 |
| 12 | 977 |
| 13 | 968 |
| 14 | 933 |
| 15 | 780 |
| 16 | 552 |
| 17 | 468 |
| 18 | 327 |
| 19 | 245 |
| 20 | 265 |

## Coverage Summary

For sample mode, these are observed random-walk outcomes. Use `Sampled Estimates` for branch-product weighted path-space estimates.

| mode | path_length_budget | targets | completed_targets | observed_terminal_prefixes | observed_root_paths | observed_boundary_hit_prefixes | observed_boundary_hit_fraction | observed_budget_exhausted_prefixes | observed_filtered_dead_ends | mean_boundary_suffix_path_mass | path_count_cap_hit_targets | expansion_cap_hit_targets | cycle_skips | root_unreachable_parent_skips |
|------|-------------------:|--------:|------------------:|------------------:|-----------:|----------------------:|----------------------:|--------------------------:|----------------------------:|-------------------------------:|---------------------------:|--------------------------:|------------:|------------------------------:|
| exact | 10 | 4 | 4 | 51 | 51 | 0 | 0.000000 | 0 | 0 | n/a | 0 | 0 | 0 | 222 |
| exact | 20 | 4 | 4 | 56 | 56 | 0 | 0.000000 | 0 | 0 | n/a | 0 | 0 | 0 | 240 |

## Target Rows

| mode | target_node | path_length_budget | terminal_prefixes | root_paths | boundary_hit_prefixes | boundary_hit_fraction | budget_exhausted_prefixes | filtered_dead_end_prefixes | mean_boundary_remaining_budget | completed | cycle_skips | root_unreachable_parent_skips |
|------|------------:|-------------------:|------------------:|-----------:|----------------------:|----------------------:|--------------------------:|---------------------------:|-------------------------------:|----------:|------------:|------------------------------:|
| exact | 5758053 | 10 | 1 | 1 | 0 | 0.000000 | 0 | 0 | n/a | yes | 0 | 14 |
| exact | 5758053 | 20 | 1 | 1 | 0 | 0.000000 | 0 | 0 | n/a | yes | 0 | 14 |
| exact | 37667066 | 10 | 2 | 2 | 0 | 0.000000 | 0 | 0 | n/a | yes | 0 | 18 |
| exact | 37667066 | 20 | 2 | 2 | 0 | 0.000000 | 0 | 0 | n/a | yes | 0 | 18 |
| exact | 884414 | 10 | 37 | 37 | 0 | 0.000000 | 0 | 0 | n/a | yes | 0 | 123 |
| exact | 884414 | 20 | 39 | 39 | 0 | 0.000000 | 0 | 0 | n/a | yes | 0 | 126 |
| exact | 55649431 | 10 | 11 | 11 | 0 | 0.000000 | 0 | 0 | n/a | yes | 0 | 67 |
| exact | 55649431 | 20 | 14 | 14 | 0 | 0.000000 | 0 | 0 | n/a | yes | 0 | 82 |
