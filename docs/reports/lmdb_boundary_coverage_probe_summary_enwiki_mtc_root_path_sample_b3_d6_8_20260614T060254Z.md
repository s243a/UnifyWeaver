# LMDB Boundary Coverage Probe

Graph: `enwiki_mtc_root_path_sample_b3_d6_8`

Root: `7345184`

Target selection: `root-cone-child-depth`

Selection source: `root-cone`

Parent filter: `root-cone`

Root cone depth: `20`

Root cone nodes: `18127`

Boundary suffix mass measured: `False`

Boundary nodes: `962`

Targets: `4`

Path length budgets: `10,20`

Exact mode enumerates simple parent-prefixes until root, boundary, or the path-length budget. Sample mode uses branch-product weighted random walks; its path-count totals are estimates, not direct counts. With `root-reachable`, parent expansion checks finite-horizon reachability recursively. With `root-cone`, parent expansion uses a precomputed root cone and keeps only parent nodes whose cone depth fits within the remaining path budget.

## Selection

| role | child_depth | sampled_frontier_nodes |
|------|-------------|------------------------|
| boundary | 3 | 962 |
| target | 6 | 990 |
| target | 8 | 981 |

## Root Cone

| child_depth | new_nodes |
|------------:|----------:|
| 0 | 1 |
| 1 | 32 |
| 2 | 742 |
| 3 | 962 |
| 4 | 969 |
| 5 | 981 |
| 6 | 990 |
| 7 | 991 |
| 8 | 981 |
| 9 | 971 |
| 10 | 966 |
| 11 | 950 |
| 12 | 935 |
| 13 | 922 |
| 14 | 946 |
| 15 | 955 |
| 16 | 969 |
| 17 | 976 |
| 18 | 969 |
| 19 | 962 |
| 20 | 957 |

## Coverage Summary

For sample and root-sample modes, these are observed random-walk outcomes. Use the estimate sections below for branch-product weighted path-space estimates.

| mode | path_length_budget | targets | completed_targets | observed_terminal_prefixes | observed_root_paths | observed_boundary_hit_prefixes | observed_boundary_hit_fraction | observed_budget_exhausted_prefixes | observed_filtered_dead_ends | mean_boundary_suffix_path_mass | path_count_cap_hit_targets | expansion_cap_hit_targets | cycle_skips | root_unreachable_parent_skips |
|------|-------------------:|--------:|------------------:|------------------:|-----------:|----------------------:|----------------------:|--------------------------:|----------------------------:|-------------------------------:|---------------------------:|--------------------------:|------------:|------------------------------:|
| root-sample | 10 | 4 | 4 | 4000 | 4000 | 3650 | 0.912500 | 0 | 0 | n/a | 0 | 0 | 0 | 85641 |
| root-sample | 20 | 4 | 4 | 4000 | 4000 | 3636 | 0.909000 | 0 | 0 | n/a | 0 | 0 | 0 | 85659 |

## Root Path Sample Estimates

Root-sample mode ignores boundary stopping and walks until root, budget exhaustion, or dead end. `estimated_root_paths` estimates the root-reaching search-space size from the branch-product weight. `estimated_mean_root_path_length` is the corresponding weighted mean path length.

| path_length_budget | targets | samples_per_target | mean_estimated_root_paths | mean_estimated_mean_root_path_length | mean_estimated_root_boundary_hit_fraction |
|-------------------:|--------:|-------------------:|--------------------------:|-------------------------------------:|------------------------------------------:|
| 10 | 4 | 1000 | 17.688 | 8.179 | 0.905622 |
| 20 | 4 | 1000 | 19.924 | 8.323 | 0.900566 |

## Target Rows

| mode | target_node | path_length_budget | terminal_prefixes | root_paths | boundary_hit_prefixes | boundary_hit_fraction | budget_exhausted_prefixes | filtered_dead_end_prefixes | mean_boundary_remaining_budget | completed | cycle_skips | root_unreachable_parent_skips |
|------|------------:|-------------------:|------------------:|-----------:|----------------------:|----------------------:|--------------------------:|---------------------------:|-------------------------------:|----------:|------------:|------------------------------:|
| root-sample | 44388285 | 10 | 1000 | 1000 | 1000 | 1.000000 | 0 | 0 | 7.000 | yes | 0 | 15444 |
| root-sample | 44388285 | 20 | 1000 | 1000 | 1000 | 1.000000 | 0 | 0 | 17.000 | yes | 0 | 15479 |
| root-sample | 54801046 | 10 | 1000 | 1000 | 1000 | 1.000000 | 0 | 0 | 7.000 | yes | 0 | 17964 |
| root-sample | 54801046 | 20 | 1000 | 1000 | 1000 | 1.000000 | 0 | 0 | 17.000 | yes | 0 | 18020 |
| root-sample | 71376344 | 10 | 1000 | 1000 | 1000 | 1.000000 | 0 | 0 | 5.487 | yes | 0 | 21037 |
| root-sample | 71376344 | 20 | 1000 | 1000 | 1000 | 1.000000 | 0 | 0 | 15.507 | yes | 0 | 20970 |
| root-sample | 79716218 | 10 | 1000 | 1000 | 650 | 0.650000 | 0 | 0 | 4.808 | yes | 0 | 31196 |
| root-sample | 79716218 | 20 | 1000 | 1000 | 636 | 0.636000 | 0 | 0 | 14.829 | yes | 0 | 31190 |
