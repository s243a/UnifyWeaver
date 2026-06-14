# LMDB Boundary Coverage Probe

Graph: `enwiki_mtc_shallow_splice_validation`

Root: `7345184`

Target selection: `root-cone-child-depth`

Selection source: `root-cone`

Parent filter: `root-cone`

Root cone depth: `3`

Root cone nodes: `3935`

Boundary suffix mass measured: `True`

Path value kernel: `count`

Path value branching factor: `n/a`

Path value power: `n/a`

Boundary nodes: `43`

Targets: `4`

Path length budgets: `10`

## How This Was Generated

- Boundary candidates were sampled from requested child depth(s) `1, 2` and target rows from requested child depth(s) `2, 3` using selection source `root-cone`.
- Boundary and target nodes were sampled from the precomputed root-cone depth buckets with per-depth limits `boundaries_per_depth=12` and `targets_per_depth=2`.
- Mode `exact` controls the row generator: `exact` enumerates all simple parent prefixes until root, boundary, or budget; `sample` performs branch-product weighted boundary-stopped random walks; `root-sample` samples walks to root without stopping at boundaries.
- Parent filter `root-cone` is applied during parent expansion. `root-cone` accepts only parents inside the precomputed root cone whose cone depth fits within the remaining path budget; `root-reachable` uses recursive finite-horizon reachability; `all` does no root-scope pruning.
- The root cone was built to child depth `3` with `3935` nodes, `root_cone_children_per_node=0`, and `root_cone_frontier_limit=3000`.
- Target-ancestor boundary inclusion was enabled, with `target_ancestor_boundary_limit=20`.
- Boundary suffix mass measured: `True`. When this is false, boundary-hit rows measure coverage only; they do not splice cached suffix mass into a total root-path estimate.
- Full exact validation was requested. The report includes a separate comparison between boundary-stopped suffix splicing and full filtered DFS to root for exact-mode rows.
- Path value kernel `count` defines the functional being estimated after path coverage is known. It is separate from the random-walk proposal correction.

## Table Guide

- `Selection` lists observed frontier sizes for boundary and target depths. In newer reports, the requested depths are stated above; the table may include intermediate traversal depths.
- `Root Cone` shows the bounded child-reachable cone used for root-cone filtering. These are not parent-path counts; they are child-depth frontier counts from the root.
- `Coverage Summary` aggregates observed terminal outcomes by mode and path-length budget. In exact mode these are enumerated simple-prefix counts; in sample/root-sample modes they are raw sample outcomes.
- `spliced_total_root_paths`, `spliced_total_value_sum`, and `spliced_mean_path_length` are boundary-aware estimates: direct root terminals plus suffix histogram mass/value from boundary hits.
- `Boundary Sample Estimates` and `Root Path Sample Estimates` contain branch-product weighted estimates. Use those estimate tables, not raw observed sample counts, when reasoning about path-space size.
- `Full Exact Splice Validation`, when present, checks whether boundary-stopped exact search plus suffix histograms reproduces full filtered DFS to root for the same target and budget.
- `Target Rows` is per target and budget. `root_paths` counts direct root terminals reached before a boundary stop; `boundary_hit_prefixes` counts prefixes where the boundary condition would take over.
- `root_unreachable_parent_skips` counts parent edges rejected by the active parent filter. Under `root-cone`, that includes parents outside the cone or too deep for the remaining budget, not only globally unreachable parents.
- `budget_exhausted_prefixes`, `path_count_cap_hit_targets`, and `expansion_cap_hit_targets` identify rows whose result is limited by the path budget or safety caps.

## Result Implications

- Target evaluation completed `4/4` rows without path-count or expansion caps.
- `exact` budget `10`: `44` terminal prefixes, `27` direct root paths, `17` boundary-hit prefixes, boundary-hit fraction `0.386364`, `0` budget-exhausted prefixes, and spliced root mass `63.000`.
- `1` target-budget rows have `root_paths=0` and positive boundary hits. In this report that means enumeration stopped at a boundary before reaching root; it is boundary coverage, not evidence that those targets lack root paths.
- Boundary suffix mass was measured, so boundary-hit prefixes are combined with suffix histograms to estimate total root-path mass, aggregate value, and mean path length under the remaining budget.
- The active parent filter rejected `111` parent edges. Under `root-cone`, these skips are part of the scoped experiment definition, not necessarily data errors.

## Selection

| role | child_depth | sampled_frontier_nodes |
|------|-------------|------------------------|
| boundary | 1 | 35 |
| boundary | 2 | 1007 |
| target | 2 | 1007 |
| target | 3 | 2892 |

## Root Cone

| child_depth | new_nodes |
|------------:|----------:|
| 0 | 1 |
| 1 | 35 |
| 2 | 1007 |
| 3 | 2892 |

## Coverage Summary

For sample and root-sample modes, these are observed random-walk outcomes. Use the estimate sections below for branch-product weighted path-space estimates.

| mode | path_length_budget | targets | completed_targets | observed_terminal_prefixes | observed_root_paths | observed_boundary_hit_prefixes | observed_boundary_hit_fraction | observed_budget_exhausted_prefixes | observed_filtered_dead_ends | mean_boundary_suffix_path_mass | spliced_total_root_paths | spliced_total_value_sum | spliced_mean_path_length | path_count_cap_hit_targets | expansion_cap_hit_targets | cycle_skips | root_unreachable_parent_skips | boundary_suffix_path_count_cap_hits | boundary_suffix_expansion_cap_hits |
|------|-------------------:|--------:|------------------:|---------------------------:|--------------------:|-------------------------------:|-------------------------------:|-----------------------------------:|---------------------------:|-------------------------------:|-------------------------:|------------------------:|-------------------------:|---------------------------:|--------------------------:|------------:|------------------------------:|------------------------------------:|-----------------------------------:|
| exact | 10 | 4 | 4 | 44 | 27 | 17 | 0.386364 | 0 | 0 | 2.118 | 63.000 | 63.000000 | 5.746 | 0 | 0 | 0 | 111 | 0 | 0 |

## Full Exact Splice Validation

This section compares boundary-stopped exact search plus suffix splicing against full filtered DFS to root on the same targets and path-length budgets. Comparable rows are uncapped on both sides and have measured suffix mass. For those rows, zero deltas mean the boundary condition reproduced full exact search for path mass, selected value, and mean path length.

| path_length_budget | rows | comparable_rows | exact_match_rows | max_abs_root_path_delta | max_abs_value_sum_delta | max_abs_mean_path_length_delta | boundary_partial_rows | full_partial_rows | mean_full_nodes_expanded |
|-------------------:|-----:|----------------:|-----------------:|------------------------:|------------------------:|-------------------------------:|----------------------:|------------------:|-------------------------:|
| 10 | 4 | 4 | 4 | 0.000 | 0.000000 | 0.000000 | 0 | 0 | 39.500 |

| target_node | path_length_budget | comparable | spliced_total_root_paths | full_root_paths | root_path_delta | spliced_total_value_sum | full_value_sum | value_sum_delta | spliced_mean_path_length | full_mean_path_length | mean_path_length_delta | boundary_partial | full_partial |
|------------:|-------------------:|------------|-------------------------:|----------------:|----------------:|------------------------:|---------------:|----------------:|-------------------------:|----------------------:|-----------------------:|------------------|--------------|
| 707612 | 10 | yes | 43.000 | 43 | 0.000 | 43.000000 | 43.000000 | 0.000000 | 6.395 | 6.395 | 0.000000 | no | no |
| 32026258 | 10 | yes | 10.000 | 10 | 0.000 | 10.000000 | 10.000000 | 0.000000 | 4.200 | 4.200 | 0.000000 | no | no |
| 37358201 | 10 | yes | 2.000 | 2 | 0.000 | 2.000000 | 2.000000 | 0.000000 | 3.500 | 3.500 | 0.000000 | no | no |
| 75467625 | 10 | yes | 8.000 | 8 | 0.000 | 8.000000 | 8.000000 | 0.000000 | 4.750 | 4.750 | 0.000000 | no | no |

## Target Rows

Each row is one target under one path-length budget. `root_paths` counts direct root-reaching terminals found before the search stops at a boundary. A row with `root_paths=0` and positive `boundary_hit_prefixes` is boundary-covered; it is not automatically root-unreachable. When boundary suffix mass is disabled, use these rows to judge boundary coverage rather than total root-path mass.

| mode | target_node | path_length_budget | terminal_prefixes | root_paths | boundary_hit_prefixes | boundary_hit_fraction | budget_exhausted_prefixes | filtered_dead_end_prefixes | mean_boundary_remaining_budget | completed | cycle_skips | root_unreachable_parent_skips | spliced_total_root_paths | spliced_total_value_sum | spliced_mean_path_length |
|------|------------:|-------------------:|------------------:|-----------:|----------------------:|----------------------:|--------------------------:|---------------------------:|-------------------------------:|----------:|------------:|------------------------------:|-------------------------:|------------------------:|-------------------------:|
| exact | 707612 | 10 | 30 | 19 | 11 | 0.366667 | 0 | 0 | 5.273 | yes | 0 | 73 | 43.000 | 43.000000 | 6.395 |
| exact | 32026258 | 10 | 10 | 7 | 3 | 0.300000 | 0 | 0 | 6.000 | yes | 0 | 28 | 10.000 | 10.000000 | 4.200 |
| exact | 37358201 | 10 | 1 | 0 | 1 | 1.000000 | 0 | 0 | 8.000 | yes | 0 | 4 | 2.000 | 2.000000 | 3.500 |
| exact | 75467625 | 10 | 3 | 1 | 2 | 0.666667 | 0 | 0 | 8.000 | yes | 0 | 6 | 8.000 | 8.000000 | 4.750 |
