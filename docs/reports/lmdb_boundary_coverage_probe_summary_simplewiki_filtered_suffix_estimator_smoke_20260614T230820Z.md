# LMDB Boundary Coverage Probe

Graph: `simplewiki_filtered_suffix_estimator_smoke`

Root: `2`

Target selection: `root-cone-child-depth`

Selection source: `root-cone`

Parent filter: `root-cone`

Root cone depth: `4`

Root cone nodes: `1050`

Boundary suffix mass measured: `True`

Path value kernel: `count`

Path value branching factor: `n/a`

Path value power: `n/a`

Boundary nodes: `8`

Targets: `3`

Path length budgets: `4`

## How This Was Generated

- Boundary candidates were sampled from requested child depth(s) `1` and target rows from requested child depth(s) `2` using selection source `root-cone`.
- Boundary and target nodes were sampled from the precomputed root-cone depth buckets with per-depth limits `boundaries_per_depth=5` and `targets_per_depth=3`.
- Mode `exact` controls the row generator: `exact` enumerates all simple parent prefixes until root, boundary, or budget; `sample` performs branch-product weighted boundary-stopped random walks; `root-sample` samples walks to root without stopping at boundaries.
- Parent filter `root-cone` is applied during parent expansion. `root-cone` accepts only parents inside the precomputed root cone whose cone depth fits within the remaining path budget; `root-reachable` uses recursive finite-horizon reachability; `all` does no root-scope pruning.
- The root cone was built to child depth `4` with `1050` nodes, `root_cone_children_per_node=0`, and `root_cone_frontier_limit=1000`.
- Target-ancestor boundary inclusion was enabled, with `target_ancestor_boundary_limit=10`.
- Boundary suffix mass measured: `True`. When this is false, boundary-hit rows measure coverage only; they do not splice cached suffix mass into a total root-path estimate.
- Path value kernel `count` defines the functional being estimated after path coverage is known. It is separate from the random-walk proposal correction.

## Table Guide

- `Selection` lists observed frontier sizes for boundary and target depths. In newer reports, the requested depths are stated above; the table may include intermediate traversal depths.
- `Root Cone` shows the bounded child-reachable cone used for root-cone filtering. These are not parent-path counts; they are child-depth frontier counts from the root.
- `Coverage Summary` aggregates observed terminal outcomes by mode and path-length budget. In exact mode these are enumerated simple-prefix counts; in sample/root-sample modes they are raw sample outcomes.
- `spliced_total_root_paths`, `spliced_total_value_sum`, and `spliced_mean_path_length` are boundary-aware estimates: direct root terminals plus suffix histogram mass/value from boundary hits.
- `Boundary Sample Estimates` and `Root Path Sample Estimates` contain branch-product weighted estimates. Use those estimate tables, not raw observed sample counts, when reasoning about path-space size.
- `Target Rows` is per target and budget. `root_paths` counts direct root terminals reached before a boundary stop; `boundary_hit_prefixes` counts prefixes where the boundary condition would take over.
- `root_unreachable_parent_skips` counts parent edges rejected by the active parent filter. Under `root-cone`, that includes parents outside the cone or too deep for the remaining budget, not only globally unreachable parents.
- `budget_exhausted_prefixes`, `path_count_cap_hit_targets`, and `expansion_cap_hit_targets` identify rows whose result is limited by the path budget or safety caps.

## Result Implications

- Target evaluation completed `3/3` rows without path-count or expansion caps.
- `exact` budget `4`: `3` terminal prefixes, `0` direct root paths, `3` boundary-hit prefixes, boundary-hit fraction `1.000000`, `0` budget-exhausted prefixes, and spliced root mass `3.000`.
- `3` target-budget rows have `root_paths=0` and positive boundary hits. In this report that means enumeration stopped at a boundary before reaching root; it is boundary coverage, not evidence that those targets lack root paths.
- Boundary suffix mass was measured, so boundary-hit prefixes are combined with suffix histograms to estimate total root-path mass, aggregate value, and mean path length under the remaining budget.
- The active parent filter rejected `15` parent edges. Under `root-cone`, these skips are part of the scoped experiment definition, not necessarily data errors.

## Selection

| role | child_depth | sampled_frontier_nodes |
|------|-------------|------------------------|
| boundary | 1 | 1000 |
| target | 2 | 49 |

## Root Cone

| child_depth | new_nodes |
|------------:|----------:|
| 0 | 1 |
| 1 | 1000 |
| 2 | 49 |
| 3 | 0 |

## Coverage Summary

For sample and root-sample modes, these are observed random-walk outcomes. Use the estimate sections below for branch-product weighted path-space estimates.

| mode | path_length_budget | targets | completed_targets | observed_terminal_prefixes | observed_root_paths | observed_boundary_hit_prefixes | observed_boundary_hit_fraction | observed_budget_exhausted_prefixes | observed_filtered_dead_ends | mean_boundary_suffix_path_mass | spliced_total_root_paths | spliced_total_value_sum | spliced_mean_path_length | path_count_cap_hit_targets | expansion_cap_hit_targets | cycle_skips | root_unreachable_parent_skips | boundary_suffix_path_count_cap_hits | boundary_suffix_expansion_cap_hits |
|------|-------------------:|--------:|------------------:|---------------------------:|--------------------:|-------------------------------:|-------------------------------:|-----------------------------------:|---------------------------:|-------------------------------:|-------------------------:|------------------------:|-------------------------:|---------------------------:|--------------------------:|------------:|------------------------------:|------------------------------------:|-----------------------------------:|
| exact | 4 | 3 | 3 | 3 | 0 | 3 | 1.000000 | 0 | 0 | 1.000 | 3.000 | 3.000000 | 2.000 | 0 | 0 | 0 | 15 | 0 | 0 |

## Target Rows

Each row is one target under one path-length budget. `root_paths` counts direct root-reaching terminals found before the search stops at a boundary. A row with `root_paths=0` and positive `boundary_hit_prefixes` is boundary-covered; it is not automatically root-unreachable. When boundary suffix mass is disabled, use these rows to judge boundary coverage rather than total root-path mass.

| mode | target_node | path_length_budget | terminal_prefixes | root_paths | boundary_hit_prefixes | boundary_hit_fraction | budget_exhausted_prefixes | filtered_dead_end_prefixes | mean_boundary_remaining_budget | completed | cycle_skips | root_unreachable_parent_skips | spliced_total_root_paths | spliced_total_value_sum | spliced_mean_path_length |
|------|------------:|-------------------:|------------------:|-----------:|----------------------:|----------------------:|--------------------------:|---------------------------:|-------------------------------:|----------:|------------:|------------------------------:|-------------------------:|------------------------:|-------------------------:|
| exact | 47113 | 4 | 1 | 0 | 1 | 1.000000 | 0 | 0 | 3.000 | yes | 0 | 5 | 1.000 | 1.000000 | 2.000 |
| exact | 50121 | 4 | 1 | 0 | 1 | 1.000000 | 0 | 0 | 3.000 | yes | 0 | 5 | 1.000 | 1.000000 | 2.000 |
| exact | 51600 | 4 | 1 | 0 | 1 | 1.000000 | 0 | 0 | 3.000 | yes | 0 | 5 | 1.000 | 1.000000 | 2.000 |
