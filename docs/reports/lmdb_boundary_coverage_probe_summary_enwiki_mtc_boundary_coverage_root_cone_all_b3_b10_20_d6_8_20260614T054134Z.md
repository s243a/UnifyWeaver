# LMDB Boundary Coverage Probe

Graph: `enwiki_mtc_boundary_coverage_root_cone_all_b3_b10_20_d6_8`

Root: `7345184`

Target selection: `root-cone-child-depth`

Selection source: `root-cone`

Parent filter: `root-cone`

Root cone depth: `20`

Root cone nodes: `15159`

Boundary suffix mass measured: `False`

Path value kernel: `count`

Path value branching factor: `n/a`

Path value power: `n/a`

Boundary nodes: `977`

Targets: `4`

Path length budgets: `10,20`

## How This Was Generated

- This older selection record stores observed boundary/target frontier counts over child-depths `3` and `6, 8`, but not the requested depth arguments separately.
- This selection record predates sampler-limit provenance fields; newer JSONL records include child/frontier/per-depth sampling limits directly.
- Mode `exact` controls the row generator: `exact` enumerates all simple parent prefixes until root, boundary, or budget; `sample` performs branch-product weighted boundary-stopped random walks; `root-sample` samples walks to root without stopping at boundaries.
- Parent filter `root-cone` is applied during parent expansion. `root-cone` accepts only parents inside the precomputed root cone whose cone depth fits within the remaining path budget; `root-reachable` uses recursive finite-horizon reachability; `all` does no root-scope pruning.
- The root cone was built to child depth `20` with `15159` nodes, `root_cone_children_per_node=32`, and `root_cone_frontier_limit=1000`.
- Boundary suffix mass measured: `False`. When this is false, boundary-hit rows measure coverage only; they do not splice cached suffix mass into a total root-path estimate.
- Path value kernel `count` defines the functional being estimated after path coverage is known. It is separate from the random-walk proposal correction.

## Table Guide

- `Selection` lists observed frontier sizes for boundary and target depths. In newer reports, the requested depths are stated above; the table may include intermediate traversal depths.
- `Root Cone` shows the bounded child-reachable cone used for root-cone filtering. These are not parent-path counts; they are child-depth frontier counts from the root.
- `Coverage Summary` aggregates observed terminal outcomes by mode and path-length budget. In exact mode these are enumerated simple-prefix counts; in sample/root-sample modes they are raw sample outcomes.
- `Boundary Sample Estimates` and `Root Path Sample Estimates` contain branch-product weighted estimates. Use those estimate tables, not raw observed sample counts, when reasoning about path-space size.
- `Target Rows` is per target and budget. `root_paths` counts direct root terminals reached before a boundary stop; `boundary_hit_prefixes` counts prefixes where the boundary condition would take over.
- `root_unreachable_parent_skips` counts parent edges rejected by the active parent filter. Under `root-cone`, that includes parents outside the cone or too deep for the remaining budget, not only globally unreachable parents.
- `budget_exhausted_prefixes`, `path_count_cap_hit_targets`, and `expansion_cap_hit_targets` identify rows whose result is limited by the path budget or safety caps.

## Result Implications

- Target evaluation completed `8/8` rows without path-count or expansion caps.
- `exact` budget `10`: `33` terminal prefixes, `24` direct root paths, `8` boundary-hit prefixes, boundary-hit fraction `0.242424`, and `0` budget-exhausted prefixes.
- `exact` budget `20`: `33` terminal prefixes, `24` direct root paths, `8` boundary-hit prefixes, boundary-hit fraction `0.242424`, and `0` budget-exhausted prefixes.
- `6` target-budget rows have `root_paths=0` and positive boundary hits. In this report that means enumeration stopped at a boundary before reaching root; it is boundary coverage, not evidence that those targets lack root paths.
- Boundary suffix mass was not measured, so boundary hits cannot yet be converted into total root-path mass or budgeted CDF mass from this report alone.
- The active parent filter rejected `308` parent edges. Under `root-cone`, these skips are part of the scoped experiment definition, not necessarily data errors.
- Simple-path cycle checks skipped `4` edges. That keeps rows cycle-free, but it also means cached suffixes must be interpreted with the same cycle policy.
- Filtered dead ends occurred `2` times; these are prefixes whose remaining parents were removed by the active parent filter.

## Selection

| role | child_depth | sampled_frontier_nodes |
|------|-------------|------------------------|
| boundary | 3 | 977 |
| target | 6 | 990 |
| target | 8 | 989 |

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

For sample and root-sample modes, these are observed random-walk outcomes. Use the estimate sections below for branch-product weighted path-space estimates.

| mode | path_length_budget | targets | completed_targets | observed_terminal_prefixes | observed_root_paths | observed_boundary_hit_prefixes | observed_boundary_hit_fraction | observed_budget_exhausted_prefixes | observed_filtered_dead_ends | mean_boundary_suffix_path_mass | mean_boundary_suffix_path_value | path_count_cap_hit_targets | expansion_cap_hit_targets | cycle_skips | root_unreachable_parent_skips |
|------|-------------------:|--------:|------------------:|------------------:|-----------:|----------------------:|----------------------:|--------------------------:|----------------------------:|-------------------------------:|--------------------------------:|---------------------------:|--------------------------:|------------:|------------------------------:|
| exact | 10 | 4 | 4 | 33 | 24 | 8 | 0.242424 | 0 | 1 | n/a | n/a | 0 | 0 | 2 | 154 |
| exact | 20 | 4 | 4 | 33 | 24 | 8 | 0.242424 | 0 | 1 | n/a | n/a | 0 | 0 | 2 | 154 |

## Target Rows

Each row is one target under one path-length budget. `root_paths` counts direct root-reaching terminals found before the search stops at a boundary. A row with `root_paths=0` and positive `boundary_hit_prefixes` is boundary-covered; it is not automatically root-unreachable. When boundary suffix mass is disabled, use these rows to judge boundary coverage rather than total root-path mass.

| mode | target_node | path_length_budget | terminal_prefixes | root_paths | boundary_hit_prefixes | boundary_hit_fraction | budget_exhausted_prefixes | filtered_dead_end_prefixes | mean_boundary_remaining_budget | completed | cycle_skips | root_unreachable_parent_skips |
|------|------------:|-------------------:|------------------:|-----------:|----------------------:|----------------------:|--------------------------:|---------------------------:|-------------------------------:|----------:|------------:|------------------------------:|
| exact | 884414 | 10 | 2 | 0 | 2 | 1.000000 | 0 | 0 | 8.000 | yes | 0 | 13 |
| exact | 884414 | 20 | 2 | 0 | 2 | 1.000000 | 0 | 0 | 18.000 | yes | 0 | 13 |
| exact | 55649431 | 10 | 1 | 0 | 1 | 1.000000 | 0 | 0 | 7.000 | yes | 0 | 13 |
| exact | 55649431 | 20 | 1 | 0 | 1 | 1.000000 | 0 | 0 | 17.000 | yes | 0 | 13 |
| exact | 30137594 | 10 | 3 | 0 | 3 | 1.000000 | 0 | 0 | 5.667 | yes | 0 | 30 |
| exact | 30137594 | 20 | 3 | 0 | 3 | 1.000000 | 0 | 0 | 15.667 | yes | 0 | 30 |
| exact | 44532299 | 10 | 27 | 24 | 2 | 0.074074 | 0 | 1 | 5.500 | yes | 2 | 98 |
| exact | 44532299 | 20 | 27 | 24 | 2 | 0.074074 | 0 | 1 | 15.500 | yes | 2 | 98 |
