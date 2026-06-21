# LMDB Exact Sparse Depth Sweep

Graph: `enwiki_mtc_exact_sparse_parent_distance_lmin_sweep_b10_20_smoke`

Root: `7345184`

Point cap: `50`

Tail epsilon: `0.01`

Target selection: `parent-distance`

Parent distance field: `L_min`

## Selection

| L_min_bucket | candidate_nodes | selected_targets | root_reachable_targets | filtered_targets |
|------------:|-----------------------:|-----------------:|-----------------------:|-----------------:|
| 1 | 35 | 8 | 8 | 0 |
| 2 | 1007 | 8 | 8 | 0 |
| 3 | 2092 | 8 | 8 | 0 |
| 4 | 1809 | 8 | 8 | 0 |
| 5 | 0 | 0 | 0 | 0 |
| 6 | 0 | 0 | 0 | 0 |
| 7 | 0 | 0 | 0 | 0 |
| 8 | 0 | 0 | 0 | 0 |

| root_cone_nodes | max_observed_child_depth | child_edges_examined | truncated_by_depth | truncated_by_nodes |
|----------------:|-------------------------:|---------------------:|--------------------|--------------------|
| 4944 | 4 | 31675 | yes | no |

## Depth And Budget Buckets

| L_min_bucket | budget | rows | exact_sparse | exact_matches | cycle_approx | dfs_capped | recurrence_capped | mean_child_depth | mean_L_min | mean_L_max | mean_paths | max_paths | mean_eff_bins | max_eff_bins | pct_eff_bins_le_cap | mean_dfs_nodes | mean_rec_states | mean_state_ratio | mean_time_ratio | mean_break_even_hits |
|------------:|-------:|-----:|-------------:|--------------:|-------------:|-----------:|------------------:|-----------------:|-----------:|-----------:|-----------:|----------:|--------------:|-------------:|--------------------:|---------------:|----------------:|-----------------:|----------------:|---------------------:|
| 1 | 10 | 8 | 0 | 2 | 7 | 7 | 0 | 1.000 | 1.000 | 9.000 | 1.286 | 3 | 1.143 | 2 | 100.000 | 100000.000 | 532.714 | 0.005 | 0.028 | 0.005 |
| 1 | 20 | 8 | 0 | 2 | 7 | 7 | 0 | 1.000 | 1.000 | 9.000 | 1.000 | 1 | 1.000 | 1 | 100.000 | 100000.000 | 1457.714 | 0.015 | 0.061 | 0.015 |
| 2 | 10 | 8 | 0 | 0 | 5 | 5 | 0 | 2.000 | 2.000 | 24.000 | 1.200 | 2 | 1.200 | 2 | 100.000 | 100000.000 | 683.200 | 0.007 | 0.039 | 0.007 |
| 2 | 20 | 8 | 0 | 0 | 5 | 5 | 0 | 2.000 | 2.000 | 24.000 | 1.000 | 1 | 1.000 | 1 | 100.000 | 100000.000 | 2178.800 | 0.022 | 0.109 | 0.022 |
| 3 | 10 | 8 | 0 | 0 | 5 | 5 | 0 | 3.000 | 3.000 | 24.000 | 31.000 | 85 | 4.800 | 8 | 100.000 | 100000.000 | 1766.000 | 0.018 | 0.108 | 0.018 |
| 3 | 20 | 8 | 0 | 0 | 1 | 1 | 1 | 3.000 | 3.000 | 24.000 | 1.000 | 1 | 1.000 | 1 | 100.000 | 100000.000 | 3657.000 | 0.037 | 0.223 | 0.037 |
| 4 | 10 | 8 | 0 | 0 | 4 | 4 | 0 | 4.000 | 4.000 | 24.000 | 40.750 | 79 | 4.750 | 7 | 100.000 | 100000.000 | 1590.500 | 0.016 | 0.106 | 0.016 |
| 4 | 20 | 8 | 0 | 0 | 1 | 1 | 1 | 4.000 | 4.000 | 24.000 | 1.000 | 1 | 1.000 | 1 | 100.000 | 100000.000 | 10137.000 | 0.101 | 0.712 | 0.101 |

## Interpretation

- `exact_sparse` counts reachable, uncapped rows where recurrence matched DFS, no cycle approximation was needed, and the effective support stayed under the point cap.
- `mean_break_even_hits` is state based: recurrence build states divided by saved states per cached hit, using effective bins as the cached evaluation cost.  It is a planning estimate, not a wall-clock guarantee.
- If mean effective bins stay far below the point cap, the cap is not the paid storage cost; exact sparse histograms remain the first representation to consider.
- Rows with DFS caps, recurrence caps, or cycle approximation are not exact histogram-shape evidence; their support and path-count columns describe only the capped or approximated run.
