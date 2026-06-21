# LMDB Exact Sparse Depth Sweep

Graph: `simplewiki_articles_exact_sparse_parent_distance_sweep_b10_20`

Root: `2`

Point cap: `50`

Tail epsilon: `0.01`

Target selection: `parent-distance`

## Selection

| L_max_bucket | candidate_nodes | selected_targets | root_reachable_targets | filtered_targets |
|------------:|-----------------------:|-----------------:|-----------------------:|-----------------:|
| 1 | 13529 | 50 | 50 | 0 |
| 2 | 1103 | 50 | 50 | 0 |
| 3 | 47 | 47 | 47 | 0 |
| 4 | 0 | 0 | 0 | 0 |
| 5 | 0 | 0 | 0 | 0 |
| 6 | 0 | 0 | 0 | 0 |

| root_cone_nodes | max_observed_child_depth | child_edges_examined | truncated_by_depth | truncated_by_nodes |
|----------------:|-------------------------:|---------------------:|--------------------|--------------------|
| 14680 | 3 | 14887 | no | no |

## Depth And Budget Buckets

| L_max_bucket | budget | rows | exact_sparse | exact_matches | mean_child_depth | mean_L_min | mean_L_max | mean_paths | max_paths | mean_eff_bins | max_eff_bins | pct_eff_bins_le_cap | mean_dfs_nodes | mean_rec_states | mean_state_ratio | mean_time_ratio | mean_break_even_hits |
|------------:|-------:|-----:|-------------:|--------------:|-----------------:|-----------:|-----------:|-----------:|----------:|--------------:|-------------:|--------------------:|---------------:|----------------:|-----------------:|----------------:|---------------------:|
| 1 | 10 | 50 | 50 | 50 | 1.000 | 1.000 | 1.000 | 1.000 | 1 | 1.000 | 1 | 100.000 | 5.180 | 4.180 | 0.785 | 1.625 | 1.000 |
| 1 | 20 | 50 | 50 | 50 | 1.000 | 1.000 | 1.000 | 1.000 | 1 | 1.000 | 1 | 100.000 | 5.180 | 4.180 | 0.785 | 1.514 | 1.000 |
| 2 | 10 | 50 | 50 | 50 | 1.680 | 1.680 | 2.000 | 1.360 | 3 | 1.320 | 2 | 100.000 | 10.540 | 9.180 | 0.865 | 1.779 | 0.995 |
| 2 | 20 | 50 | 50 | 50 | 1.680 | 1.680 | 2.000 | 1.360 | 3 | 1.320 | 2 | 100.000 | 10.540 | 9.180 | 0.865 | 2.351 | 0.995 |
| 3 | 10 | 47 | 47 | 47 | 2.553 | 2.553 | 3.000 | 1.277 | 3 | 1.447 | 3 | 100.000 | 12.468 | 11.170 | 0.891 | 1.860 | 1.022 |
| 3 | 20 | 47 | 47 | 47 | 2.553 | 2.553 | 3.000 | 1.277 | 3 | 1.447 | 3 | 100.000 | 12.468 | 11.170 | 0.891 | 2.291 | 1.022 |

## Interpretation

- `exact_sparse` counts reachable, uncapped rows where recurrence matched DFS, no cycle approximation was needed, and the effective support stayed under the point cap.
- `mean_break_even_hits` is state based: recurrence build states divided by saved states per cached hit, using effective bins as the cached evaluation cost.  It is a planning estimate, not a wall-clock guarantee.
- If mean effective bins stay far below the point cap, the cap is not the paid storage cost; exact sparse histograms remain the first representation to consider.
