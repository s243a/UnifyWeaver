# LMDB Exact Sparse Depth Sweep

Graph: `simplewiki_articles_exact_sparse_depth_sweep_b10_20`

Root: `2`

Point cap: `50`

Tail epsilon: `0.01`

## Selection

| child_depth | sampled_frontier_nodes | selected_targets | root_reachable_targets | filtered_targets |
|------------:|-----------------------:|-----------------:|-----------------------:|-----------------:|
| 0 | 1 | 0 | 0 | 0 |
| 1 | 13720 | 50 | 50 | 0 |
| 2 | 1106 | 50 | 50 | 0 |
| 3 | 47 | 47 | 47 | 0 |
| 4 | 0 | 0 | 0 | 0 |

## Depth And Budget Buckets

| child_depth | budget | rows | exact_sparse | exact_matches | mean_paths | max_paths | mean_eff_bins | max_eff_bins | pct_eff_bins_le_cap | mean_dfs_nodes | mean_rec_states | mean_state_ratio | mean_time_ratio | mean_break_even_hits |
|------------:|-------:|-----:|-------------:|--------------:|-----------:|----------:|--------------:|-------------:|--------------------:|---------------:|----------------:|-----------------:|----------------:|---------------------:|
| 1 | 10 | 50 | 50 | 50 | 1.020 | 2 | 1.020 | 2 | 100.000 | 5.380 | 4.360 | 0.791 | 1.608 | 1.000 |
| 1 | 20 | 50 | 50 | 50 | 1.020 | 2 | 1.020 | 2 | 100.000 | 5.380 | 4.360 | 0.791 | 1.591 | 1.000 |
| 2 | 10 | 50 | 50 | 50 | 1.140 | 2 | 1.140 | 2 | 100.000 | 10.620 | 9.480 | 0.885 | 2.129 | 1.000 |
| 2 | 20 | 50 | 50 | 50 | 1.140 | 2 | 1.140 | 2 | 100.000 | 10.620 | 9.480 | 0.885 | 1.941 | 1.000 |
| 3 | 10 | 47 | 47 | 47 | 1.277 | 3 | 1.447 | 3 | 100.000 | 12.468 | 11.170 | 0.891 | 2.024 | 1.022 |
| 3 | 20 | 47 | 47 | 47 | 1.277 | 3 | 1.447 | 3 | 100.000 | 12.468 | 11.170 | 0.891 | 2.257 | 1.022 |

## Interpretation

- `exact_sparse` counts reachable, uncapped rows where recurrence matched DFS, no cycle approximation was needed, and the effective support stayed under the point cap.
- `mean_break_even_hits` is state based: recurrence build states divided by saved states per cached hit, using effective bins as the cached evaluation cost.  It is a planning estimate, not a wall-clock guarantee.
- If mean effective bins stay far below the point cap, the cap is not the paid storage cost; exact sparse histograms remain the first representation to consider.
