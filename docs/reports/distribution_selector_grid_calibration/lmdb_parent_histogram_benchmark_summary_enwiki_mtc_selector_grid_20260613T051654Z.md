# LMDB Parent Histogram Benchmark

Graph: `enwiki_mtc_selector_grid`

Root: `7345184`

## Selection

| child_depth | sampled_frontier_nodes |
|-------------|------------------------|
| 0 | 1 |
| 1 | 32 |
| 2 | 512 |
| 3 | 512 |
| 4 | 512 |

## Summary

| target_budget_rows | reachable_rows | capped_rows |
|--------------------|----------------|-------------|
| 90 | 90 | 0 |

## Histogram Cost By Budget

| budget | rows | reachable | mean_paths | p95_paths | max_paths | mean_bins | p95_bins | max_bins | mean_nodes_expanded | mean_cycle_skips | capped_rows |
|--------|------|-----------|------------|-----------|-----------|-----------|----------|----------|---------------------|------------------|-------------|
| 4 | 30 | 30 | 4.200 | 11.000 | 16 | 1.933 | 3.000 | 3 | 745.5 | 47.6 | 0 |
| 5 | 30 | 30 | 11.167 | 36.000 | 36 | 2.967 | 4.000 | 4 | 3861.3 | 512.1 | 0 |
| 6 | 30 | 30 | 30.000 | 81.000 | 110 | 4.033 | 5.000 | 5 | 18888.7 | 4009.3 | 0 |

## Parent Degree Signal For Reachable Rows

| budget | rows | mean_full_p | b_full | mean_root_p | b_root |
|--------|------|-------------|--------|-------------|--------|
| 4 | 30 | 4.300 | 4.798450 | 2.267 | 2.911765 |
| 5 | 30 | 4.300 | 4.798450 | 2.733 | 3.292683 |
| 6 | 30 | 4.300 | 4.798450 | 3.100 | 3.516129 |

## Fit Error By Model

| model | rows | mean_l1 | p95_l1 | max_l1 | mean_cdf_error |
|-------|------|---------|--------|--------|----------------|
| binomial_fit | 90 | 0.241081 | 0.814815 | 0.982400 | 0.068143 |
| shifted_gamma_fit | 90 | 0.834301 | 1.277469 | 1.352708 | 0.402378 |

## Packed Exact Candidate Selection

| representation | rows | mean_bytes | mean_cdf | mean_w1 |
|----------------|------|------------|----------|---------|
| packed_sparse_histogram | 180 | 57.867 | 0.000000 | 0.000000 |
| quantized_cdf_table | 180 | 29.956 | 0.000003 | 0.000006 |
| tail_pruned_histogram | 540 | 75.733 | 0.000000 | 0.000000 |

## Representation Policy Selection

| workload | selected | rows | mean_bytes | mean_cdf | mean_w1 |
|----------|----------|------|------------|----------|---------|
| prefix_mass | quantized_cdf_table | 180 | 29.956 | 0.000003 | 0.000006 |
| arbitrary_functional | packed_sparse_histogram | 180 | 57.867 | 0.000000 | 0.000000 |

## Representation Policy By Budget And Depth

| child_depth | budget | workload | model_rows | parametric_cdf_pass | selected_counts | mean_bins | capped_hist_rows |
|------------:|-------:|----------|-----------:|--------------------:|-----------------|----------:|-----------------:|
| 2 | 4 | prefix_mass | 20 | 2 | quantized_cdf_table:20 | 2.800 | 0 |
| 2 | 4 | arbitrary_functional | 20 | 2 | packed_sparse_histogram:20 | 2.800 | 0 |
| 2 | 5 | prefix_mass | 20 | 0 | quantized_cdf_table:20 | 3.800 | 0 |
| 2 | 5 | arbitrary_functional | 20 | 0 | packed_sparse_histogram:20 | 3.800 | 0 |
| 2 | 6 | prefix_mass | 20 | 0 | quantized_cdf_table:20 | 5.000 | 0 |
| 2 | 6 | arbitrary_functional | 20 | 0 | packed_sparse_histogram:20 | 5.000 | 0 |
| 3 | 4 | prefix_mass | 20 | 11 | quantized_cdf_table:20 | 1.900 | 0 |
| 3 | 4 | arbitrary_functional | 20 | 11 | packed_sparse_histogram:20 | 1.900 | 0 |
| 3 | 5 | prefix_mass | 20 | 0 | quantized_cdf_table:20 | 3.000 | 0 |
| 3 | 5 | arbitrary_functional | 20 | 0 | packed_sparse_histogram:20 | 3.000 | 0 |
| 3 | 6 | prefix_mass | 20 | 0 | quantized_cdf_table:20 | 4.000 | 0 |
| 3 | 6 | arbitrary_functional | 20 | 0 | packed_sparse_histogram:20 | 4.000 | 0 |
| 4 | 4 | prefix_mass | 20 | 19 | quantized_cdf_table:20 | 1.100 | 0 |
| 4 | 4 | arbitrary_functional | 20 | 19 | packed_sparse_histogram:20 | 1.100 | 0 |
| 4 | 5 | prefix_mass | 20 | 9 | quantized_cdf_table:20 | 2.100 | 0 |
| 4 | 5 | arbitrary_functional | 20 | 9 | packed_sparse_histogram:20 | 2.100 | 0 |
| 4 | 6 | prefix_mass | 20 | 0 | quantized_cdf_table:20 | 3.100 | 0 |
| 4 | 6 | arbitrary_functional | 20 | 0 | packed_sparse_histogram:20 | 3.100 | 0 |
