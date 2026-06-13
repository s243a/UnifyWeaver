# LMDB Parent Histogram Benchmark

Graph: `simplewiki_articles_selector_grid`

Root: `2`

## Selection

| child_depth | sampled_frontier_nodes |
|-------------|------------------------|
| 0 | 1 |
| 1 | 512 |
| 2 | 32 |
| 3 | 0 |

## Summary

| target_budget_rows | reachable_rows | capped_rows |
|--------------------|----------------|-------------|
| 60 | 60 | 0 |

## Histogram Cost By Budget

| budget | rows | reachable | mean_paths | p95_paths | max_paths | mean_bins | p95_bins | max_bins | mean_nodes_expanded | mean_cycle_skips | capped_rows |
|--------|------|-----------|------------|-----------|-----------|-----------|----------|----------|---------------------|------------------|-------------|
| 4 | 20 | 20 | 1.050 | 1.000 | 2 | 1.050 | 1.000 | 2 | 8.3 | 0.0 | 0 |
| 5 | 20 | 20 | 1.050 | 1.000 | 2 | 1.050 | 1.000 | 2 | 8.3 | 0.0 | 0 |
| 6 | 20 | 20 | 1.050 | 1.000 | 2 | 1.050 | 1.000 | 2 | 8.3 | 0.0 | 0 |

## Parent Degree Signal For Reachable Rows

| budget | rows | mean_full_p | b_full | mean_root_p | b_root |
|--------|------|-------------|--------|-------------|--------|
| 4 | 20 | 4.950 | 5.606061 | 1.050 | 1.095238 |
| 5 | 20 | 4.950 | 5.606061 | 1.050 | 1.095238 |
| 6 | 20 | 4.950 | 5.606061 | 1.050 | 1.095238 |

## Fit Error By Model

| model | rows | mean_l1 | p95_l1 | max_l1 | mean_cdf_error |
|-------|------|---------|--------|--------|----------------|
| binomial_fit | 60 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| shifted_gamma_fit | 60 | 0.038080 | 0.000000 | 0.761594 | 0.019040 |

## Packed Exact Candidate Selection

| representation | rows | mean_bytes | mean_cdf | mean_w1 |
|----------------|------|------------|----------|---------|
| packed_sparse_histogram | 120 | 36.600 | 0.000000 | 0.000000 |
| quantized_cdf_table | 120 | 26.100 | 0.000000 | 0.000000 |
| tail_pruned_histogram | 360 | 52.600 | 0.000000 | 0.000000 |

## Representation Policy Selection

| workload | selected | rows | mean_bytes | mean_cdf | mean_w1 |
|----------|----------|------|------------|----------|---------|
| prefix_mass | quantized_cdf_table | 120 | 26.100 | 0.000000 | 0.000000 |
| arbitrary_functional | packed_sparse_histogram | 120 | 36.600 | 0.000000 | 0.000000 |

## Representation Policy By Budget And Depth

| child_depth | budget | workload | model_rows | parametric_cdf_pass | selected_counts | mean_bins | capped_hist_rows |
|------------:|-------:|----------|-----------:|--------------------:|-----------------|----------:|-----------------:|
| 1 | 4 | prefix_mass | 20 | 20 | quantized_cdf_table:20 | 1.000 | 0 |
| 1 | 4 | arbitrary_functional | 20 | 20 | packed_sparse_histogram:20 | 1.000 | 0 |
| 1 | 5 | prefix_mass | 20 | 20 | quantized_cdf_table:20 | 1.000 | 0 |
| 1 | 5 | arbitrary_functional | 20 | 20 | packed_sparse_histogram:20 | 1.000 | 0 |
| 1 | 6 | prefix_mass | 20 | 20 | quantized_cdf_table:20 | 1.000 | 0 |
| 1 | 6 | arbitrary_functional | 20 | 20 | packed_sparse_histogram:20 | 1.000 | 0 |
| 2 | 4 | prefix_mass | 20 | 19 | quantized_cdf_table:20 | 1.100 | 0 |
| 2 | 4 | arbitrary_functional | 20 | 19 | packed_sparse_histogram:20 | 1.100 | 0 |
| 2 | 5 | prefix_mass | 20 | 19 | quantized_cdf_table:20 | 1.100 | 0 |
| 2 | 5 | arbitrary_functional | 20 | 19 | packed_sparse_histogram:20 | 1.100 | 0 |
| 2 | 6 | prefix_mass | 20 | 19 | quantized_cdf_table:20 | 1.100 | 0 |
| 2 | 6 | arbitrary_functional | 20 | 19 | packed_sparse_histogram:20 | 1.100 | 0 |
