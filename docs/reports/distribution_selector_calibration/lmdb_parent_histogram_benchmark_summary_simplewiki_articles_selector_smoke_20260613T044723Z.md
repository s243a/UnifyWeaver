# LMDB Parent Histogram Benchmark

Graph: `simplewiki_articles_selector_smoke`

Root: `2`

## Selection

| child_depth | sampled_frontier_nodes |
|-------------|------------------------|
| 0 | 1 |
| 1 | 32 |
| 2 | 1 |
| 3 | 0 |
| 4 | 0 |

## Summary

| target_budget_rows | reachable_rows | capped_rows |
|--------------------|----------------|-------------|
| 3 | 3 | 0 |

## Histogram Cost By Budget

| budget | rows | reachable | mean_paths | p95_paths | max_paths | mean_bins | p95_bins | max_bins | mean_nodes_expanded | mean_cycle_skips | capped_rows |
|--------|------|-----------|------------|-----------|-----------|-----------|----------|----------|---------------------|------------------|-------------|
| 4 | 1 | 1 | 1.000 | 1.000 | 1 | 1.000 | 1.000 | 1 | 7.0 | 0.0 | 0 |
| 6 | 1 | 1 | 1.000 | 1.000 | 1 | 1.000 | 1.000 | 1 | 7.0 | 0.0 | 0 |
| 8 | 1 | 1 | 1.000 | 1.000 | 1 | 1.000 | 1.000 | 1 | 7.0 | 0.0 | 0 |

## Parent Degree Signal For Reachable Rows

| budget | rows | mean_full_p | b_full | mean_root_p | b_root |
|--------|------|-------------|--------|-------------|--------|
| 4 | 1 | 2.000 | 2.000000 | 1.000 | 1.000000 |
| 6 | 1 | 2.000 | 2.000000 | 1.000 | 1.000000 |
| 8 | 1 | 2.000 | 2.000000 | 1.000 | 1.000000 |

## Fit Error By Model

| model | rows | mean_l1 | p95_l1 | max_l1 | mean_cdf_error |
|-------|------|---------|--------|--------|----------------|
| binomial_fit | 3 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| shifted_gamma_fit | 3 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Packed Exact Candidate Selection

| representation | rows | mean_bytes | mean_cdf | mean_w1 |
|----------------|------|------------|----------|---------|
| packed_sparse_histogram | 6 | 36.000 | 0.000000 | 0.000000 |
| quantized_cdf_table | 6 | 26.000 | 0.000000 | 0.000000 |
| tail_pruned_histogram | 18 | 52.000 | 0.000000 | 0.000000 |

## Representation Policy Selection

| workload | selected | rows | mean_bytes | mean_cdf | mean_w1 |
|----------|----------|------|------------|----------|---------|
| prefix_mass | quantized_cdf_table | 6 | 26.000 | 0.000000 | 0.000000 |
| arbitrary_functional | packed_sparse_histogram | 6 | 36.000 | 0.000000 | 0.000000 |
