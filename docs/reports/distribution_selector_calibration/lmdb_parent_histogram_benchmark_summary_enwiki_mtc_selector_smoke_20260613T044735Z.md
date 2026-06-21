# LMDB Parent Histogram Benchmark

Graph: `enwiki_mtc_selector_smoke`

Root: `7345184`

## Selection

| child_depth | sampled_frontier_nodes |
|-------------|------------------------|
| 0 | 1 |
| 1 | 32 |
| 2 | 256 |
| 3 | 256 |
| 4 | 256 |

## Summary

| target_budget_rows | reachable_rows | capped_rows |
|--------------------|----------------|-------------|
| 45 | 42 | 15 |

## Histogram Cost By Budget

| budget | rows | reachable | mean_paths | p95_paths | max_paths | mean_bins | p95_bins | max_bins | mean_nodes_expanded | mean_cycle_skips | capped_rows |
|--------|------|-----------|------------|-----------|-----------|-----------|----------|----------|---------------------|------------------|-------------|
| 4 | 15 | 15 | 3.800 | 8.000 | 11 | 1.933 | 3.000 | 3 | 601.9 | 47.3 | 0 |
| 6 | 15 | 15 | 18.800 | 42.000 | 68 | 3.800 | 5.000 | 5 | 14773.5 | 3520.9 | 0 |
| 8 | 15 | 12 | 20.167 | 43.000 | 66 | 4.583 | 7.000 | 7 | 50000.0 | 19280.3 | 15 |

## Parent Degree Signal For Reachable Rows

| budget | rows | mean_full_p | b_full | mean_root_p | b_root |
|--------|------|-------------|--------|-------------|--------|
| 4 | 15 | 3.800 | 4.263158 | 2.067 | 2.935484 |
| 6 | 15 | 3.800 | 4.263158 | 2.733 | 3.292683 |
| 8 | 12 | 4.083 | 4.510204 | 2.917 | 3.514286 |

## Fit Error By Model

| model | rows | mean_l1 | p95_l1 | max_l1 | mean_cdf_error |
|-------|------|---------|--------|--------|----------------|
| binomial_fit | 42 | 0.309813 | 1.044447 | 1.286054 | 0.091852 |
| shifted_gamma_fit | 42 | 0.734725 | 1.087337 | 1.434367 | 0.345684 |

## Packed Exact Candidate Selection

| representation | rows | mean_bytes | mean_cdf | mean_w1 |
|----------------|------|------------|----------|---------|
| packed_sparse_histogram | 84 | 60.857 | 0.000000 | 0.000000 |
| quantized_cdf_table | 84 | 30.714 | 0.000004 | 0.000009 |
| tail_pruned_histogram | 252 | 80.286 | 0.000000 | 0.000000 |

## Representation Policy Selection

| workload | selected | rows | mean_bytes | mean_cdf | mean_w1 |
|----------|----------|------|------------|----------|---------|
| prefix_mass | quantized_cdf_table | 84 | 30.714 | 0.000004 | 0.000009 |
| arbitrary_functional | packed_sparse_histogram | 84 | 60.857 | 0.000000 | 0.000000 |
