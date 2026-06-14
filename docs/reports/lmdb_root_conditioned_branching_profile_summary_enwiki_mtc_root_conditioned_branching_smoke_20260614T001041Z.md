# Root-Conditioned Parent Branching Profile

Graph: `enwiki_mtc_root_conditioned_branching_smoke`

Root: `7345184`

## Selection

| retained_nodes | max_observed_depth | child_edges_examined | truncated_by_depth | truncated_by_nodes | total_elapsed_ms |
|---------------:|-------------------:|---------------------:|--------------------|--------------------|-----------------:|
| 5000 | 3 | 56867 | no | yes | 3277.627 |

## Overall Moments

| degree_scope | nodes | mean_p | p95_p | p99_p | max_p | E[p^2]/E[p] | mean_excess | zero_parent_nodes |
|--------------|------:|-------:|------:|------:|------:|-------------:|------------:|------------------:|
| raw_full_graph | 5000 | 4.152 | 8.000 | 11.000 | 202 | 7.048550 | 6.048550 | 0 |
| root_conditioned | 4999 | 1.698 | 3.000 | 5.000 | 9 | 2.164035 | 1.164035 | 1 |
| outside_root | 4564 | 2.690 | 6.000 | 8.000 | 197 | 7.009775 | 6.009775 | 436 |

## Depth Buckets

| child_depth | nodes | raw_b | root_conditioned_b | mean_raw_p | mean_root_p | mean_outside_fraction |
|------------:|------:|------:|-------------------:|-----------:|------------:|----------------------:|
| 0 | 1 | 3.000000 | n/a | 3.000 | 0.000 | 1.000 |
| 1 | 35 | 4.675676 | 2.621622 | 4.229 | 2.114 | 0.489 |
| 2 | 1007 | 4.818951 | 2.349706 | 3.993 | 1.860 | 0.495 |
| 3 | 3957 | 7.610850 | 2.105674 | 4.193 | 1.653 | 0.546 |

## Notes

- `raw_full_graph` counts all parents of retained nodes.
- `root_conditioned` counts only parents that are also retained descendants of the chosen root.
- This profile is preprocessing evidence for the estimator; the query path should consume the resulting prior instead of discovering the root-conditioned subgraph online.
