# Root-Conditioned Parent Branching Profile

Graph: `simplewiki_articles_root_conditioned_regime_depth4`

Root: `2`

## Selection

| retained_nodes | max_observed_depth | child_edges_examined | truncated_by_depth | truncated_by_nodes | total_elapsed_ms |
|---------------:|-------------------:|---------------------:|--------------------|--------------------|-----------------:|
| 14680 | 3 | 14887 | no | no | 172.532 |

## Overall Moments

| degree_scope | nodes | mean_p | p95_p | p99_p | max_p | E[p^2]/E[p] | mean_excess | zero_parent_nodes |
|--------------|------:|-------:|------:|------:|------:|-------------:|------------:|------------------:|
| raw_full_graph | 14679 | 4.155 | 7.000 | 10.000 | 28 | 4.966093 | 3.966093 | 1 |
| root_conditioned | 14679 | 1.014 | 1.000 | 2.000 | 3 | 1.028750 | 0.028750 | 1 |
| outside_root | 14642 | 3.149 | 6.000 | 9.000 | 26 | 4.201484 | 3.201484 | 38 |

## Depth Buckets

| child_depth | nodes | raw_b | root_conditioned_b | mean_raw_p | mean_root_p | mean_outside_fraction |
|------------:|------:|------:|-------------------:|-----------:|------------:|----------------------:|
| 0 | 1 | n/a | n/a | 0.000 | 0.000 | 0.000 |
| 1 | 13720 | 4.820676 | 1.029173 | 4.055 | 1.014 | 0.704 |
| 2 | 923 | 6.530671 | 1.023555 | 5.670 | 1.012 | 0.769 |
| 3 | 36 | 4.190476 | 1.000000 | 3.500 | 1.000 | 0.632 |

## Notes

- `raw_full_graph` counts all parents of retained nodes.
- `root_conditioned` counts only parents that are also retained descendants of the chosen root.
- This profile is preprocessing evidence for the estimator; the query path should consume the resulting prior instead of discovering the root-conditioned subgraph online.
