# Public Product-Kalman holdout: enwiki

Status: exploratory frozen-protocol evaluation.

Primary split seed: `0`; calibration/evaluation rows: `122/124`.

## Continuous evaluation

| variant | NLL | MSE | Mahal/dim | q95 | mean PIT KS | coverage MAE |
|---|---:|---:|---:|---:|---:|---:|
| `prior` | +0.9479 | 0.2313 | 2.332 | 11.400 | 0.439 | 0.256 |
| `independent_kalman` | +0.3314 | 0.1039 | 2.240 | 10.701 | 0.372 | 0.198 |
| `product_kalman` | +0.3480 | 0.0990 | 2.074 | 10.656 | 0.332 | 0.174 |
| `hop_product_kalman` | -0.5058 | 0.0778 | 1.079 | 5.646 | 0.149 | 0.038 |

Paired primary-split NLL gains (positive favors candidate):

- `independent_kalman_to_hop_product_kalman`: +0.8372 [+0.5995, +1.0898]
- `independent_kalman_to_product_kalman`: -0.0166 [-0.1216, +0.0948]
- `prior_to_hop_product_kalman`: +1.4537 [+1.1595, +1.7365]
- `prior_to_product_kalman`: +0.5999 [+0.4547, +0.7446]
- `product_kalman_to_hop_product_kalman`: +0.8538 [+0.6282, +1.0779]

## JointPosterior evaluation

| variant | accuracy | log-loss | ECE-10 | margin AURC (95% CI) |
|---|---:|---:|---:|---:|
| `joint_baseline` | 0.677 | 0.872 | 0.144 | 0.157 [0.099, 0.236] |
| `joint_plus_constant` | 0.677 | 0.874 | 0.144 | 0.156 [0.098, 0.234] |
| `joint_plus_hop` | 0.685 | 0.886 | 0.149 | 0.155 [0.097, 0.232] |

## Decision

**do_not_promote**
- fused JointPosterior log-loss does not improve
- fused JointPosterior ECE does not improve
- fused AURC interval is not below baseline point estimate

The targets come from one non-deterministic LLM judge. Split stability is secondary because seeds reuse
the same 250-pair corpus; it is not an independent replication.
