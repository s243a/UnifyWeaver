# Public Product-Kalman holdout: pearltrees

Status: exploratory frozen-protocol evaluation.

Primary split seed: `0`; calibration/evaluation rows: `93/112`.

## Continuous evaluation

| variant | NLL | MSE | Mahal/dim | q95 | mean PIT KS | coverage MAE |
|---|---:|---:|---:|---:|---:|---:|
| `prior` | +1.7661 | 0.2759 | 2.863 | 13.700 | 0.525 | 0.280 |
| `independent_kalman` | +1.2163 | 0.1541 | 2.838 | 13.000 | 0.467 | 0.262 |
| `product_kalman` | +1.3873 | 0.1607 | 2.907 | 13.561 | 0.451 | 0.275 |
| `hop_product_kalman` | -0.2647 | 0.0962 | 0.853 | 4.232 | 0.194 | 0.043 |

Paired primary-split NLL gains (positive favors candidate):

- `independent_kalman_to_hop_product_kalman`: +1.4810 [+1.1352, +1.8597]
- `independent_kalman_to_product_kalman`: -0.1710 [-0.2626, -0.0785]
- `prior_to_hop_product_kalman`: +2.0308 [+1.6598, +2.4308]
- `prior_to_product_kalman`: +0.3788 [+0.2526, +0.5154]
- `product_kalman_to_hop_product_kalman`: +1.6520 [+1.2905, +2.0184]

## JointPosterior evaluation

| variant | accuracy | log-loss | ECE-10 | margin AURC (95% CI) |
|---|---:|---:|---:|---:|
| `joint_baseline` | 0.545 | 0.915 | 0.119 | 0.308 [0.216, 0.419] |
| `joint_plus_constant` | 0.545 | 0.915 | 0.119 | 0.308 [0.216, 0.420] |
| `joint_plus_hop` | 0.518 | 0.953 | 0.148 | 0.328 [0.229, 0.449] |

## Decision

**do_not_promote**
- fused JointPosterior log-loss does not improve
- fused JointPosterior ECE does not improve
- fused AURC interval is not below baseline point estimate

The targets come from one non-deterministic LLM judge. Split stability is secondary because seeds reuse
the same 250-pair corpus; it is not an independent replication.
