# Product-Kalman continuous selective risk: enwiki

Status: exploratory frozen-protocol evaluation.

Primary seed `0`; `40` valid splits.

## Primary trace-covariance gate

- Spearman: `+0.0397`; permutation `p=0.336663`; bootstrap 95% CI `[-0.1212, +0.2016]`.
- Normalized AURC: `0.9338`; permutation `p=0.189810`; bootstrap 95% CI `[0.8116, 1.0592]`.
- Secondary log-determinant normalized AURC: `0.9549`.

| coverage | selective squared error |
|---:|---:|
| 25% | 0.06738 |
| 50% | 0.07377 |
| 75% | 0.07725 |
| 100% | 0.07784 |

## Split stability

- Positive Spearman: `18/40`.
- Normalized AURC below one: `22/40`.

## Decision

**do_not_use_as_gate**
- primary trace-risk Spearman does not pass the one-sided permutation rule
- primary trace-risk AURC does not pass the one-sided permutation rule
- normalized AURC bootstrap upper bound is not below one
- fewer than 75% of valid splits have positive Spearman
- fewer than 75% of valid splits have normalized AURC below one
