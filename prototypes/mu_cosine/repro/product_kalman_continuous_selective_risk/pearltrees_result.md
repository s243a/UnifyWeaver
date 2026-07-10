# Product-Kalman continuous selective risk: pearltrees

Status: exploratory frozen-protocol evaluation.

Primary seed `0`; `39` valid splits.

## Primary trace-covariance gate

- Spearman: `+0.2388`; permutation `p=0.001998`; bootstrap 95% CI `[+0.0481, +0.4092]`.
- Normalized AURC: `0.9022`; permutation `p=0.059940`; bootstrap 95% CI `[0.7717, 1.0428]`.
- Secondary log-determinant normalized AURC: `0.9022`.

| coverage | selective squared error |
|---:|---:|
| 25% | 0.08632 |
| 50% | 0.08037 |
| 75% | 0.09212 |
| 100% | 0.09622 |

## Split stability

- Positive Spearman: `33/39`.
- Normalized AURC below one: `35/39`.

## Decision

**do_not_use_as_gate**
- primary trace-risk AURC does not pass the one-sided permutation rule
- normalized AURC bootstrap upper bound is not below one
