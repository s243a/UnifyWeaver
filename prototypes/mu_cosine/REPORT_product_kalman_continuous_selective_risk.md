# Product-Kalman Covariance Is Not A Public Cross-Corpus Abstention Gate

Status: exploratory frozen-protocol result. Do **not** use `trace(V(hop))` as a per-row continuous abstention gate.

## Question

The public holdout evaluation showed that hop-conditioned Product-Kalman covariance substantially improves
continuous NLL and aggregate calibration, while its posterior mean does not improve operator-family classification.
This follow-up tested the remaining proposed use: can the predicted covariance identify which individual `(D, S)`
predictions have larger squared error?

`PROTOCOL_product_kalman_continuous_selective_risk.md` was frozen before inspecting these gate outcomes.

## Endpoint

For each held-out row:

```text
predicted risk: trace(V_i)
realized loss:  ||y_i - mu_i||^2
```

Rows were sorted from lower to higher predicted risk. Partial tied hop blocks used expected loss under a uniform
within-block ordering, so file order could not improve the risk-coverage curve. The primary split used 1000
row-permutation tests and 1000 bootstrap resamples. All composition-valid splits supplied secondary stability.

## Primary Results

| corpus | Spearman | permutation p | normalized AURC | permutation p | bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| enwiki | +0.040 | 0.337 | 0.934 | 0.190 | [0.812, 1.059] |
| Pearltrees | +0.239 | 0.002 | 0.902 | 0.060 | [0.772, 1.043] |

Normalized AURC below one is favorable. The full frozen rule required both permutation endpoints, an AURC
bootstrap upper bound below one, and directional stability in at least 75% of valid splits in each corpus.

Enwiki fails every gate criterion. Its small primary advantage is compatible with random row assignment, the
Spearman bootstrap interval crosses zero, and average split behavior is slightly unfavorable.

Pearltrees is suggestive but does not pass. Trace risk has a positive primary association and directional stability,
but the selective AURC permutation test is `p=0.060` and its bootstrap interval crosses one. The preregistered rule
does not permit promoting that partial result or pooling it with enwiki.

## Coverage Curves

| corpus | 25% risk | 50% risk | 75% risk | 100% risk |
|---|---:|---:|---:|---:|
| enwiki | 0.0674 | 0.0738 | 0.0773 | 0.0778 |
| Pearltrees | 0.0863 | 0.0804 | 0.0921 | 0.0962 |

The non-monotone intermediate values are another warning against interpreting these coarse five-level scores as a
stable routing policy.

## Split Stability

| corpus | positive Spearman | normalized AURC below one | mean Spearman | mean normalized AURC |
|---|---:|---:|---:|---:|
| enwiki | 18/40 | 22/40 | -0.023 | 1.017 |
| Pearltrees | 33/39 | 35/39 | +0.193 | 0.896 |

Seeds reuse the same 250-row corpus and are not independent replications. They show robustness to branch assignment,
not new-data confirmation.

## Decision

**Do not deploy hop-conditioned covariance trace as a per-row continuous abstention gate.**

This does not contradict the prior NLL result. Calibration and discrimination answer different questions:

- a covariance can correctly describe average residual scale by hop and improve Gaussian likelihood;
- the same covariance can still be too coarse to rank individual errors within or across hop groups; and
- good confidence-region coverage does not imply good selective-risk ordering.

Accordingly, `V(hop)` remains supported for Gaussian likelihood, Kalman-style updates, and two-sided uncertainty
regions such as `mu +/- z * sqrt(diag(V))`. It is not supported as the query-level answer to “which prediction
should be deferred?” The Product-of-Experts intuition may supply a lower consensus estimate of `mu`, but covariance
supplies uncertainty around that estimate; neither object becomes an effective abstention score merely by existing.

## Secondary Volume Score

`log(det(V))` was frozen as a secondary diagnostic. Its normalized AURC is `0.955` on enwiki and `0.902` on
Pearltrees. It does not alter the decision. On Pearltrees it induces the same five-level ordering as trace.

## Implications

A future abstention model would need query-varying evidence beyond hop, fitted as a calibrated gate on out-of-fold
predictions. Candidate inputs could include posterior margin, source disagreement, or trained-neighbor support, but
they must enter a learned held-out combiner rather than hand-set confidence weights. Such a model requires a new
frozen protocol and cannot be recovered post hoc from this result.

Categorical decisions continue to use the registered `JointPosterior` margin gate. No categorical confidence claim
is made here.

## Limitations

- The predicted risk has only five distinct values because covariance is a smooth function of integer hop.
- Targets come from one non-deterministic LLM judge rather than independent human labels.
- Each corpus has only 250 rows, with smaller identity-disjoint evaluation subsets.
- Pearltrees' path-local graph measurement is constant and does not provide row-level graph discrimination.
- The experiment tests squared-error selection, not escalation benefit against an actually observed stronger judge.

## Reproducibility

Exact JSON, rendered summaries, and primary row arrays are archived in
`repro/product_kalman_continuous_selective_risk/`. They consume the previously committed fixed feature tables, so no
GPU inference or private title data is needed to rerun the statistics.
