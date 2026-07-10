# Product-Kalman Continuous Selective-Risk Protocol

Status: frozen before inspecting covariance-gate outcomes. Exploratory, not confirmatory.

## Question

The public holdout evaluation found that hop-conditioned Product-Kalman covariance improves continuous Gaussian
NLL and calibration, but its posterior mean does not improve the operator-family classifier. This follow-up asks
whether the predicted covariance is useful in its proper role: identifying continuous predictions that should be
accepted, deferred, or sent to a more expensive judge.

This protocol does not add Product-Kalman outputs to categorical source vectors and does not replace the registered
`JointPosterior` margin gate for categorical decisions.

## Fixed Inputs

- Use the committed enwiki and Pearltrees feature tables in
  `repro/product_kalman_public_holdout/` without regenerating labels or model readouts.
- Reuse the exact branch assignment, canonical identity closure, valid-split composition rules, seeds `0..39`,
  affine graph calibration, shrinkage `0.05`, jitter `1e-6`, and smooth hop-covariance fit from
  `PROTOCOL_product_kalman_public_holdout.md`.
- Evaluate each corpus separately. SimpleMind remains excluded.
- The first composition-valid seed is the primary split; all other valid seeds are stability analyses.

## Predicted Risk And Realized Loss

For held-out row `i`, let the hop-conditioned Product-Kalman prediction be

```text
mean:       mu_i
covariance: V_i
target:     y_i
```

The primary predicted risk is

```text
u_i = trace(V_i)
```

because a calibrated Gaussian predicts `E[||y_i - mu_i||^2] = trace(V_i)`. Lower `u_i` means greater confidence.
The realized primary loss is the matching total squared error

```text
L_i = ||y_i - mu_i||^2.
```

`log(det(V_i))` is reported as a secondary volume score but cannot determine the decision. It emphasizes joint
ellipse volume rather than the expected Euclidean loss used by the primary endpoint.

No topology, degree, target value, residual, or categorical correctness value is included in the confidence score.
Although `V_i` is hop-conditioned, its parameters are fitted only from calibration residuals.

## Selective Metrics

Sort held-out rows from lowest to highest predicted risk. Because `V(hop)` gives rows at the same hop identical
scores, define every partial tied block by its expected cumulative loss under a uniform random order within that
block. This makes the curve invariant to file order. At each retained coverage `k/n`, selective risk is the expected
mean realized loss among the first `k` rows. Report:

- risk at 25%, 50%, 75%, and 100% coverage, using `ceil(n * coverage)` rows;
- area under the risk-coverage curve (AURC), averaged over all `n` prefix risks;
- normalized AURC, defined as AURC divided by the full-coverage mean loss;
- Spearman rank correlation between predicted risk and realized loss, with average ranks for ties; and
- per-hop mean predicted risk, realized loss, and row count as a calibration diagnostic.

The no-information control is row-exchangeability: permute the predicted risks relative to fixed realized losses.
For the primary split, use `K=1000` permutations and seed `0`. Compute finite-permutation one-sided p-values with a
plus-one correction for:

1. Spearman correlation at least as large as observed; and
2. AURC at most as small as observed.

Also report percentile bootstrap intervals (`B=1000`, seed `0`) for Spearman correlation and normalized AURC by
resampling held-out rows and recomputing both statistics. Degenerate bootstrap samples with constant risk are
recorded and omitted from the correlation interval only.

## Stability

Repeat the trace-risk evaluation over every composition-valid split. These splits reuse pairs and are not
independent replications. Report the number and fraction of splits with:

- positive trace-risk versus realized-loss Spearman correlation; and
- normalized AURC below one.

The primary permutation tests remain the inferential endpoint; split direction is a robustness requirement.

## Decision Rule

Call hop-conditioned covariance **eligible as a continuous abstention gate** only if, independently in both enwiki
and Pearltrees:

- primary trace-risk Spearman is positive with one-sided permutation `p < 0.01`;
- primary trace-risk AURC beats the row-permutation null with one-sided `p < 0.01`;
- the bootstrap 95% upper bound for normalized AURC is below `1.0`; and
- at least 75% of valid splits have positive Spearman and normalized AURC below `1.0`.

Failure on any requirement is reported as evidence against deploying this covariance as a per-row gate. No
post-hoc alternative covariance scalar, coverage cutoff, split subset, or corpus pooling may repair a failure.

Passing this rule would support routing continuous `(D, S)` estimates only. It would not establish categorical
abstention quality, judge independence, or a causal interpretation of hop-dependent error.
