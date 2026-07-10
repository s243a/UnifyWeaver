# Public Product-Kalman Holdout Protocol

Status: frozen before comparative model results. Exploratory, not confirmatory.

## Scope

Evaluate the committed audited enwiki and Pearltrees campaign views separately. SimpleMind is excluded because its
local titles have not been approved for external judging. No pooled result may replace the per-corpus results.

## Fixed Inputs

- Pair, response, and title-policy artifacts: `repro/product_kalman_cross_corpus/` at the protocol commit.
- Judge targets: committed `gpt-5.5-low` audited response views; no labels are regenerated.
- Frozen encoder: `intfloat/e5-small-v2`, query prefix for the ancestor and passage prefix for the descendant.
- Frozen predictor checkpoint: `model_prod.pt`; no fine-tuning on campaign rows.
- Primary graph views: enwiki category DAG and Pearltrees path-local principal-parent lineages.

The materialized row table must retain pair IDs, endpoint IDs, canonical identity fields, branch/tree provenance,
hop, all fixed source values, continuous targets, and operator-family targets. Expensive e5/model inference is run
once; every candidate uses the same table.

## Sources And Targets

Continuous state and target:

```text
D = max(mu_fwd over subcategory, subtopic, element_of, super_category)
S = max(mu over see_also, assoc)
```

The fixed model prior is `(model_D, model_S)`, where `model_D` is the maximum of forward and reverse HIER readouts
and `model_S` is the SYM readout. The graph measurement is the primary-view upward random-walk hit probability,
affine-calibrated to `D` on calibration rows only, with `H = [1, 0]`. In a path-local tree this probability is one
for every retained descendant/ancestor pair; that loss of within-corpus discrimination is a result, not a reason to
substitute a different graph channel after seeing outcomes.

The categorical target is the argmax of three judge `applies` families:

- `directional`: maximum over the four directional relations;
- `symmetric`: maximum over `see_also` and `assoc`;
- `open_world`: maximum over `none` and `unknown`.

Ties use the fixed order above and are counted in the report.

## Split Contract

Use branch/tree-level assignment where available, then remove rows until canonical identity closure is strict. An
identity component joins endpoint IDs to frozen canonical titles; no component may occur in both calibration and
evaluation. Split seeds are `0..39`, evaluation branch fraction is `0.5`, and rows crossing the resulting identity
boundary are omitted.

The primary split is the first seed in ascending order with at least 80 calibration rows, 30 evaluation rows, all
three operator families in calibration, at least two operator families in evaluation, and all five hop levels on
both sides. This choice depends only on split composition, never model scores. Remaining valid seeds are stability
analyses.

## Continuous Ladder

Fit all means and covariance parameters on calibration rows only. Use shrinkage `0.05`, diagonal shrinkage target,
jitter `1e-6`, and mu-space errors.

1. fixed model prior;
2. independent Gaussian/Kalman control (`C = 0`);
3. correlated Product-Kalman with constant `P`, `R`, and `C`;
4. hop-conditioned correlated Product-Kalman using the existing smooth Cholesky `Sigma(hop)` parameterization.

Report held-out Gaussian NLL, MSE, Mahalanobis per dimension, squared-Mahalanobis q95, marginal PIT KS, and central
50/80/90/95 percent coverage. Report paired row-bootstrap NLL intervals (`B=1000`, seed `0`) on the primary split
and split-level direction/stability over all valid seeds.

## JointPosterior Ladder

Fit the registered multinomial logistic `JointPosterior` (`hidden=0`, seed `0`) on calibration rows. The baseline
source vector is fixed e5 forward/reverse similarity, model `D/S`, graph measurement, and hop. Compare that same
head after adding:

1. constant Product-Kalman posterior `D/S` means;
2. hop-conditioned Product-Kalman posterior `D/S` means.

Predicted covariance is not used as a per-row confidence weight. Confidence is the posterior top-1 minus top-2
margin. Report accuracy, categorical log-loss, ECE with 10 equal-width confidence bins, and margin-gated AURC with
`B=1000` percentile bootstrap intervals.

## Decision Rule

Product-Kalman is not promoted over the registered baseline unless, within each claimed corpus:

- its continuous held-out NLL improves over both the fixed prior and independent control, with the paired 95%
  interval above zero;
- mean marginal PIT KS and mean absolute central-coverage error are each no more than `0.02` above the better
  of the prior and independent controls, and absolute Mahalanobis-per-dimension distance from one is no more than
  `0.10` above the better control;
- adding its fused means lowers JointPosterior categorical log-loss and ECE; and
- the fused-feature AURC 95% interval lies below the baseline AURC point estimate.

Failure on any axis is reported as evidence against promotion, not repaired by post-hoc source, split, or model
variants. Single-judge targets and ontology differences remain limitations regardless of outcome.
