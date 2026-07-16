# Product-Kalman Public Holdout Evaluation

Status: exploratory frozen-protocol result. Product-Kalman is **not promoted** over the registered
`JointPosterior` baseline.

## Question

Does a covariance-aware Product-Kalman update improve held-out continuous prediction and, if so, do its
posterior means add useful information to the registered operator-family classifier?

The protocol was frozen in `PROTOCOL_product_kalman_public_holdout.md` before comparative outcomes were
computed. Enwiki and Pearltrees were evaluated separately. SimpleMind was excluded because its local titles
were not approved for external judging.

## Inputs And Splits

- Each corpus contributes 250 audited pairs balanced across hops 1 through 5.
- Targets are fixed `gpt-5.5-low` judge labels from the committed public cross-corpus campaign.
- The prior is the frozen `model_prod.pt` directional/symmetric readout.
- Enwiki uses streamed access to the scoped category LMDB. Its graph measurement has 196 distinct values.
- Pearltrees uses pair-specific principal-parent paths. Every retained ancestor pair has graph measurement one,
  so affine calibration reduces this channel to the calibration-set mean target.
- Canonical identity closure is strict across calibration and evaluation.
- Enwiki has 40 valid splits. Pearltrees has 39; seed 3 fails the frozen composition requirements.
- Seed 0 is the composition-selected primary split for both corpora.

## Continuous Result

Positive NLL gain favors the candidate.

| corpus | variant | primary NLL | Mahal/dim | mean PIT KS | coverage MAE |
|---|---|---:|---:|---:|---:|
| enwiki | prior | +0.948 | 2.332 | 0.439 | 0.256 |
| enwiki | independent Kalman | +0.331 | 2.240 | 0.372 | 0.198 |
| enwiki | constant Product-Kalman | +0.348 | 2.074 | 0.332 | 0.174 |
| enwiki | hop Product-Kalman | **-0.506** | **1.079** | **0.149** | **0.038** |
| Pearltrees | prior | +1.766 | 2.863 | 0.525 | 0.280 |
| Pearltrees | independent Kalman | +1.216 | 2.838 | 0.467 | 0.262 |
| Pearltrees | constant Product-Kalman | +1.387 | 2.907 | 0.451 | 0.275 |
| Pearltrees | hop Product-Kalman | **-0.265** | **0.853** | **0.194** | **0.043** |

Primary paired row-bootstrap gains for hop Product-Kalman:

| corpus | versus | NLL gain | 95% interval |
|---|---|---:|---:|
| enwiki | prior | +1.454 | [+1.159, +1.737] |
| enwiki | independent Kalman | +0.837 | [+0.599, +1.090] |
| enwiki | constant Product-Kalman | +0.854 | [+0.628, +1.078] |
| Pearltrees | prior | +2.031 | [+1.660, +2.431] |
| Pearltrees | independent Kalman | +1.481 | [+1.135, +1.860] |
| Pearltrees | constant Product-Kalman | +1.652 | [+1.290, +2.018] |

The split-stability direction is unusually consistent. Hop Product-Kalman beats both the prior and independent
control on all 40 enwiki splits and all 39 valid Pearltrees splits. Its mean gain over the independent control is
`+0.820` for enwiki and `+1.701` for Pearltrees.

Constant covariance is not enough. Constant Product-Kalman trails the independent control on the enwiki primary
split and on average in Pearltrees. The improvement comes from the frozen hop-conditioned covariance form, not
from correlation modeling alone.

## JointPosterior Result

Adding Product-Kalman posterior means does not improve the registered operator-family classifier.

| corpus | variant | accuracy | log-loss | ECE-10 | margin AURC |
|---|---|---:|---:|---:|---:|
| enwiki | registered sources | 0.677 | **0.872** | **0.144** | 0.157 |
| enwiki | plus constant means | 0.677 | 0.874 | 0.144 | 0.156 |
| enwiki | plus hop means | 0.685 | 0.886 | 0.149 | 0.155 |
| Pearltrees | registered sources | **0.545** | 0.915 | **0.119** | **0.308** |
| Pearltrees | plus constant means | **0.545** | **0.915** | 0.119 | 0.308 |
| Pearltrees | plus hop means | 0.518 | 0.953 | 0.148 | 0.328 |

The small enwiki accuracy increase does not survive proper scoring or calibration. Across split seeds, adding hop
means worsens average log-loss in both corpora and has no stable ECE or AURC advantage. In neither primary split
does the fused AURC interval lie below the baseline point estimate.

## Decision

**Do not promote Product-Kalman as an end-to-end replacement for the registered `JointPosterior` baseline.**

The continuous part clears its frozen NLL and calibration gates. The categorical fusion part fails all three
promotion requirements: log-loss, ECE, and selective-risk separation.

This distinction is informative rather than contradictory. Hop-conditioned covariance predicts continuous error
geometry well, including in Pearltrees where the path-local graph observation is constant. The resulting posterior
mean is not automatically a better feature for deciding which operator family applies. Mean fusion and uncertainty
calibration are different jobs.

## Interpretation And Next Test

The evidence supports retaining `Sigma(hop)` as a calibrated likelihood/error-geometry component. It does not
support treating the Product-Kalman posterior mean as additional operator evidence. A later preregistered test may
use the predicted covariance for likelihood scoring or a calibrated abstention policy without feeding it back as a
raw classifier feature.

The Pearltrees result should not be read as validation of its graph measurement: that measurement is constant and
therefore contributes no within-corpus discrimination. It instead shows that the learned relationship between hop
and residual covariance transfers to a more tree-like, principal-parent corpus. Enwiki remains the cleaner graph
measurement test because its LMDB-derived hit probability varies by row and avoids local title-typo exposure.

## Limitations

- Targets come from one non-deterministic LLM judge and are not independent human labels.
- Split seeds reuse the same 250 pairs; stability is not independent replication.
- The smooth hop covariance is a regularizer, not established generative truth.
- Pearltrees path-local graph probabilities are degenerate at one for retained ancestor pairs.
- Title correction sensitivity in Pearltrees remains nontrivial; this evaluation uses the frozen audited view.
- These are exploratory cross-corpus results, not a confirmatory extension of the earlier enwiki `Sigma(hop)` test.

## Reproducibility

Durable artifacts are in `repro/product_kalman_public_holdout/`. The feature tables preserve all fixed source values,
targets, identity components, and split provenance, so the statistical evaluation does not require rerunning GPU
inference. Regenerable e5 caches and private/local graph exports are excluded; their hashes remain in the manifests.
