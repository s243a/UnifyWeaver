# Cheap-judge JointPosterior comparison — node-disjoint combiner/calibration result

Run 2026-07-10 from the post-#3648 validation branch. Protocol and non-claims:
`DESIGN_cheap_judge_joint_posterior.md`.

## Bottom line

The learned `JointPosterior` does **not** establish a win over the dense correlated-Gaussian conditioner plus
its D/S LR bridge on this declared three-way task. Seed 0 is near-tied; fresh seed 1 significantly favours the
Gaussian+bridge pipeline. The #3648 dense method therefore remains a strong interpretable baseline rather than
a control that has already been superseded.

This is fidelity to GPT-5.5's macro decision (directional / symmetric / other), not independent ground truth.
The campaign's `cur_rel` cannot be used: it is `subcategory` on all 1,770 rows.
The reported ECE/log-loss audit raw cross-entropy outputs; no held labels were used for post-hoc calibration.

## Primary split and headline comparison

Seed 0 partitions nodes before retaining within-side pairs. No held endpoint appears in this follow-up's
combiner/calibration training. The campaign-independent upstream checkpoint is not audited as node-independent
from every earlier training source, so this is not an end-to-end unseen-node claim.

| corpus | train / held / crossing rows dropped | method | accuracy | log-loss | ECE-10 | margin AURC [component-bootstrap 95% CI] |
|---|---:|---|---:|---:|---:|---:|
| exploratory | 329 / 173 / 498 | D/S hard-max linear reference | 0.931 | 0.161 | 0.020 | 0.006 [0.002, 0.013] |
| | | dense Gaussian + D/S LR bridge, all | **0.879** | **0.331** | 0.071 | **0.025 [0.012, 0.042]** |
| | | JointPosterior LR, all sources | 0.873 | 0.333 | **0.043** | 0.026 [0.013, 0.045] |
| | | factored PoE, equal | 0.844 | 0.508 | 0.100 | 0.032 [0.017, 0.050] |
| | | factored PoE, separability | 0.809 | 0.543 | 0.118 | 0.065 [0.038, 0.097] |
| fresh | 263 / 108 / 329 | D/S hard-max linear reference | **0.935** | **0.217** | **0.028** | **0.012 [0.003, 0.025]** |
| | | dense Gaussian + D/S LR bridge, all | **0.787** | **0.573** | **0.062** | **0.084 [0.046, 0.137]** |
| | | JointPosterior LR, all sources | 0.778 | 0.591 | 0.092 | 0.092 [0.049, 0.146] |
| | | factored PoE, equal | 0.722 | 0.753 | 0.166 | 0.084 [0.046, 0.128] |
| | | factored PoE, separability | 0.796 | 0.853 | 0.341 | 0.077 [0.044, 0.119] |

Paired `AURC(Joint/all) - AURC(Gaussian/all)` (negative favours JointPosterior):

- exploratory: **+0.0008 [-0.0053, +0.0065]**;
- fresh: **+0.0074 [-0.0123, +0.0280]**.

Partition sensitivity matters. On seed 1, exploratory Gaussian+bridge versus Joint is
`.897/.326/.070/.033` versus `.903/.272/.019/.021` (accuracy/log-loss/ECE/AURC), paired delta
`-0.0127 [-0.0403,+0.0030]`; fresh is `.728/.618/.131/.111` versus `.737/.670/.088/.150`, paired delta
**`+0.0384 [+0.0056,+0.0781]`**, favouring Gaussian+bridge. Seed 0 remains the declared primary split, but
its near-tie is not a partition-robust equivalence result.

The endpoint-component bootstrap had 126 blocks (largest 8 rows) for exploratory and 73 blocks (largest 4)
for fresh. These blocks are sufficiently dispersed for the interval to be more informative than a row
bootstrap, though it remains a one-split operating-judge result.

The D/S linear-reference gap is diagnostic, not a ceiling. It can reflect max-pooling information loss, but
also finite training data, LR misspecification/optimisation, and incoherence between the judge's relation
probabilities and relation-specific μ values. A flexible cross-fitted decoder would be needed to estimate a
representation ceiling.

## Source rungs and dependence

| corpus | method family | prior only | +graph | +Luna | all |
|---|---|---:|---:|---:|---:|
| exploratory | dense Gaussian + bridge AURC | 0.181 | 0.072 | 0.032 | **0.025** |
| | JointPosterior AURC | 0.181 | 0.066 | 0.044 | **0.026** |
| fresh | dense Gaussian + bridge AURC | 0.348 | 0.149 | 0.101 | **0.084** |
| | JointPosterior AURC | 0.299 | 0.166 | 0.115 | **0.092** |

The seed-0 all-source point estimate is best for each family, but its paired ablation is conditional on that
one fitted partition. Deltas are `AURC(all) - AURC(without source)`; negative favours adding the source:

| corpus | family | add graph [95% CI] | add debiased Luna [95% CI] |
|---|---|---:|---:|
| exploratory | dense Gaussian + bridge | -0.0073 [-0.0183, +0.0003] | **-0.0469 [-0.0791, -0.0183]** |
| | JointPosterior | **-0.0177 [-0.0423, -0.0025]** | **-0.0402 [-0.0768, -0.0099]** |
| fresh | dense Gaussian + bridge | -0.0163 [-0.0471, +0.0076] | **-0.0644 [-0.1337, -0.0133]** |
| | JointPosterior | -0.0237 [-0.0815, +0.0325] | **-0.0746 [-0.1380, -0.0307]** |

Conditional on the seed-0 fitted split, debiased Luna's interval favours inclusion in all four cells. This is
not robustness across node partitions: a seed-1 sensitivity makes fresh Joint add-Luna inconclusive and fresh
Joint add-graph harmful (`+0.0441 [+0.0104,+0.0806]`); fresh Joint add-Luna becomes
`-0.0739 [-0.1573,+0.0007]`. Graph's seed-0 point estimate is beneficial in all four, but its interval excludes
zero only for exploratory JointPosterior. The result is not evidence of source independence. On exploratory
training
rows, `corr(graph_S,luna_S)=+0.68`,
`corr(graph_D,luna_D)=+0.60`, and `corr(prior_D,graph_D)=+0.58`. On fresh, the corresponding graph-D/Luna-D
correlation is +0.56. The learned joint head and fitted cross-covariance can price this redundancy; a factored
control cannot.

The separability-weighted factored control illustrates why one metric is insufficient: on fresh it has high
accuracy and low AURC, but its log-loss 0.853 and ECE 0.341 reveal severe miscalibration. It is not a preferred
combiner.

The hard macro target resolves exact ties in fixed class order (directional, symmetric, other). Across the
1,770-row campaign there are 7 exact top ties and 41 normalized top-two margins below 0.02. The result JSON now
persists each comparison row's macro vector, margin, tie flag, and train/held/cross-dropped membership.

## Within-judge pooling diagnostic

Hard max, probability-weighted pooling, and temperature-softmax pooling (`T=0.10`) are reductions of relation
scores inside **one** GPT-5.5 response. They are not ways of choosing among prior/graph/Luna experts.

| corpus | linear-reference pooling | full-corpus mean D/S shift | full-corpus D/S corr | accuracy | log-loss | AURC |
|---|---|---:|---:|---:|---:|---:|
| exploratory | hard max | 0 / 0 | 1 / 1 | **0.931** | **0.161** | **0.006** |
| | probability weighted | -0.052 / -0.028 | 0.983 / 0.998 | 0.919 | **0.161** | **0.006** |
| | softmax `T=0.10` | -0.018 / -0.014 | 0.998 / 0.998 | 0.925 | 0.164 | 0.007 |
| fresh | hard max | 0 / 0 | 1 / 1 | **0.935** | **0.217** | 0.012 |
| | probability weighted | -0.073 / -0.052 | 0.982 / 0.997 | 0.889 | 0.251 | 0.016 |
| | softmax `T=0.10` | -0.030 / -0.019 | 0.998 / 0.999 | **0.935** | 0.223 | **0.011** |

In this same-response macro-decision diagnostic at `T=0.10`, no alternative has a clear point improvement over
hard max. The shifts/correlations use the full matched corpus, probability-weighted pooling reuses probabilities
that define the macro target, and no paired pooling-difference intervals were computed. This does not establish
hard max as the best continuous target. A production comparison must regenerate the prior and every judge
measurement under one pooling rule; mixing reductions changes the estimand.

## Reproduction

```bash
cd prototypes/mu_cosine
python3 test_cheap_judge_joint_posterior.py
python3 run_cheap_judge_joint_posterior.py \
  --ckpt model_prod_namecond.pt \
  --seed 0 --held-frac 0.40 --epochs 400 --boot 500 \
  --pool-temperature 0.10 \
  --out /tmp/cheap_judge_joint_posterior_seed0.json
```

Inputs are the existing local campaign artifacts:
`/tmp/mu_data/campaign_scored.tsv`, `/tmp/mu_data/campaign_scored_luna.tsv`, and the graph/e5 artifacts used by
`load_campaign_datasets`. No new judge calls are made.

Eight focused tests pass. They cover macro aggregation/ties, combiner endpoint disjointness, endpoint-component
construction, component bootstrap, Gaussian bridge quadrature, and alternative within-judge pooling.
