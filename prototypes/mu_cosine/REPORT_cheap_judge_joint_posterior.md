# Cheap-judge JointPosterior comparison — node-disjoint combiner/calibration result

Run 2026-07-10 from the post-#3648 validation branch. Protocol and non-claims:
`DESIGN_cheap_judge_joint_posterior.md`.

## Bottom line

The learned `JointPosterior` does **not** establish a win over the dense correlated-Gaussian conditioner plus
its D/S LR bridge on this declared three-way task. After both experiments use the same audited, stratum-aware
node splitter, seeds 0 and 1 are near-tied on both corpora and every paired AURC interval includes zero. The
#3648 dense method therefore remains a strong interpretable baseline rather than a control that has already
been superseded.

This is fidelity to GPT-5.5's macro decision (directional / symmetric / other), not independent ground truth.
The raw campaign's `cur_rel` cannot be used: it is `subcategory` on all 1,770 rows; 1,700 rows remain after
matching the exploratory and fresh campaign views to Luna outputs.
The reported ECE/log-loss audit raw cross-entropy outputs; no held labels were used for post-hoc calibration.

## Primary split and headline comparison

Seed 0 selects the best of 64 outcome-blind node assignments using retained coverage and macro-class strata,
then retains within-side pairs. No held endpoint appears in this follow-up's combiner/calibration training.
This endpoint split is not edge-disjoint or k-hop-isolated from the ambient parent graph used to construct graph
features. The campaign-independent upstream checkpoint is not audited as node-independent from every earlier
training source, so this is not an end-to-end unseen-node claim.

| corpus | train / held / crossing rows dropped | method | accuracy | log-loss | ECE-10 | margin AURC [component-bootstrap 95% CI] |
|---|---:|---|---:|---:|---:|---:|
| exploratory | 365 / 161 / 474 | D/S hard-max linear reference | **0.925** | **0.170** | **0.018** | **0.008 [0.003, 0.016]** |
| | | dense Gaussian + D/S LR bridge, all | 0.851 | 0.378 | **0.037** | **0.033 [0.018, 0.054]** |
| | | JointPosterior LR, all sources | **0.870** | **0.377** | 0.071 | 0.035 [0.020, 0.059] |
| | | factored PoE, equal | 0.795 | 0.596 | 0.143 | 0.043 [0.024, 0.070] |
| | | factored PoE, separability | 0.770 | 0.584 | 0.109 | 0.077 [0.043, 0.118] |
| fresh | 262 / 117 / 321 | D/S hard-max linear reference | **0.923** | **0.212** | **0.058** | **0.009 [0.002, 0.021]** |
| | | dense Gaussian + D/S LR bridge, all | **0.718** | **0.598** | 0.121 | **0.108 [0.069, 0.165]** |
| | | JointPosterior LR, all sources | 0.701 | 0.660 | **0.113** | 0.113 [0.072, 0.175] |
| | | factored PoE, equal | 0.726 | 0.808 | 0.135 | 0.118 [0.073, 0.175] |
| | | factored PoE, separability | 0.650 | 0.817 | 0.156 | 0.146 [0.093, 0.217] |

Paired `AURC(Joint/all) - AURC(Gaussian/all)` (negative favours JointPosterior):

- exploratory: **+0.0013 [-0.0061, +0.0081]** (122 blocks; largest 5 rows);
- fresh: **+0.0058 [-0.0075, +0.0193]** (72 blocks; largest 7 rows).

Partition sensitivity matters. On seed 1, exploratory Gaussian+bridge versus Joint is
`.854/.339/.050/.033` versus `.884/.322/.046/.025` (accuracy/log-loss/ECE/AURC), paired delta
`-0.0085 [-0.0288,+0.0033]` (122 blocks; largest 4); fresh is `.750/.593/.071/.102` versus
`.759/.618/.080/.104`, paired delta `+0.0022 [-0.0228,+0.0293]` (69 blocks; largest 6). These audited
splits do not support a method-level AURC difference, but two seeds are not an equivalence study.

The endpoint-component bootstrap respects shared endpoints, but AURC can over-emphasize the earliest
high-confidence errors ([AUGRC, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/047c84ec50bd8ea29349b996fc64af4b-Paper-Conference.pdf))
and its empirical estimator is biased at small held-set sizes
([population AURC](https://arxiv.org/html/2410.15361v4)). An AUGRC robustness check and more node partitions
remain follow-up work; the present intervals are one-split operating-judge results.

The D/S linear-reference gap is diagnostic, not a ceiling. It can reflect max-pooling information loss, but
also finite training data, LR misspecification/optimisation, and incoherence between the judge's relation
probabilities and relation-specific μ values. A flexible cross-fitted decoder would be needed to estimate a
representation ceiling.

## Source rungs and dependence

| corpus | method family | prior only | +graph | +Luna | all |
|---|---|---:|---:|---:|---:|
| exploratory | dense Gaussian + bridge AURC | 0.202 | 0.071 | 0.040 | **0.033** |
| | JointPosterior AURC | 0.208 | 0.058 | 0.042 | **0.035** |
| fresh | dense Gaussian + bridge AURC | 0.388 | 0.197 | 0.143 | **0.108** |
| | JointPosterior AURC | 0.347 | 0.232 | 0.120 | **0.113** |

The seed-0 all-source point estimate is best for each family, but its paired ablation is conditional on that
one fitted partition. Deltas are `AURC(all) - AURC(without source)`; negative favours adding the source:

| corpus | family | add graph [95% CI] | add debiased Luna [95% CI] |
|---|---|---:|---:|
| exploratory | dense Gaussian + bridge | -0.0068 [-0.0166, +0.0010] | **-0.0373 [-0.0659, -0.0114]** |
| | JointPosterior | -0.0074 [-0.0215, +0.0040] | **-0.0231 [-0.0452, -0.0059]** |
| fresh | dense Gaussian + bridge | **-0.0357 [-0.0778, -0.0055]** | **-0.0888 [-0.1592, -0.0324]** |
| | JointPosterior | -0.0067 [-0.0336, +0.0215] | **-0.1188 [-0.1864, -0.0514]** |

Debiased Luna's interval favours inclusion in all four seed-0 cells and all four audited seed-1 cells. Graph's
seed-0 point estimate is beneficial in all four, but its interval excludes zero only for fresh Gaussian; the
same pattern holds at seed 1. This two-seed stability remains descriptive and is not evidence of source
independence. On exploratory training rows, `corr(graph_S,luna_S)=+0.60`,
`corr(graph_D,luna_D)=+0.63`, and `corr(prior_D,graph_D)=+0.40`. On fresh, the corresponding graph-D/Luna-D
correlation is +0.65. The learned joint head and fitted cross-covariance can price this redundancy; a factored
control cannot.

The factored controls illustrate why one metric is insufficient: fresh equal-weight PoE has slightly higher
accuracy than both joint methods, but worse log-loss and ECE than either. It is not a preferred combiner.

The hard macro target resolves exact ties in fixed class order (directional, symmetric, other). Across the
1,700-row matched campaign there are 7 exact top ties and 41 normalized top-two margins below 0.02. The result
JSON persists each comparison row's macro vector, margin, tie flag, and train/held/cross-dropped membership.

## Within-judge pooling diagnostic

Hard max, probability-weighted pooling, and temperature-softmax pooling (`T=0.10`) are reductions of relation
scores inside **one** GPT-5.5 response. They are not ways of choosing among prior/graph/Luna experts.

| corpus | linear-reference pooling | full-corpus mean D/S shift | full-corpus D/S corr | accuracy | log-loss | AURC |
|---|---|---:|---:|---:|---:|---:|
| exploratory | hard max | 0 / 0 | 1 / 1 | **0.925** | **0.170** | **0.008** |
| | probability weighted | -0.052 / -0.028 | 0.983 / 0.998 | 0.919 | 0.186 | 0.010 |
| | softmax `T=0.10` | -0.018 / -0.014 | 0.998 / 0.998 | 0.919 | 0.177 | 0.009 |
| fresh | hard max | 0 / 0 | 1 / 1 | **0.923** | **0.212** | **0.009** |
| | probability weighted | -0.073 / -0.052 | 0.982 / 0.997 | 0.906 | 0.255 | 0.016 |
| | softmax `T=0.10` | -0.030 / -0.019 | 0.998 / 0.999 | **0.923** | 0.214 | 0.010 |

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
  --seed 0 --held-frac 0.40 --split-candidates 64 --epochs 400 --boot 500 \
  --pool-temperature 0.10 \
  --out /tmp/cheap_judge_joint_posterior_seed0.json
```

Inputs are the existing local campaign artifacts:
`/tmp/mu_data/campaign_scored.tsv`, `/tmp/mu_data/campaign_scored_luna.tsv`, and the graph/e5 artifacts used by
`load_campaign_datasets`. No new judge calls are made.

Eight focused tests pass. They cover macro aggregation/ties, combiner endpoint disjointness, endpoint-component
construction, component bootstrap, Gaussian bridge quadrature, and alternative within-judge pooling.
