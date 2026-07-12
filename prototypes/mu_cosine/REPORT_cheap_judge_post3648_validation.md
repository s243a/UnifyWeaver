# Cheap-judge pipeline after PR #3648: corrected validation and handoff

Run 2026-07-10 on `codex/post-3648-validation`. This report supersedes the merged PR's deployment headline;
it does not rewrite the historical PR record. Each section names its target and uncertainty unit because the
matched-cost proxy, Gaussian NLL, and macro-decision AURC answer different questions.

## Bottom line

1. **Do not retain “k≥4 wins at new-corpus scale.”** After nesting pseudo-target generation outside the paid
   validation fold, no exact-budget cheap arm has a positive training-subsample interval excluding zero. At
   `n=160,k=2`, all 10 S subsample deltas are negative on both corpora, but this fixed-partition precision
   diagnostic is not an inferential harm claim.
2. **Free graph_S is real, but mostly a pre-Luna channel.** Its free-only S-NLL gain is positive across all 40
   node partitions and the fixed held-node interval excludes zero on both corpora. Its increment after debiased
   Luna is small and both fixed intervals include zero. This tests graph_S as a final-posterior input, not the
   graph judge's separate role in familiarizing or adapting a model to the dataset.
3. **Debiased Luna is still valuable, but task- and partition-dependent.** It dominates the analytic D/S
   Gaussian ladder. Macro-decision AURC favours it on seed 0, but one fresh seed-1 Joint interval includes zero.
   The leakage-free matched-cost ridge proxy does not show a general spending advantage for cheap labels.
4. **JointPosterior does not establish a win over Gaussian+bridge.** After both runners use the same audited,
   stratum-aware node splitter, paired AURC intervals include zero on both corpora at seeds 0 and 1.
5. **The joint square-root/QR conditioner is numerically equivalent and GPU-capable, but not the fixed-design
   throughput winner.** A matched design-cached dense correlated gain beat QR on CPU and CUDA in every static cell.
   Reserve root-threading QR for changing/sequential blocks, pending their own matched benchmark.

## 1. Matched-cost proxy: strict node-disjoint primary result

Protocol fixes:

- campaign-independent prior;
- strict node partition (`held-node-frac=0.40`), all crossing pairs discarded;
- exact realised-spend assertions and integral price ratios;
- one stable corpus/replicate permutation, invariant to `--n` order;
- identity-keyed paid-label inner split; calibration/covariance/pseudo-target/ridge/scaler fit on paid
  inner-train, lambda scored only on paid inner-valid true labels, then full-pipeline refit on all paid rows;
- 10 paired training-subsample replicates;
- an `A+free` control using the same `n` paid 5.5 labels and all remaining zero-cost prior+graph targets.

Coverage at split seed 0:

| corpus | train pairs | held pairs | crossing pairs dropped |
|---|---:|---:|---:|
| exploratory | 377 | 168 | 455 |
| fresh | 242 | 99 | 359 |

Only untruncated, exact-budget cells support a matched-cost claim. Deltas below are candidate minus `A: 5.5
only`; intervals are percentile bootstraps over the 10 paired training-subsample replicates and condition on
this one fixed held-node split. The column therefore reports **subsample precision, not an inferential CI**;
it is not a population interval over possible node partitions.

| corpus | budget / ratio | D delta [95% subsample precision] | S delta [95% subsample precision] | interpretation |
|---|---|---:|---:|---|
| exploratory | n=80, k=2 | -0.007 [-0.046,+0.021] | -0.011 [-0.052,+0.042] | no nominal improvement |
| exploratory | n=80, k=4 | -0.037 [-0.102,+0.027] | -0.049 [-0.114,+0.030] | no nominal improvement |
| fresh | n=80, k=2 | -0.003 [-0.028,+0.020] | -0.012 [-0.035,+0.011] | no nominal improvement |
| fresh | n=80, k=4 | +0.023 [-0.026,+0.065] | -0.001 [-0.043,+0.043] | no nominal improvement |
| exploratory | n=160, k=2 | -0.023 [-0.076,+0.040] | -0.050 [-0.099,-0.010] | negative in these subsamples; no population harm claim |
| fresh | n=160, k=2 | -0.029 [-0.073,+0.006] | -0.062 [-0.103,-0.021] | negative in these subsamples; no population harm claim |

The free control is also a tradeoff, not a universal winner. At `n=80`, `A+free−A` is exploratory D
-0.039 [-0.082,-0.005], S -0.012 [-0.091,+0.067], and fresh D +0.046 [-0.007,+0.088], S
-0.071 [-0.119,-0.025]. At fresh `n=160` its D/S deltas are +0.019 [+0.004,+0.034] and
-0.057 [-0.108,-0.014]. Direct `k=4−A+free` at `n=80` is exploratory D +0.002 [-0.062,+0.070],
S -0.037 [-0.084,+0.005], and fresh D -0.023 [-0.078,+0.035], S +0.070 [-0.004,+0.145]. No paid
allocation Pareto-dominates the other across endpoints and corpora.

The intervals above are **nominal training-subsample intervals**, not confirmation: they hold the node
partition and held outcomes fixed and do not adjust for endpoint multiplicity or multiple cells/endpoints.
The descendant-disjoint mode remains available to reproduce the historical split but has not been rerun under
the final nested protocol. Next work should repeat the exact-budget cells across preregistered node partitions
and use held-node resampling for correlation deltas.

Exact reproduction:

```bash
python3 -u prototypes/mu_cosine/sim_matched_cost.py \
  --ckpt prototypes/mu_cosine/model_prod_namecond.pt \
  --split node-disjoint --split-seed 0 --held-node-frac 0.40 --split-candidates 64 \
  --n 80 160 --k 2 4 --reps 10 --seed 0 --bootstrap-reps 5000 --confidence 0.95
```

SHA-256: prior checkpoint `c1cfc3a3827e42a1993f4286b6a881aee7ff10eb56a76367735b9ec8fdf11f7d`;
GPT-5.5 campaign `c2acf399aadd35c3797171d5b42d64e45b07055802092cc28becf308d460ef09`;
Luna campaign `a8d951b4fd05f0ca111fbe9d9c23881bb47b790ccb335731113bfd4ae77ffe6e`;
exploratory e5 `037396a1d6552892d3d6aa04b3cc12bc6d5e6a532d3697e28da1f28e04810700`;
fresh e5 `ab51eb5d07cdd3bed34112a1d92005fcd3f8c46245cf068a7ab0a225afc6cd6b`.

## 2. graph_S: 40 node partitions plus held-node uncertainty

Mean value is baseline S-NLL minus augmented S-NLL. Split SD is descriptive partition stability. The fixed
seed-0 interval uses paired two-endpoint/pigeonhole node resampling on the held partition.

| corpus / graph_S role | mean ± split SD | positive split seeds | fixed estimate [95% node CI] |
|---|---:|---:|---:|
| exploratory, free-only | +0.4034 ± 0.0605 | 40/40 | **+0.3586 [+0.1537,+0.5519]** |
| fresh, free-only | +0.6477 ± 0.0807 | 40/40 | **+0.6565 [+0.2723,+1.0383]** |
| exploratory, after Luna | +0.0968 ± 0.0353 | 40/40 | +0.0874 [-0.0608,+0.2668] |
| fresh, after Luna | +0.0765 ± 0.0417 | 39/40 | +0.0219 [-0.1759,+0.2067] |

This confirms graph_S as useful free supervision. It does not confirm a material increment once debiased Luna
is present. That conclusion is deliberately narrow: the graph judge can also provide broad, nearly free
structural supervision that teaches a model a dataset's entities, vocabulary, topology, and relation patterns.
This evaluation holds the upstream representation fixed and therefore does not measure that dataset-
familiarization benefit. Removing graph supervision based on the post-Luna fusion increment would be an invalid
inference; a matched adaptation ablation is separate future work.

## 3. Corrected Luna campaign

`fine_tune_fused_head_luna.py` now separates the trainable channel-head initialisation/anchor from the
campaign-independent Gaussian prior and performs global train-only affine Luna calibration before covariance
fitting. It also has an analytic-only mode and an explicit historical reproduction mode.

Corrected held NLL (`prior | +graph | +Luna`) is exploratory `-0.285 | -0.568 | -1.493` and fresh
`+0.405 | -0.018 | -1.101`; Luna's incremental value is +0.925/+1.083. In the deterministic 800-step head
run, within-stratum D correlation improves from Luna-head 0.373/0.398 to fused 0.402/0.467. S does not improve:
0.369/0.296 to 0.361/0.261. Full historical comparison, provenance, and reproduction commands are in
`REPORT_luna_campaign.md`.

## 4. Same-split JointPosterior comparison

The campaign's `cur_rel` is constant `subcategory`, so an eight-way relation comparison would be false. The
declared bridge instead predicts the operating judge's three-way macro decision (directional/symmetric/other)
from the identical source vector `[prior_D,prior_S,graph_D,graph_S,luna_D,luna_S]` on one audited,
stratum-aware node-disjoint combiner/calibration split. Endpoint labels are isolated, but graph features still
use the ambient parent graph; this is not edge-disjoint or k-hop-isolated. The upstream checkpoint is
campaign-independent but not audited as endpoint-independent from all earlier training.

| seed / corpus | Gaussian + D/S LR bridge acc / log-loss / ECE / AURC | JointPosterior acc / log-loss / ECE / AURC | paired AURC Joint−Gaussian [95% CI] | endpoint blocks / largest |
|---|---:|---:|---:|---:|
| 0 / exploratory | .851 / .378 / .037 / .033 | .870 / .377 / .071 / .035 | +.0013 [-.0061,+.0081] | 122 / 5 |
| 0 / fresh | .718 / .598 / .121 / .108 | .701 / .660 / .113 / .113 | +.0058 [-.0075,+.0193] | 72 / 7 |
| 1 / exploratory | .854 / .339 / .050 / .033 | .884 / .322 / .046 / .025 | -.0085 [-.0288,+.0033] | 122 / 4 |
| 1 / fresh | .750 / .593 / .071 / .102 | .759 / .618 / .080 / .104 | +.0022 [-.0228,+.0293] | 69 / 6 |

Add-Luna intervals are below zero in all four family/corpus cells at both audited seeds. AURC remains a
small-sample, early-error-sensitive metric; AUGRC is an explicit robustness follow-up, not a post-hoc replacement
for this table. Hard max, probability-weighted pooling, and temperature-softmax pooling were audited **within
one judge** as same-response point diagnostics at `T=.10`, without paired selection intervals; this does not
prove hard max is the best continuous target. These are not expert-selection rules. See
`REPORT_cheap_judge_joint_posterior.md`.

## 5. Joint square-root/QR conditioner

The NumPy reference maintains `P^-1 = U^T U`, decorrelates nonzero prior/measurement cross-covariance through
the Schur complement, Cholesky-whitens each conditionally independent block, and Householder-triangularises the
information pre-array. It matches the dense conditioner for random nonzero-C problems, is row-permutation
invariant after sign normalisation, and gives identical batch/streamed results for conditionally independent
blocks.

The PyTorch backend adds generic batched designs, a fixed-design compact `geqrf`/`ormqr` path that never forms
Q, and the matched design-cached dense-gain baseline. Twenty-nine combined tests pass, including CPU/CUDA
float32/float64 parity, distinct-design CUDA batches, sequential CUDA root threading, and protection against
caller-side mutation of compiled coefficient inputs. On the local GTX 1660
SUPER, static `n=2,m=4` is fastest on CPU dense: 0.058 ms for one row and 0.113 ms for 4096 rows, versus CPU QR
0.220/0.596 ms. Large batches do benefit from CUDA dense: 0.129 ms at `n=32,m=32,batch=4096` and 0.131 ms at
`n=128,m=32`, but CUDA QR remains slower at 0.543/1.121 ms. These are single-run, on-device compute point
measurements; they exclude transfer and do not benchmark the sequential regime where QR carries the updated root.
See `DESIGN_joint_square_root_qr_conditioner.md` and
`REPORT_joint_square_root_qr_benchmark.md`.

## Ordered next work

1. Repeat strict matched-cost `n=80,k∈{2,4}` and `n=160,k=2` across preregistered node partitions;
   calculate paired held-node intervals and declare a multi-endpoint decision rule before looking at results.
2. Add a human-verified or independent-judge target. All current decision/head results measure GPT-5.5
   operating-judge fidelity.
3. Test per-stratum/per-D-bin Luna calibration only inside nested training data; keep global affine as baseline.
4. If revisiting JointPosterior, use aggregated soft macro probabilities and a nested calibration split. Do not
   add capacity merely because the linear head tied.
5. Integrate the Torch QR conditioner only behind a parity/benchmark gate. Keep the dense Cholesky conditioner
   for n=2 single-row work; compare compiled QR against a compiled dense gain for large same-design batches,
   and prefer root-threading QR when sequential blocks change the posterior.
