# Cheap-judge pipeline baseline: graph_S is a free S channel; the scheme wins at low budget

The two zero-scoring-cost validations accompanying `DESIGN_cheap_judge_pipeline.md`. Data: the 1,700
pair-matched dual-judge campaign rows; prior = `model_prod_namecond.pt` agnostic readouts — the
campaign-INDEPENDENT prior (blocker 1). The earlier draft used `model_channel_heads_namecond_r0.pt`, whose
trunk was fine-tuned on the campaign train split, giving an optimistic P0 on the fit rows; all tables below
were regenerated against the campaign-independent checkpoint. The qualitative findings survive; the exact
NLLs moved. Note (against the pre-registered expectation that the graph channels would GROW once the prior
was de-memorized): graph_S's free-only value actually SHRANK modestly (+0.53/+0.84 → +0.41/+0.64), because
the honest prior is itself a better S predictor on held rows. A second correction (blocker 3, luna
bias-correction) then FLIPPED the free-tier-vs-cheap-judge ordering on S — see §1.

## 1. The symmetric-graph S channel (user's point) — the free tier now measures S

`run_sym_channel_fusion.py`: graph_S = a 4-feature linear model (inverse common-ancestor lateral distance,
shared-parent, shared-grandparent, is-ancestor), calibrated to S on the train split — the multivariate
analog of the d→D affine calibration. Added as an S measurement row (H=[0,1]); 6×6 joint blocks per split,
40 descendant-disjoint splits. S-MARGINAL NLL (the channel under test):

Luna is bias-corrected first (global train-split affine, blocker 3 / DESIGN §2) before entering the
covariance fit, so the two luna rungs below reflect the debiased cheap judge.

| rung | expl | fresh |
|---|---|---|
| prior | −0.282 | +0.230 |
| +graph_D | −0.359 | +0.186 |
| **+graph_D+graph_S (free-only)** | **−0.765** | **−0.454** |
| +graph_D+luna | −0.799 | −0.506 |
| ALL | −0.896 | −0.579 |

- **graph_S free-only value: +0.41/+0.64 NLL** (per-split SE 0.006/0.009, 40/40 splits positive; blocker 4)
  — the free tier goes from prior+graph_D to a genuine S measurement; graph_S is a real, zero-cost S channel.
- **CLAIM FLIP (blocker 3):** the earlier draft said the free-only tier BEATS prior+graph+luna on S on
  both corpora. That was an ARTIFACT of leaving luna's tilt folded into R as variance. Once luna is
  bias-corrected as DESIGN §2 specifies, the **cheap judge is the better S measurement**: +graph_D+luna
  now beats free-only on both corpora (expl −0.799 < −0.765; fresh −0.506 < −0.454).
- **graph_S still adds after luna, but only +0.10/+0.07** (per-split SE 0.005/0.005, 40/40 & 38/40 splits
  positive; was +0.25/+0.45 with the uncalibrated luna) — small but positive, so graph_S is not redundant
  with the debiased judge.
- **Stats caveat (blocker 4):** the SEs above are now PAIRED per-split (mean of per-split value differences
  ± across-split SE), not the earlier pooled row-SE — held rows repeat across the 40 splits, so a pooled
  row-SE is not an independent-sample SE. And `descendant_disjoint_split` is descendant-disjoint, not
  node-disjoint, so these ladders are **EXPLORATORY** (a node-disjoint / block-disjoint split would be the
  confirmatory design). The sim §2 numbers are resample SDs over 10 draws and are likewise exploratory.
- Raw feature strength: corr(1/(1+d_sym), S) = +0.60/+0.43; shared-parent +0.62/+0.48.
- Honesty note: part of graph_S's power is reproducing the stratum ordering (the strata are graph-built);
  that is deployable signal for fusion NLL (deployment has the same features), but a within-stratum ladder
  would show a smaller number — same caveat shape as the S-head decomposition.

Consequence for the DESIGN: "the graph doesn't observe S" is retired. The free tier (prior ⊕ graph_D ⊕
graph_S) is the fusion floor every judge call must improve on, and S fusion is non-trivial without any
judge.

## 2. Matched-cost simulation — the scheme wins where it's meant to (low coverage)

`sim_matched_cost.py`: equal 5.5-call budget n; arm A = n pure 5.5 labels; arm B = an n_ov = max(30, 0.3n)
dual-scored overlap (labels + block fit) + `k·n − n_ov·(k+1)` luna-bulk pairs with FUSED targets
(prior⊕graph_D⊕graph_S⊕luna, luna debiased per §1). Downstream estimator: ridge on frozen e5 pair-features,
λ by inner holdout (fixed λ hit the 768-feature interpolation threshold — double descent — an instructive
proxy artifact). Held corr vs 5.5 labels, 10 resamples.

**Evaluation frame (uniform):** every number in this report measures FIDELITY TO THE gpt-5.5-low operating
judge (the current production target), not agreement with independent ground truth. "Wins" mean "recovers
the operating judge's labels better at equal cost".

Budget accounting (blocker 2): the overlap uses a 30-row floor, so at n=80 n_ov=30 > 0.3n=24; the bulk is
sized from n_ov (not 0.3n) so realized spend is exactly n. Cells whose bulk exceeds the scored pool are
flagged **TRUNC** and excluded from matched-cost claims (they understate arm B). Truncated: expl n=160 k=8,
n=320 k≥4, n=640 all; fresh n=160 k=8, n=320 k≥4.

Arm-B luna is bias-corrected (blocker 3) inside the fused targets, matching §1.

| budget n | best NON-truncated arm (expl D/S) | best NON-truncated arm (fresh D/S) |
|---|---|---|
| 80 | **B k=8** (+0.599/+0.695 vs A +0.552/+0.669); **k=2 LOSES S** (+0.639) | **B k=8** (+0.442/+0.402 vs A +0.382/+0.369); **k=2 LOSES S** (+0.360) |
| 160 | **B k=2/k=4 ≥ A** both channels (+0.580/+0.681 vs A +0.555/+0.651) | mixed: A on D (+0.461 vs +0.400–0.441), B on S (+0.416–0.431 vs +0.399) |
| 320 | B k=2 ~parity D (+0.584 vs +0.586); A S +0.702 vs B +0.696 | A ahead both (+0.466/+0.434 vs +0.434/+0.419) |
| 640 | all B TRUNC — no matched-cost claim | — (n > pool, skipped) |

- **Low-budget regime: the scheme wins at n=80 on both corpora; at n=160 it wins expl, mixed fresh (A on
  D, B on S).** NOT "wins at every k" — the earlier headline was false: at fresh n=160 the best B on D
  (+0.441) still trails A (+0.461), and at n=80 the k=2 arm now loses S on both corpora (the win is carried
  by the higher-k arms, biggest at k=8). This is the regime the scheme targets (a new corpus with a limited
  budget: Pearltrees, mindmap).
- High-budget convergence is partly REAL (with enough 5.5 labels, pure labels win on D — fused targets
  carry luna noise) and partly ARTIFACT: the bulk purchase truncates at the scored pool (at n=320, k=4
  already wants more luna rows than exist), so large-n·k cells understate arm B.
- S favors the scheme at low budget (n≤160 on both corpora); at n=320 the pure-5.5 arm reaches parity/ahead
  on S. graph_S inside the fused targets does the work at low budget (§1).
- Proxy caveat: ridge-on-e5 is not the transformer head; both arms share it, so the ORDERING is the
  claim, not the magnitudes. One full-head confirmation at a single grid point is the upgrade if a real
  budget decision hangs on this.

**Reading for the budget decision:** at new-corpus scale (coverage low relative to corpus), k≥4 wins
outright while k=2 is marginal (ties D, can lose S); the scheme's advantage shrinks as coverage saturates.
The practical recipe stands: luna everywhere + small random 5.5 overlap + graph channels in the fusion +
routed 5.5 on conflict + sonnet-5 tiebreaks on disagreement.

## 3. Positioning vs the JointPosterior dense baseline (control) — deferred comparison

The correlated-Gaussian conditioner used here (`correlated_update_H` / `gaussian_condition_update`) is the
**interpretable dense baseline** in the project's uncertainty playbook
(`DESIGN_uncertainty_estimation_playbook.md`, `THEORY_evidence_fusion.md`): it fits the full cross-channel
error covariance on held/overlap rows and prices every correlation explicitly (`K = (PHᵀ + C)S⁻¹`), which
is exactly the playbook's remedy for "sources are not independent → naive PoE over-confidences." Its role
here is a **control**: it is the honest, few-parameter (~20 numbers) fusion whose gains any learned combiner
must beat to justify its opacity. The playbook's champion learned combiner, `JointPosterior` (mu_posterior.py),
is a calibrated softmax over RELATION classes; it down-weights redundant e5-shared signal automatically
rather than through a hand-fit covariance.

**Why the head-to-head is deferred (explicit TODO).** JointPosterior outputs a relation-class distribution;
the Kalman conditioner outputs a continuous (D, S) Gaussian posterior. A fair same-held-split comparison
needs a metric bridge (map relation-class probabilities to D/S, or evaluate both by a common
decision/AURC/NLL surrogate) — that wiring is > 1h and is deferred. TODO(blocker 6): implement
`JointPosterior.fit` on the campaign overlap features and compare AURC + decision-flip accuracy against the
correlated-Gaussian posterior on the same descendant-disjoint held split, using the playbook's margin-gate
evaluation. Until then, treat the correlated-Gaussian numbers above as the dense-baseline control, not as a
claim that the explicit filter beats a learned joint head.

## Repro

```
python3 make_pipeline_diagrams.py          # figures/pipeline_dataflow.png, figures/fusion_architecture.png
python3 run_sym_channel_fusion.py          # default --ckpt model_prod_namecond.pt (campaign-independent)
python3 sim_matched_cost.py --k 2 4 8
python3 test_cheap_judge_blockers.py       # blocker 2 + 7 unit tests (torch-free)
```
