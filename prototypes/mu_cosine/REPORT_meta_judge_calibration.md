# Meta-judge: two-timescale CE-calibration of the Kalman μ (results)

Implements DESIGN_meta_judge_calibration.md: a candidate-ranking cross-entropy judge calibrates the
μ that feeds the Kalman fusion, with sonnet-5 (subagent-scored, held OUT of the fusion) as the
independent reference. Inference ranks folders by the calibrated Kalman μ (no judge in the loop).
Three mechanism results (each tied to a design decision the user raised) all land in the predicted
direction — but the filing metric does not move: e5 stays the best ranker and, bluntly, the
fine-tune HURTS the conditioned μ ranker relative to the untrained base (§4). The meta-judge is a
validated mechanism without a payoff at this data scale.

## 1. The two-timescale emerges from information content — but only under a precision-gated update

User's claim: the judge should calibrate SLOWLY and the Kalman μ update FAST, and this "falls out
automatically" because each batch carries less information for the (quantization-noisy) ranking CE
than for the MSE distillation.

Tested, not assumed. The naive realization (run both every batch, plain SGD/Adam on the judge) does
NOT self-separate: step size follows the learning rate, not the information, so the judge drifted
~66× TOO FAST (Adam actively normalizes the SNR away to a unit step). The claim holds only under a
**precision/SNR-gated update**: the judge step is scaled by `gain = ‖m‖²/power` of its gradient EMA
(≈1 coherent signal, ≈0 pure noise). Measured over 800 steps:

    emergent judge gain ‖m‖²/power ≈ 0.10   →  ~90% of the CE gradient is noise that cancels

So the candidate-ranking CE IS quantization-noise-dominated, exactly as claimed, and the gate turns
that low information into a small, self-limited step — the two-timescale falls out of the
information content, realized through the program's own Kalman/precision machinery rather than a
hand-set `lr_slow`. (Naive SGD would have needed the timescale tuned by hand.) The general principle
— why Adam and SGD both break emergent-timescale learning and a precision-weighted update is
required — is derived in THEORY_emergent_timescale_learning.md.

## 2. Scale-free negatives carry more information than uniform

User's point: the ranking signal has more information when the folder choices are semantically
CLOSE. Uniform negatives are mostly trivial (far) and dilute the gradient. Fix: scale-free sampling
over the e5-distance rank, `P(rank r) ∝ (r+1)^-1` (Zipf) — mostly close/confusable candidates plus a
heavy tail of a few distant ones (information in the close choices; far ones anchor the easy
regime). Measured (800 steps, both with the sonnet anchor):

| negatives | emergent gain | final ranking CE ↓ | NEAR μ-beats-e5 rate |
|---|---|---|---|
| scale-free (α=1) | **0.105** | **1.857** | **0.360** |
| uniform (α=0) | 0.099 | 2.182 | 0.345 |

Scale-free raises the CE information (gain up, CE down) and does slightly better on the confusable
stratum — the prediction holds in sign on every axis, though the magnitudes are modest.

## 3. Held-out sonnet helps more than the negative distribution

Sonnet-5 scored the 300-row overlap via subagents (D-correlated 0.803 with 5.5, its own −0.10/−0.22
tilt — independent signal, not a copy); it is the calibration anchor NOT in the Kalman fusion.
Ablation (both scale-free):

| config | NEAR μ-beats-e5 rate | final CE ↓ |
|---|---|---|
| scale-free + sonnet | **0.360** | 1.857 |
| scale-free, NO sonnet | 0.280 | 1.981 |

Removing the held-out judge hurts the confusable stratum more than removing scale-free negatives —
the meta hook (an independent judge the fusion cannot see) is doing real work.

## 4. Filing metric — directionally positive on the hard cases, marginal in absolute terms

Stratified (diagnose_filing_e5_vs_mu.py), meta-judge (scale-free + sonnet) vs the Filing v1 baseline:

| stratum | e5 R@1 | baseline μ-max R@1 | meta-judge μ-max R@1 |
|---|---|---|---|
| NEAR (confusable) | 0.000 | 0.000 | 0.007 |
| MID | 0.000 | 0.038 | 0.023 |
| FAR (e5 easy) | 0.617 | 0.060 | 0.060 |

Aggregate filing MRR (conditioned rankers; base = untrained `model_prod_namecond_full`, the
pretrained model with a pearltrees card at r=0):

| ranker (MRR ↑) | e5-cos | base (untrained) | Filing v1 tuned | meta-judge |
|---|---|---|---|---|
| mu-max-cond | — | **0.112** | 0.075 | 0.065 |
| mu-elem-cond | — | 0.105 | 0.081 | 0.027 |
| margin-gate (e5⊕μ) | — | 0.206 | 0.191 | 0.204 |
| e5-cos (reference) | **0.294** | 0.294 | 0.294 | 0.294 |

The blunt aggregate result: **fine-tuning HURTS the conditioned μ ranker, and the meta-judge does
not reverse it** — the untrained pretrained base has the best conditioned MRR (0.112), Filing v1
tuning drops it to 0.075, the meta-judge to 0.065. On this small partial-DAG campaign the fine-tune
overfits/narrows the ranker relative to the broad pretrained μ, and calibration cannot buy that
back; e5 (0.294) is unbeaten throughout.

**Honest verdict.** The meta-judge does exactly what the design predicted on the confusable stratum
— where e5 is completely blind (NEAR/MID recall@1 = 0), the calibrated μ reranks the true folder
above e5 on 36% of hard queries (vs 28% without the held-out judge), and it is the only config that
gets any NEAR recall@1 at all (0.007). Both of the user's design points (scale-free negatives, the
information-limited slow timescale) and the held-out-sonnet meta hook are validated in sign. BUT the
absolute filing metric barely moves: e5 still dominates the aggregate because its easy FAR-third
wins (recall@1 0.617 there) swamp the hard-case reranking, which mostly lifts the true folder toward
the middle of the list rather than to #1. So this is a **directionally-confirmed but practically-
marginal** result — the calibration mechanism works and helps exactly where predicted, but at this
scale (300-row overlap, partial DAG, ~200 NEAR queries) it does not overturn e5 as the deployed
ranker. The value is a proof-of-mechanism for CE-calibrated μ + the precision-gated two-timescale,
and a clear signal that the leverage is on the confusable cases — which points at more overlap data
and the distinct-role / learned-metric extensions rather than declaring a ranker win here.

A second, blunter finding sits underneath (§4 table): on this small partial-DAG campaign the
fine-tune itself HURTS the conditioned μ ranker relative to the untrained pretrained base (0.112 →
0.075 → 0.065), and the meta-calibration does not recover it. The pretrained μ is broader than what
300 overlap rows can improve; the campaign narrows it. So the deployable conclusion for Filing v1 is
unchanged and if anything sharper: **rank with e5 (0.294); if a μ ranker is wanted, the untrained
pretrained conditioned μ (0.112) beats every fine-tuned variant** — the campaign is too small/narrow
to help the ranker, and the fused model's real value stays the label factory + conflict router, not
the ranking head. The meta-judge's contribution is a validated mechanism (precision-gated
two-timescale, scale-free CE, held-out anchor) awaiting a corpus large enough to exercise it.

## Caveats

- All the confusable-stratum recall@1 numbers are tiny in absolute count (0.007 = 1/134); the
  μ-beats-e5 RATE (over ~200 NEAR queries) is the more stable signal, and even it moves by ~0.01–0.08.
- Single 300-row overlap, partial Pearltrees DAG, one node-disjoint split — same coverage limits as
  Filing v1; a null-sized effect is expected to be noisy.
- The meta-judge checkpoint is Pearltrees-only (the Filing v1 finding-6 drift applies unchanged).
- Sonnet judged 300 pairs via subagents (not the production codex path); provenance is the subagent
  transcripts, not a scored-with-codex run.

## Repro

```
# sonnet held-out scoring (subagents) → assemble
python3 assemble_sonnet_scores.py
# train (scale-free + sonnet = the design; ablations via --neg-alpha 0 / --no-sonnet)
python3 train_meta_judge.py --steps 800 --neg-alpha 1.0 --out model_pt_meta_judge.pt
# evaluate on the filing metric + confusable-stratum diagnostic
python3 eval_pearltrees_filing.py --tuned model_pt_meta_judge.pt
python3 diagnose_filing_e5_vs_mu.py --tuned model_pt_meta_judge.pt
```
