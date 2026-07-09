# Lever A: the judge as a measurement channel — reliability bound, +1.94 NLL, and conflict-routed decisions

*`run_judge_channel.py`, 2026-07-09. Adds the LLM judge as a SECOND measurement channel via an independent
re-scoring run of the fresh Behavior corpus (250 pairs, gpt-5.5-low, 0 failures), and answers the deployment
question "when is a judge call worth it?" under both NLL and DECISION utility. Constant covariance blocks,
correlated 5x5 joint (prior 2 + graph 1 + judge 2), 40 descendant-disjoint splits. EXPLORATORY.*

## 1. Judge self-consistency — the long-requested reliability bound

Same 250 pairs, two independent runs of the same judge:

| channel | corr(j1, j2) | MAE | sd(diff) | R_judge (iid) |
|---|---|---|---|---|
| D | **+0.954** | 0.064 | 0.093 | 0.0043 |
| S | +0.766 | 0.071 | 0.091 | 0.0041 |

The judge is HIGHLY reliable on directional labels, moderately on symmetric (consistent with S being the
genuinely fuzzier relation). Per-label noise sd ≈ 0.065. This retroactively bounds the "single LLM judge"
caveat carried by every prior result.

## 2. The fusion ladder — the judge channel's value

| rung | NLL ↓ | Mahal/dim |
|---|---|---|
| prior | +0.408 | 1.42 |
| prior+graph | −0.011 | 1.40 |
| prior+judge | −1.913 | 1.10 |
| **prior+graph+judge** | **−1.947** | **1.09** |

Judge-channel value: **+1.94 NLL/row** (vs the graph's +0.42) — and it also fixes most of the calibration
(1.40 → 1.09). **Honest circularity flag:** the target IS judge1 and judge2 shares its family biases, so this
is value against judge-DEFINED truth — the operating truth for filing, but an upper bound w.r.t. any
independent truth. The correlated 5x5 joint prices the judge2↔target correlation; an independent-PoE fusion
here would catastrophically double-count (the earlier anti-scaling result, now with a live confident channel).

## 3. NLL routing — a theory-confirming NULL

Cumulative NLL gain captured vs fraction of judge calls: hop-IV, ambiguity, and conflict all sit ON the random
diagonal; only the (hindsight) oracle beats it (47% @ 25%). **This is what §7's autonomous-covariance principle
predicts:** with constant covariance blocks, expected information value is context-independent, so no observable
policy can beat random on EXPECTED NLL gain — the oracle's edge is the unexploitable noise realization. The
null and the theory agree.

## 4. DECISION routing — the deployment answer

NLL is the wrong deployment objective; filing is a decision (posterior D vs 0.5). Measured: the judge FLIPS
25% of decisions, and **85% of flips are corrections** (toward judge-truth). Fraction of flips captured:

| policy | 10% calls | 25% | 50% | 75% |
|---|---|---|---|---|
| random | 11% | 25% | 50% | 74% |
| hop-IV | 9% | 29% | **59%** | **84%** |
| ambiguity | 14% | 25% | 50% | 82% |
| **conflict** | 12% | **34%** | **59%** | 79% |
| oracle (NLL) | 26% | 48% | 65% | 75% *(sub-random!)* |

- **`conflict` (graph-vs-prior innovation) is the winning observable policy:** 1.36x random at a 25% budget.
  When the existing channels disagree, the judge's arbitration is most likely to change the decision.
- The NLL-oracle goes SUB-random past 50% — NLL-gain and decision-value are genuinely different objectives.

## Production recipe (Lever A, v1)

Fuse the judge as a correlated measurement channel when present (+1.94 NLL, calibration 1.09). Route judge
calls by CHANNEL CONFLICT under a budget; expect ~⅓ of decision flips captured at a ¼ budget, with 85% of
flips being corrections. Do NOT route by expected NLL gain (provably ≈ random under constant covariance).

## Caveats

Same-judge-family circularity (target = judge1); one corpus; constant blocks (hop-conditioning entered only the
IV policy); the oracle is an NLL-oracle (a flip-oracle would be the true decision ceiling); flip threshold 0.5
un-tuned; routing lifts are modest (1.1–1.4x) — real but not dramatic in this homogeneous slice; heterogeneous
corpora should widen policy separation.

## Repro

```
# second judge run (needs node 22 for codex scoring):
python3 score_with_codex.py --pairs /tmp/mu_data/sigma_hop_fresh_pairs.tsv --batch 10 \
  --out /tmp/mu_data/sigma_hop_fresh_scored_j2.tsv --responses /tmp/mu_data/sigma_hop_fresh_responses_j2.txt
python3 run_judge_channel.py
```
