# Transitive ordinal constraint — held-out verification (stage 3)

First empirical test of the merged transitive implementation. Baseline vs treatment (`--transitive-weight 1`),
warm-start from `model_nodetype.pt`, 200 steps, 2 seeds, evaluated on a **leakage-aware node-split holdout**
(825 pairs whose destination nodes are held out → no shared endpoints with the training triples).

| seed/arm | satisfaction `μ_trans≤μ_bound` | μ_bound (anti-collapse) | μ_trans | gap |
|---|---|---|---|---|
| 1 / base | 80% | 0.511 | 0.201 | 0.31 |
| 1 / treat | 93% | 0.837 | 0.220 | 0.62 |
| 2 / base | 77% | 0.330 | 0.169 | 0.16 |
| 2 / treat | 93% | 0.775 | 0.127 | 0.65 |
| **base mean** | 78% | 0.42 | — | 0.24 |
| **treat mean** | **93%** | **0.81** | — | **0.63** |

## Verdict — PASS, and it replicates (both seeds)
1. **Anti-collapse passed decisively (the opposite of collapse):** treatment `μ_bound` *rose* to 0.78–0.84
   (≈ true REL_SPEC 0.9), vs baseline 0.33–0.51. The ranking loss **reinforced** the bound-edge calibration
   instead of gaming the constraint by dropping μ — and **stabilised** it (treat tight, base swings).
2. **Generalisation:** 93% satisfaction on both seeds (vs ~78% baseline) on a **leakage-aware node-split**
   holdout → genuine generalisation, not memorisation.
3. **Decay learned:** the gap ~doubled (0.24 → 0.63), consistently.

## Honest caveat
The **baseline is undertrained** (μ_bound 0.42; warm-start never saw these physics/EE nodes, 200 steps), so
part of treatment's μ_bound *magnitude* is faster convergence, not pure generalisation — a longer-baseline run
would isolate that. But the replicated **93% satisfaction** and the **no-collapse** are robust regardless.

## Why this lands where the tail augmentation didn't
Clean **graph-truth** target (no LLM, no judge-noise), so the eval is **deterministic with no agreement
ceiling** — it can fully resolve the effect, and it came back a tight, repeatable yes. Contrast the
inferred-tail: ~80% judge-noise, ±0.15 metric, inconclusive vs an independent judge.

## Justifies the stage-2 upgrades
The min-bound, naive-CE floor works and generalises → the deferred upgrades are now worth pursuing:
heteroscedastic loss (per-pair variance), dual-ascent λ (target a satisfaction rate), noisy-OR multi-path,
and the LLM-anchored multi-factor term (a better `μ_bound` estimator). A **converged-baseline** A/B (more
steps) should come first, to isolate the pure-generalisation magnitude from convergence speed.
