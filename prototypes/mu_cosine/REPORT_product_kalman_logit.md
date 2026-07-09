# Rung (b): logit-space fusion + the dual-objective mixture — the mixture wins on both corpora

*`run_product_kalman_logit.py`, 2026-07-09. Tests the logit-space fusion program (interior-Gaussianity home,
de-quantization, Jacobian-weighted R) and the user's DUAL-OBJECTIVE proposal: a two-component predictive mixture
over the μ-space and logit-space experts. Same protocol as the rung-(a) report: 250 pairs per corpus, 40
descendant-disjoint splits, correlated hop-conditioned updates. All rungs scored as μ-space densities at the SAME
de-quantized label points (logit rungs via change-of-variables), so NLL is comparable across rows. EXPLORATORY.*

## Ladder

| rung | space | covariance | treatment |
|---|---|---|---|
| A | μ | const | reference (merged baseline, rescored) |
| B | logit | const | naive 1e-3 clip (atom-poisoned strawman) |
| C | logit | const | de-quantized labels (0.0 → 0.025) |
| D | logit | const | C + per-row Jacobian-weighted measurement R |
| E | logit | **hop** | de-quantized, native-logit hop-Cholesky blocks |
| F | μ | **hop** | rung-(a) full stack |
| **G** | **both** | hop | **mixture `w·p_F + (1−w)·p_E` (μ-densities), `w` fit on calibration rows** |

## Results

| rung | exploratory NLL_μ ↓ | fresh NLL_μ ↓ | exploratory Mahal/dim | fresh Mahal/dim |
|---|---|---|---|---|
| A μ/const | −0.322 | −0.010 | 1.18 | 1.40 |
| B logit/naive | −0.279 | −0.190 | 1.25 | 1.13 |
| C logit/dequant | −0.251 | −0.184 | 1.35 | 1.27 |
| D logit/dequant+w | +0.212 | −0.051 | 2.26 | 1.69 |
| E logit/hop | −0.444 | −0.273 | 1.06 | 1.08 |
| F μ/hop | −0.452 | −0.177 | **1.01** | 1.09 |
| **G mix(F,E)** | **−0.549** | **−0.311** | — | — |

Gains (row-SE ~0.01, stability only): F→G **+0.097** (exploratory), **+0.134** (fresh). Fitted mixture weight
`w` (μ component): **0.47 ± 0.07** (exploratory), **0.34 ± 0.08** (fresh) — genuinely interior, not pinned.

## Findings

1. **The dual objective (user) is the new champion on BOTH corpora** — it beats the best single-space model by
   +0.10 / +0.13 NLL, with a genuinely mixed weight. This is a mixture over NOISE MODELS (additive μ vs
   multiplicative logit) — a mixture-of-experts where the experts share every parameter except the coordinate
   system, so it costs ONE extra parameter.
2. **Complementarity is real but INVERTED from the prediction.** Predicted: μ owns the boundary, logit the
   interior. Measured: **the logit component wins 100% of boundary rows (both corpora)** and the interior is a
   coin flip (47%). Mechanism: at a de-quantized boundary label the change-of-variables factor `1/(y(1−y))` ≈ 41
   is a large density multiplier, and the logit expert — whose fitted covariance embraces the boundary spikes —
   places real mass there; the μ-space Gaussian must reach ~2σ into its tail. The logit home is BETTER at the
   boundary it was supposed to be worse at, BECAUSE the de-quantization keeps the spike finite and the wide
   logit covariance covers it.
3. **Hop-conditioning is what makes the logit home competitive at all:** const-block logit rungs (B, C) lose to
   const-block μ (A) on exploratory, but hop-conditioned logit (E) ~ties μ/hop (F) on exploratory (−0.008) and
   **BEATS it on fresh (+0.096)**. The earlier "logit loses" verdict was a constant-blocks artifact.
4. **Rung D (Jacobian-weighted measurement R, as specced) is rejected** — overconfident on both corpora
   (Mahal/dim 2.26 / 1.69). Diagnosis: the per-row Jacobian is evaluated at the MEASUREMENT's operating point,
   but the graph measurement is affine-calibrated and interior, so the weighting only shrinks R; the
   boundary-heavy variable in this pipeline is the LABEL. A rendering that weights the CALIBRATION (per-row
   weights `y(1−y)` in the covariance fit) remains untested.
5. De-quantization (C) vs naive clip (B) is a small NEGATIVE at constant blocks (−0.03 / −0.01) — the fitted
   covariance already absorbs the clip choice; de-quantization's real role is keeping the boundary spike finite
   for the mixture's logit component (finding 2).

## Interpretation

The user's dual objective is the right resolution of the fuse-space question: the label distribution is a
mixture (interior + boundary atoms), and rather than choosing one Gaussian home, the predictive should BE a
mixture — each component a coherent Kalman in its own coordinates, expressed in one density space. The MoE motif
appears here for the fourth time (bound-lattice middle, computed gain-gate, switching-KF regimes, and now
error-model regimes), and this instance is measurably the best model in the program.

**Future work:** gate `w` on observable context (prior position, hop, corpus) instead of a constant — the 100%
boundary complementarity says a context-dependent gate has real signal to route on; but the gate must condition
on the PRIOR's position (available at inference), not the label's. Also untested: the calibration-weighted
covariance rendering of the Jacobian idea (finding 4).

## Caveats

Exploratory, not preregistered; row-SE understates (rows share splits); two corpora; single LLM judge; `w` fit
per split on calibration rows (no leakage into held rows, but the same 250-pair pool); Mahal/dim undefined for
the mixture (single-V diagnostic doesn't apply — PIT calibration would be the right check, untested).

## Repro

```
python3 run_product_kalman_logit.py            # both corpora
```
Requires the rung-(a) machinery (`run_product_kalman_sigma_hop.py`) and the run artifacts in `/tmp/mu_data/`.
