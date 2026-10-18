# Statistically-linearized transport: new NLL champion; shape half-fixed (S yes, D no)

*`run_product_kalman_statlin.py`, 2026-07-09. Builds the statistical-linearization refinement (user): each label
is a NARROW DISTRIBUTION (uniform over the quantization half-step), transported to logit space by 3-pt
Gauss–Legendre quadrature → per-row (ell_mean, ell_var, L*). Structural Sigma(h) fit by heteroscedastic MLE with
ell_var as KNOWN per-row noise (no double-count); scoring adds it back; mu-space density uses the
distribution-averaged Jacobian L* (finite at boundaries: L*=106 at y=0 vs point J=10^4 — endpoints keep weight,
as designed). Same protocol; EXPLORATORY.*

## Results (NLL mu-density; PIT-KS vs uniform, ~0.025 = calibrated shape)

| rung | expl NLL | fresh NLL | expl KS_D/KS_S | fresh KS_D/KS_S |
|---|---|---|---|---|
| F mu/hop | −0.456 | −0.174 | 0.183 / 0.052 | 0.081 / 0.179 |
| E_pt logit point (merged ref) | −0.447 | −0.273 | 0.178 / 0.061 | 0.130 / 0.142 |
| **E_sl logit statlin** | **−0.513** | **−0.327** | 0.198 / 0.051 | 0.137 / 0.107 |
| G_pt mix(F,E_pt) (prev champion) | −0.551 | −0.312 | 0.180 / 0.043 | 0.112 / 0.162 |
| **G_sl mix(F,E_sl)** | **−0.584** | **−0.342** | 0.197 / **0.041** | 0.123 / **0.132** |

Gains: E_pt→E_sl +0.066/+0.054; G_pt→G_sl +0.033/+0.030 (row-SE ~0.005, stability only).

## Findings

1. **New NLL champion on both corpora (G_sl).** The statistical linearization pays consistently at both the
   expert and mixture level. The S-channel PIT also improves (transform-invariant, so this is genuine shape
   gain, not L* density-conversion credit).
2. **The D-channel shape does NOT improve** (KS_D ~0.12–0.20 everywhere; calibrated ≈ 0.025). D is where the
   exact-boundary ATOMS concentrate. The soft-blob transport handles *near*-boundary mass; a true point mass is
   not a narrow distribution. The residual failure is the discrete component asserting itself: the principled
   fix remains a discrete/continuous mixture likelihood — or wider transport for exact-boundary reports (a
   judge's "0.0" may mean "somewhere in [0, 0.1]", coarser than the half-step).
3. **The three boundary treatments are now unified and ranked:** de-quantization (move the point) < Jacobian
   weighting (zero the weight; rejected earlier) < statistical linearization (average the slope; wins). One
   mechanism — "a boundary report is a narrow distribution" — with the quadrature version its best rendering
   short of an explicit discrete component.

## Production recipe after this arc

- Predictive distribution: **G_sl** — mixture of the mu/hop Kalman and the statistically-linearized logit/hop
  Kalman, constant w fit on calibration.
- Point estimate: the mu/hop expert alone (unchanged; the BG verdict).
- Known remaining defect: D-channel shape (boundary atoms) — discrete/continuous mixture likelihood is the
  designed next step, with diminishing returns expected (+0.03 was this arc's gain).

## Repro

```
python3 run_product_kalman_statlin.py     # both corpora; heteroscedastic MLE makes this the slowest runner (~20 min)
```
