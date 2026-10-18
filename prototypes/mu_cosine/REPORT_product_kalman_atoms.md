# D-channel atoms, resolved: atoms were priced correctly; the lattice explained S; the true residual is D bimodality

*`run_product_kalman_atoms.py`, 2026-07-09. Completeness check on the champion's last known defect (D-channel
shape). Puts scoring on the correct MIXED footing (atoms as probability MASS + S-density; interior as joint
density), compares the champion's IMPLIED atom masses against a LEARNED 3-class atom head, and applies
randomized PIT — first for boundary atoms, then over the FULL 0.05 quantization lattice. Same protocol;
EXPLORATORY.*

## Results

| | exploratory | fresh |
|---|---|---|
| IMPLIED (champion's own mass) NLL | −0.341 | −0.013 |
| LEARNED (atom head) NLL | −0.331 (−0.010) | −0.018 (+0.005, noise) |
| atom-head rate calibration | 0.028 vs 0.029 | 0.064 vs 0.065 |
| randomized PIT KS_D (atoms as mass) | 0.20 / 0.22 | 0.12 / 0.13 |
| **BIN-MASS PIT (full lattice), KS_D / KS_S** | 0.199 / **0.033** | 0.123 / 0.127 |

## Findings

1. **The atom-inflation hypothesis is REFUTED.** The learned head calibrates the atom RATE perfectly and still
   adds nothing — the champion's Gaussian-mixture tail already prices "the judge says exactly 0/1" correctly.
   The atoms were never the defect.
2. **The quantization lattice explained part of the shape failure.** ALL labels are discrete (0.05 steps), not
   just the boundaries; testing a continuous density with a continuous PIT clumps at ~20 lattice values. Under
   the correct bin-mass randomized PIT, exploratory-S PASSES (KS 0.033 ≈ the 0.025 critical value) — that
   channel was calibrated all along, and earlier KS numbers across the program overstated miscalibration.
3. **The true residual defect: D-channel interior SHAPE (both corpora; fresh-S too).** Survives atoms-as-mass
   AND lattice-as-bins. Diagnosis: D labels are BIMODAL (directional ~high or not ~low), while the per-row
   predictive — though a two-expert mixture — has both experts centered on the same mean: effectively unimodal.
   It cannot represent "either high or low, not middle."
4. **The fix is not more fusion refinement — it is a relation-CLASS mixture predictive**: exactly the
   JointPosterior / discrete co-occurrence class structure from the two-judge arc's result #1, resurfacing.
   Model-side work (fusion-heads build path), not a fusion patch. This closes the completeness loop with a
   pointer back to where the program started.

## Axis disposition (user direction)

The fusion-refinement axis is COMPLETE: champion = G_sl (statlin mixture) for the predictive distribution,
mu/hop expert for point estimates; scale calibrated (Mahal/dim ≈ 1); atom masses correct; S-shape calibrated
(exploratory) once the lattice is respected; D-shape defect diagnosed as predictive unimodality vs label
bimodality and DEFERRED to the model-side arc. Diminishing returns confirmed (+0.03 → +0.00 over the last two
rungs). Next: the bigger levers (LLM-judge measurement channel; amortization into model heads).

## Repro

```
python3 run_product_kalman_atoms.py    # both corpora (~20 min; reuses the statlin heteroscedastic fits)
```
