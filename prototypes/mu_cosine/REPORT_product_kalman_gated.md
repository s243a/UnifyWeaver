# Gating the dual-objective mixture: constant-w stands; the computed gate's failure separates mean from distribution

*`run_product_kalman_gated.py`, 2026-07-09. Tests three gate-training principles on the merged dual-objective
mixture (REPORT_product_kalman_logit.md), on identical hop-conditioned components (F = mu/hop Kalman, E =
logit/hop Kalman). Adds the calibration check that works for mixtures (per-channel PIT/KS) and two diagnostics
from discussion: complementarity in projected-ERROR vs DENSITY terms, and the graph-misclassification
robustness test. Same protocol: 250 pairs x 2 corpora, 40 descendant-disjoint splits. EXPLORATORY.*

## The gate ladder

| gate | principle | fitted params |
|---|---|---|
| G const-w | density-mixture NLL, grid | 1 |
| H1 gated-NLL | logistic gate on (prior-boundary-proximity, hop), ridge-regularized | 3 |
| H2 gated-error | (user) responsibilities from PROJECTED direct-space errors → regularized logistic | 3 |
| H3 computed-BG | (user) total error = weighted sum of projected errors → Bates–Granger/Kalman w, computed | 0 |

The projection (user): the logit expert's posterior mean maps to direct space EXACTLY via sigmoid (the median of
the pushforward); errors then live in one space where they ADD LINEARLY under convex combination — which makes
the optimal point-fusion weight closed-form (Bates–Granger = the scalar Kalman gain for two correlated
estimates).

## Results (held-out NLL, mu-density terms)

| gate | exploratory | fresh |
|---|---|---|
| F mu/hop (component) | −0.461 | −0.176 |
| E logit/hop (component) | −0.453 | −0.284 |
| G const-w | −0.554 | −0.312 |
| H1 gated-NLL | −0.580 (+0.025) | −0.313 (+0.001) |
| **H2 gated-error** | **−0.590 (+0.035)** | −0.311 (−0.001) |
| H3 computed-BG | −0.463 (−0.091) | −0.248 (−0.064) |

H1 note: an UNREGULARIZED first run saturated (coefficients →∞ = hard routing; +0.080 on exploratory that did
NOT replicate on fresh). The ridge version keeps a smaller gain with sane coefficients (prox +0.42, hop +0.89).

## Findings

1. **Constant-w remains the production recommendation.** Gates add up to +0.035 where observable context
   predicts regime (exploratory: logit wins 77% of prior-interior rows in density terms) and collapse to
   ≈constant where it doesn't (fresh: 49%/51% coin flip). Regularized gates degrade gracefully; unregularized
   ones overfit to hard routing. The user's error-trained H2 is the best gate where signal exists.
2. **H3's failure is the most informative cell.** Bates–Granger computes w = 1.00 (exploratory) — for POINT
   estimates, use the mu expert alone (its projected errors dominate and the experts' errors are highly
   correlated, so blending buys nothing). Yet the density mixture at w≈0.47 beats mu-alone by +0.10. Both are
   optimal — for different objectives. **The mixture's entire win is uncertainty-shape value, not mean value.**
   Architecture consequence: report point estimates from the mu expert; report uncertainty from the mixture.
3. **The 100%-boundary complementarity was mostly the Jacobian.** Re-measured in projected-ERROR terms: logit
   closer on only 57%/66% of boundary rows (vs 100% in density terms). Density-complementarity ≠
   error-complementarity — the density view credits the logit expert for the change-of-variables factor (~41x
   at de-quantized boundaries), the error view does not.
4. **Graph-misclassification robustness (user prediction): mixed, and diagnosable.** Predicted: the logit
   expert degrades more when the graph misclassifies (unbounded log-odds influence vs mu's bounded innovations).
   Measured (|m−D| > 0.25): fresh mildly supports (ratio logit/mu 1.07→1.16), exploratory opposes (1.48→1.09).
   Diagnosis: the affine calibration COMPRESSES the graph channel's range, so logit(m) never reaches the extreme
   log-odds where the unbounded-influence asymmetry bites. The mechanism is real but this channel structurally
   cannot trigger it — it becomes relevant when an uncalibrated confident channel (e.g., a raw LLM judge
   asserting mu=0.98) is wired in. No innovation gate feature is justified by current evidence.
5. **PIT rejects uniformity for every rung** (KS 0.04–0.19 vs ~0.025 critical at n≈3000): Mahal/dim ≈ 1 says the
   SCALE is right; PIT sees the SHAPE mismatch (boundary atoms; non-Gaussian residue). All models remain
   shape-miscalibrated; the mixture improves S-channel KS slightly, not D. Honest open issue.

## Where the fusion program stands after this arc

- Best model: **the constant-w dual-objective mixture** (merged) — +0.10/+0.13 over the best single space.
- Best point estimator: **the mu/hop expert alone** (H3's verdict).
- Gates: pay only where observable context predicts regime; corpus-dependent; H2 (error-trained) preferred when
  used. The MoE gate did NOT earn a permanent place — the computed-gain principle ("learn the uncertainty,
  derive the weights") survives for the FUSION but the mixture weight resists being computed from point-error
  statistics because it prices distribution shape.
- Open: PIT shape-miscalibration (boundary atoms — a discrete/continuous mixture likelihood would be the
  principled fix); calibration-weighted covariance fit (untested rendering of the Jacobian idea); judge channel
  as a second measurement (where the robustness asymmetry should actually bite).

## Repro

```
python3 run_product_kalman_gated.py        # both corpora
```
Requires the merged rung-(a)/(b) machinery and `/tmp/mu_data/` artifacts.

## Future work: statistical linearization of the mu↔logit transport (user, 2026-07-09)

Two additions from post-merge discussion:

**Gate taxonomy, settled.** The textbook MoE gate IS error weighting in a common space: classical
responsibilities with Gaussian experts are a softmax over scaled negative squared errors, well-defined only once
the errors are projected into one space. In this ladder H2 is the textbook gate; the density-trained gates
(G/H1) are the mixture-model-matched variant differing by exactly the change-of-variables term (hence 100% vs
57–66% boundary complementarity); H3 is not a gate but the optimal GLOBAL point-blend (Bates–Granger).

**Quasilinearization / statistical linearization (user refinement, deferred).** Replace the POINT Jacobian used
to transport random variables between direct and logit space with the linearization that minimizes expected
squared error over the variable's (narrow) uncertainty distribution — statistical linearization (Kazakov; the
statistically-linearized filter; the unscented transform is its sigma-point implementation). For Gaussian x,
Stein's lemma gives the closed form:

```text
L* = Cov(x, g(x)) / Var(x) = E[g'(x)]     (the distribution-averaged Jacobian)
```

Consequences:
- **Endpoints are no longer zero-weight** (user's point): in the logit→mu direction the weight becomes
  E[sigmoid'(ell)] > 0 for any finite-width distribution — the point-Jacobian's boundary zero was an artifact of
  treating an uncertain report as a point.
- **Subsumes de-quantization:** moving the point (0.0 → 0.025) was a hack for the same fact (boundary reports
  are uncertain); statistical linearization smooths the SLOPE using the report's actual width (the quantization
  half-step is the natural input distribution). One mechanism replaces two.
- **Fixes variance transport:** the point Jacobian under-propagates uncertainty where the link's curvature is
  extreme (the boundary); minimizing over the distribution adds the curvature-induced spread (the UT
  correction). This is also a candidate treatment for the PIT shape-miscalibration left open above.
