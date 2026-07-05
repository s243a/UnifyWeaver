# Uncertainty / multi-source μ-estimation — playbook

*Start here whenever you're combining multiple signals into a μ / confidence / relation estimate, or reaching
for "confidence weighting." This encodes lessons the project learned the slow way (see the arc: #3356 → #3357 →
#3359 → #3387 → #3391, and the SYM dual-judge course-correction, `DESIGN_sym_estimation_integration.md`).*

## The one-line rule

**Don't hand-set independent confidence weights over correlated sources. Fit a LEARNED, CALIBRATED combiner
(`JointPosterior`) on HELD-OUT data, and use confidence as a MARGIN GATE, not a per-item weight.**

## The five pitfalls (each one we actually hit)

1. **Sources are not independent → naive/inverse-variance fusion over-confidences.** In this project every
   model readout *consumes frozen e5*, so `e5`, `μ_SYM`, `μ_HIER`, `μ_ELEM` are correlated (measured **+0.751**,
   #3359). A product-of-experts / precision-weighting that assumes independence double-counts the shared signal.
   Evidence: a factored equal-weight PoE scored **50.9%, below the 51.1% majority baseline** (#3359). *Fix:* a
   learned joint head over the full vector (it down-weights the redundant part), or model the residual /
   down-weight by separability (#3357).
2. **Anti-correlated sources carry structure that additive weighting throws away.** Subcategory and element
   readouts are *anti-correlated*, and conditioning on both *jointly IS the directional asymmetry* (#3359). Two
   separate positive weights can't represent that; a joint head can.
3. **Confidence is a WEAK per-query signal → it's a GATE, not a weight; and it's MARGIN, not level.** Per-query
   ρ(confidence, correctness) is +0.03…+0.15 with CIs crossing 0 (#3391). Absolute μ *level* is even
   anti-correlated with correctness on under-trained checkpoints; **margin (top1−top2)** is the better
   selective-risk gate (`AURC_margin < AURC_level`, #3391). So use confidence to *route / abstain*, not to
   multiply a per-item contribution.
4. **Data-limit / "how much training support" ≠ graph degree.** A node's degree is its *own* edge count; it does
   not measure whether *nearby (e5-similar)* nodes were trained, which is what governs a readout's reliability
   (the model generalises through e5). #3356 used a **label-confidence tag + `P(μ|op)` calibration against the
   tagged set**, re-estimated as the model evolves — *not* a structural proxy. If you need a data-density term,
   use trained-neighbour density in e5 space or the `P(μ|op)` calibration, and treat it as a gate.
5. **Measure on HELD-OUT, node-disjoint splits — never on training pairs.** Correlating a model's own readouts
   against a target it was trained on inflates *and can reorder* effect sizes (a readout ordering measured on
   training pairs can flip held-out). Split so no node appears in both train and held.

## The workflow (what "doing it right" looks like)

1. **Assemble the source vector** — every candidate signal (static e5, each model readout fwd/rev, any external
   judge, and *decorrelated* structural signals like `1/d`). Tool: `mu_posterior.py` (`e5_mu_fn`,
   `model_readout_fn`, `struct_dist_fn`).
2. **Look before you combine** — print the **separability** per source and the **correlation matrix**. High |r|
   ⇒ redundancy; a genuinely new source should be *decorrelated* from the existing cluster (that's its whole
   value — e.g. `1/d` covers the lateral/sibling axis the vertical membership operators miss).
3. **Fit the calibrated joint head** (`JointPosterior`, logistic or small MLP) on a **held-out, node-disjoint**
   split. Compare against the factored PoE (equal + separability-weighted) as controls.
4. **Report the honest metrics:** accuracy, log-loss, **ECE with a stated binning** (bins sensitive to choice),
   and **AURC gated by margin, with a bootstrap 95% CI** (AURC is noisy on small held-out sets). A new source
   *earns its keep* iff the with-source AURC CI sits below the without-source point estimate.
5. **Ablate each source** (with vs without) on the *same* split. Redundant sources show overlapping intervals.

## Anti-patterns (stop if you're doing these)

- Writing per-item `c_k` confidence multipliers and inverse-variance-averaging them. (→ pitfall 1/3.)
- Treating `corr(signal, target)` on the training set as a source's reliability. (→ pitfall 5.)
- Using absolute μ level as the confidence gate. (→ pitfall 3.)
- Node degree as a "data density" / confidence term. (→ pitfall 4.)
- Adding a source and asserting it helps because it's conceptually motivated — *test it* (step 4/5).

## Tools & references
- **`mu_posterior.py`** — `MuPosterior` (per-source `P(μ|rel)`, bands, separability), `JointPosterior` (the
  calibrated combiner), `struct_dist_fn` / `e5_mu_fn` / `model_readout_fn` (sources), `aurc`/`aurc_boot`/
  `margin_conf` (the gate + selective-risk metric). Tests: `test_mu_posterior.py`, `test_mu_posterior_dist.py`.
- **Design:** `DESIGN_mu_sources_and_estimation.md` (#3357), `DESIGN_sym_estimation_integration.md` (the
  course-correction), `DESIGN_inferred_operator_superposition.md` §2–3 (#3356 calibration + noise decomposition).
- **Evidence:** `REPORT_mu_posterior.md` (#3359 — the correlation numbers, joint > PoE), `DESIGN_model_applications.md` / PR #3391 (margin-not-level, AURC), `REPORT_eval_methodology.md` (#3387).
