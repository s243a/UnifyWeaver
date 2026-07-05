# Integrating the SYM dual judge into the μ-estimation architecture

*(joint posterior + margin gate + the distance source — a course-correction, 2026-07-05)*

## TL;DR

The SYM dual-judge work (`DESIGN_sym_dual_judge.md`) built a **hand-set, inverse-variance precision fusion**
of structural signals (`--struct-blend precision|membership`, fixed `c_dist/c_subcat/c_elem`). Reviewing the
project's own estimation PRs (#3356, #3357, #3359, #3387, #3391) shows that fusion is the **wrong shape on
three counts**. The project already has the right machinery — the **calibrated `JointPosterior`**
(`mu_posterior.py`, #3359) over the full μ-readout vector, **gated by margin** (#3391). The genuinely new and
non-redundant contribution of the dual-judge work is **one source**: the structural-embedding distance `1/d`,
which is decorrelated from the model/e5 cluster and carries the **lateral / sibling** axis the model readouts
structurally miss. **Plan: add `1/d` to the JointPosterior's source vector and let the calibrated head learn
the combination; keep the hand-fusion modes only as A/B controls.**

## Why the hand-set precision fusion is wrong (three findings)

The fusion `μ_graph = Σ_s (c_s·region_s·value_s) / Σ_s (c_s·region_s)` is optimal only for **independent**
estimators weighted by **reliable per-item confidences**. Neither holds:

1. **The sources are correlated — inverse-variance over-confidences (#3357, #3359).** The model *consumes*
   frozen e5, so `μ_SYM`, `μ_HIER`, `μ_ELEM` (all model readouts) share signal. #3359 *measured*
   **e5↔model μ correlation = +0.751**. Combining correlated votes as if independent double-counts the shared
   e5 signal — exactly the over-confidence #3357 §1 warns about. #3359 confirmed it: a **factored
   product-of-marginals with equal weights scored 50.9%, *below* the 51.1% majority baseline**; even the
   #3357 separability-weighted correction only reached 51.1%.
2. **Subcategory and element are ANTI-correlated — their joint is the asymmetry (#3359, Finding 4).** I treated
   `c_subcat` and `c_elem` as two independent *positive* confidences added together. But #3359 found subcat and
   element **anti-correlated**, and *conditioning on both jointly IS the directional asymmetry* (realised from
   data, no hand-built term). Additive independent weighting throws that structure away.
3. **Confidence is a weak per-query signal — a GATE, not a per-item WEIGHT (#3391).** #3391 established the
   confidence signal is **margin (top1−top2 μ), not level** (level is *anti-correlated* with correctness on
   under-trained checkpoints; `AURC_margin < AURC_level` on all four checkpoints). And even margin is a **weak
   per-query predictor** (Spearman ρ +0.03…+0.15, CI∋0 on 3/4 checkpoints) — good for **selective-risk gating**
   (which source to trust, when to abstain), **not** for a precise per-pair `c·value` multiplier. My hand-set
   per-pair confidences ask more of confidence than it can deliver.

Corollary — the **data-limit accounting** was also off: my `region` used **graph degree** as a data-density
proxy. #3356 did *not* use graph degree; its confidence was a **label-confidence tag + `P(μ|op)` calibration
against the tagged (training) set** (re-estimated as the model evolves). Degree measures an endpoint's *own*
training-edge count, not whether *nearby (e5-similar)* nodes were trained — which is what governs the readout's
reliability, since the model generalises through e5.

## The correct architecture (already in the project)

```
sources  →  JointPosterior (calibrated, held-out)  →  P(relation | μ-vector)  →  margin gate (selective risk)
```

- **`JointPosterior` (`mu_posterior.py`, #3359)** — a small discriminative head (logistic / 1-hidden MLP) over
  the **whole** μ-readout vector, fit on **held-out tagged data**. It *learns* the combination, so it absorbs
  the e5↔model correlation, the subcat↔element anti-correlation, and stays **calibrated** (confidence matches
  accuracy). This is #3357's "calibrated log-linear combiner" recommendation, implemented and shown to beat the
  product-of-marginals.
- **Margin gate (#3391)** — use `top1−top2` of the posterior as the **selective-risk gate** (route / abstain),
  not a per-pair weight. `--confidence-mode margin`.
- **`model_readout_fn` (`mu_posterior.py`)** already extracts the full model readout vector (SYM + directionals)
  and *already flags* that these are correlated and "should be down-weighted / decorrelated."

## What is genuinely new: the distance (sibling) source

The dual-judge work's lasting value is **not** the fusion mechanism — it's a **source**:

- **`1/d` (structural-embedding distance) is the one input decorrelated from the model/e5 cluster.** It was
  trained *separately* on graph distance (`structural_embedding.py`, reciprocal target), so it is not a
  re-reading of e5.
- **It carries the lateral / sibling axis the model readouts structurally cannot.** Membership operators are
  vertical (ancestor↔descendant); *siblings have zero membership in either direction* yet are related. Measured
  (14 562 pairs): **sibling/cousin = 37 % of pairs, 0 % membership coverage**, real relatedness (target μ 0.13
  vs distant 0.00), and `corr(1/d, SYM) = +0.136` there — so `1/d` is the *sole* structural signal *covering*
  that 37 % (a **coverage** claim). NB `+0.136` is **weak** (explains <2 % of variance): it justifies adding
  `1/d` as a **candidate** source, and is *not* evidence that `1/d` alone does much work or will improve the
  calibrated posterior. (See `DESIGN_sym_dual_judge.md` "Division of labor.")

So `1/d` is a plausibly **non-redundant** source to add to the vector (#3357) — it covers an axis the correlated
model readouts miss. **Whether it actually helps is the hypothesis this design's follow-up test decides** (joint
head with vs without `1/d`, held-out, separability + AURC — below), not something established here.

## Build plan

1. **Add `1/d` as a source in `mu_posterior.py`.** A `struct_dist_fn(struct_emb)` → `readouts(pairs)` returning
   `3/(1+‖Δ‖)`, alongside `e5_mu_fn` and `model_readout_fn`. Cheap (embedding lookup).
2. **Fit `JointPosterior` over `{e5, μ_SYM, μ_HIER fwd/rev, μ_ELEM fwd/rev, 1/d}`** on held-out tagged pairs.
   Report per-source **separability** and the **correlation matrix** (the tool already does this).
3. **The honest test:** does adding `1/d` **raise held-out accuracy / lower AURC** over the correlated
   model+e5 sources? Compare joint-with-`1/d` vs joint-without, and vs the current hand-fusion. Gate by margin;
   report AURC (bootstrap CI, per #3391) + ECE.
4. **Data-limit term, done right (#3356):** if a per-item confidence is still wanted on top of the joint head,
   use **`P(μ|op)` calibration against the tagged set** (or a trained-neighbour-density in e5 space), **not**
   graph degree — and use it as a **margin gate**, not a weight.

## Status of the hand-fusion modes

`--struct-blend inside | outside | precision | membership` and the fixed `c_dist/c_subcat/c_elem` buffers stay
in the code as **A/B controls / ablations**, but are **superseded** as the recommended path by the JointPosterior
route above. They are the naive/equal-weight PoE shape #3359 measured at/below majority; keep them only to
quantify how much the calibrated joint head buys. `DESIGN_sym_dual_judge.md` is updated to point here.

## References
- `DESIGN_sym_dual_judge.md` — the dual-judge finding, the confidence architecture (now corrected here), the
  sibling "division of labor," and the `c_dist`/`c_subcat`/`c_elem` measurements.
- `DESIGN_mu_sources_and_estimation.md` (#3357) — sources not independent; down-weight / residual / calibrated
  combiner; dual-vs-unified.
- `mu_posterior.py`, `REPORT_mu_posterior.md` (#3359) — `JointPosterior`, e5↔model +0.751, subcat↔elem
  anti-correlation, joint beats product-of-marginals.
- `DESIGN_inferred_operator_superposition.md` (#3356) — label-confidence tag + `P(μ|op)` calibration; §3
  four-source noise decomposition.
- `DESIGN_model_applications.md`, PR #3391 — margin (not level) as the selective-risk gate; margin weak
  per-query; AURC.
- `REPORT_eval_methodology.md` (#3387) — what the learned layer adds over e5 is a widening *margin*.
