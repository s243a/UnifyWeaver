# Handoff to sol — math/statistics lane (owner-directed division of labor)

Owner's standing directive: fable does practical engineering (training), sol does the
mathematics and statistical analysis. This is the accumulated math-lane queue after shadow
rounds 1–7 (discussion thread: PR #4039; runners merged through #4046; phase-binding design
docs: PR #4047).

## Context you need (numbers, all shadow-tier, title-table v1-stems unless noted)

- Frozen e5 title-cosine floor on SM-FS: **0.5723** (v1), **0.5765** (v2-typofix).
  Title-table versions are pinned in `prototypes/mu_cosine/sm_fs_title_table_manifest.json`;
  every MRR must name its version from now on.
- Trained-trunk trendline across mechanism fixes: 0.280 → 0.347 → 0.385 → **0.440**
  (wiki-pretrained + Dirichlet trimix fine-tune; runners `sm_fs_trimix3_shadow.py`,
  `sm_fs_wiki_trunk_shadow.py`).
- Mechanisms found this week: σ(logit)·8 CE convention was a gradient bottleneck (fix:
  losses in logit space — owner's rule: "μ-space for targets and interfaces, logit space
  for losses and fusion"); graded-MSE channel is a calibration-drag stabilizer that must be
  present from step 0 (fold 2 collapses in every arm without it); anchored bilinear head
  died at wiki scale (30k decisions, collapses off floor at every config — e5 residual
  errors are per-node/contextual, not a global linear transform); trunk reaches 90% of its
  e5 floor at 27k wiki decisions vs 67% at 288 SM-FS decisions.

## Work items, in priority order

### S1. Reserve-spending protocol (the confirmatory gate)
The 5 shadow folds have now absorbed ~15 experimental arms; adaptive selection against
them is accumulating. The 1,481 reserved SM-FS rows remain untouched. Design the
confirmatory protocol for first contact with the reserve: trigger condition (proposal: the
first configuration whose shadow MRR CI excludes the current-version e5 floor), the exact
statistic and correction for the multiplicity already spent on the shadow folds, and what
gets preregistered. Keep it lightweight — owner rejected process-heavy gating (the Tier-2
pause stands); one page, one trigger, one test.

### S2. Scaling-curve estimand (crossover-n)
`sm_fs_scaling_curve_shadow.py` (epoch-matched) gives a flat-to-declining curve
(n∈{36,72,144,288}: 0.562/0.556/0.543/0.547 vs floor 0.573) for the anchored head; the
trunk's wiki numbers (67%→90% of floor from 288→27k decisions) are two points of a second
curve. Questions: (a) is epoch-matched or early-stopped the right design for a
bias-variance read; (b) can a crossover-n estimate (trained-beats-floor) be extracted or
bounded from these data; (c) what sample sizes would a third corpus (Pearltrees) need to
be informative?

### S3. Why does Dirichlet superposition stabilize?
Empirical facts: two-channel fixed mix stabilizes at 0.347; three fixed channels
destabilized (day 3); per-step λ ~ Dirichlet(.5,.5,.5) over (graded-MSE, rank-CE, e5-KL)
with loss Σλk·Lk and λ-blended operator token is BOTH the most stable recipe ever run AND
the best performer (0.385 scratch / 0.440 warm). Candidate framings: stochastic loss
weighting as implicit gradient-noise regularization; the graded channel as a trust-region;
the λ-blended conditioning as task augmentation. A short note separating these (what would
distinguish them experimentally) directly informs how we scale training. Note: at eval,
λ2/λ3 collapse to the same op row (both map to LINEAGE_RANK) — the eval surface has one
real axis until the e5-judge conditioning slot is added.

### S4. P4 coherence diagnostic (fusion/Kalman lane — gate-independent)
From `docs/design/PHASE_BINDING_*.md` (PR #4047, merged docs): when unit-phasor bundles
are superposed, the per-bin resultant length before renormalization is a phase-agreement
score — a free per-component confidence signal. Task: formalize its relationship to the
evidence-fusion stack (`THEORY_evidence_fusion.md`, `DESIGN_amortized_fusion_heads.md`) —
is resultant length a usable precision proxy for the Kalman heads' missing covariance, and
under what independence assumptions? This item's value does not depend on the phase-binding
P2 gate; it is pure fusion-lane math.

## Constraints
- Shadow-tier work: never score the 1,481 reserved rows outside the S1 protocol you design.
- Fitting remains double-locked; do not touch freezer/privacy/filing-cache files.
- All numbers you produce carry the `title_table_version` they were computed on.
- Engineering questions (runners, GPU, data generation) come back to fable; keep this lane
  on estimands, tests, and proofs.
