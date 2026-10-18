# PREREGISTRATION DRAFT — SM-FS reserve spend (v1, for coordinator seal + sol adversarial review)

Status: DRAFT. The reserve is untouched and remains untouched until this document is
sealed by the coordinator (sol adversarial review first if quota permits). One-shot:
the reserve is spent by being measured. No re-runs, no extensions, no post-hoc arms.
Results are recorded regardless of outcome.

## 1. What is being confirmed

The shadow-tier SM-FS conclusions, on rows no experiment has ever touched:
  (C1) Frozen e5 title-cosine is the standing filing champion on SM-FS
       (shadow: 0.5723 on title-table v1-stems; 0.5765 on v2-typofix).
  (C2) The best trained configuration trails e5 but far exceeds the practical floor
       (shadow: 0.480, trimix fine-tune warm-started from the wiki replay trunk).
  (C3) The residual champion-vs-trained gap is corpus-surface (terse leaves), per the
       transfer-null triangulation — tested via the expansion-table arm (C4).
  (C4) CONDITIONAL ARM — only if the owner-reviewed abbreviation-expansion table exists
       at seal time: e5 on expansion-corrected titles. Shadow bracket [0.53, 0.69]
       (lower = leave-one-out lower bound; upper = owner-sign-off-legitimized).
       If the table is not reviewed by seal time, this arm is DROPPED at seal — it
       cannot be added later.

## 2. Declared artifacts (frozen at seal)

  - Champion:      frozen e5, title-table version DECLARED AT SEAL (v1-stems for direct
                   comparability with all shadow numbers, v2-typofix reported alongside).
  - Challenger:    trimix3 fine-tune (graded corner), warm-start
                   ~/mu_data/wiki_lineage_v2/trunk_wiki_40000.pt, trained on ALL FIVE
                   shadow folds (the confirmatory analog of the 0.480 config; exact
                   hyperparameters = sm_fs_trimix3_shadow.py at the sealing commit,
                   seed 3997001, single seed — declared, not tuned).
  - The challenger checkpoint is TRAINED BEFORE the reserve is opened and its sha256
    recorded in the seal. Nothing about it may change afterward.

## 3. Exactly what is measured (and nothing else)

On the 1,481 reserved rows only:
  - full-catalog MRR and recall@{1,5} for: champion (v1 titles), champion (v2 titles),
    challenger, and C4 if armed. Ranks via the verified pipeline ranker
    (pl.recompute_rank). No other quantity, stratification, or model is evaluated.
  - One bootstrap CI (n=2000 resamples, seed 3997999) per pairwise contrast:
    champion-vs-challenger, and C4-vs-champion if armed.

## 4. Success / failure criteria (declared)

  - C1 confirmed if champion MRR on the reserve is within ±0.05 of its shadow value
    (0.5723 v1); otherwise the shadow methodology is flagged for audit.
  - C2 confirmed if challenger MRR >= 0.43 (shadow 0.480 minus a 0.05 generalization
    allowance); challenger BEATS champion only if the CI excludes zero.
  - C4 (if armed) confirms the surface verdict if expansion-corrected e5 minus plain
    champion has CI excluding zero in the positive direction.

## 5. Outcome commitments

  - All outcomes: numbers recorded in the repo verbatim, including failures.
  - If C1 fails: shadow-eval audit before any further SM-FS work.
  - If C2 fails: the trained-stack program on SM-FS is demoted to research-only;
    deployment guidance is e5(+expansion if C4 held).
  - If C4 held: expansion table graduates to title-table v3 in the manifest and
    becomes the deployed champion configuration.
  - No outcome authorizes a second reserve measurement.

## 6. Roles

  - Training lane (this document's author): executes exactly §3 after seal; publishes.
  - Coordinator: seals; verifies artifact hashes at seal and at execution.
  - sol: adversarial review pre-seal (attack surface: criteria gaming, champion
    definition, bracket validity, one-shot enforcement).
