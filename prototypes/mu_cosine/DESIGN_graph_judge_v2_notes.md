# Graph judge v2 + diffusion follow-ups: design constraints from the #3865 review thread

Short capture (2026-07-18) of the reviewer guidance (gpt-5.6-sol) and standing decisions that must
shape the NEXT round, so they are not lost in the PR thread. Not an implementation plan.
Prospective evaluation of anything below is governed by the #3875 protocol.

## 1. Teacher vs feature (from the walk/μ ablation)

The blend ablation showed hit_prob/μ contribute ~nothing AS INFERENCE COLUMNS next to e5 + h_s.
That does not make them worthless: a signal can be redundant at inference yet valuable as
SUPERVISION (the walk teaches transitivity the trunk cannot read off titles). Keep the
teacher-vs-feature distinction explicit in any graph-judge-v2 design: dropping a family from the
ranker is not evidence against using it as a training target.

## 2. Directionality constraint on diffusion (reviewer, accepted)

The grounded resistor diffusion is a SYMMETRIC/reversible killed-walk object. It is NOT a drop-in
replacement for directional parent-walk targets (hit_prob's semantics). Graph judge v2 should be
either (a) a semantically gated DIRECTED killed walk (separate directed-Laplacian/reversible
construction — the design doc explicitly forbids feeding an asymmetric transition matrix into the
symmetric API), or (b) keep a separate direction channel and use symmetric diffusion only for
proximity/screening. Do not conflate the two.

## 3. "Tree-ness" measurement (reviewer, accepted)

Correlation between hit_prob and distance features is NOT a corpus-invariant tree-ness measure nor
a closure-edge budget. Prefer: structural parent multiplicity / path counts, plus cross-validated
CONDITIONAL R² — how much hit_prob explains after distance and ancestor features are known.

## 4. Graph-judge-v2 acceptance (reviewer, accepted)

Do not accept on a flatter Σ(hop) alone — lower variance can hide increased bias. Compare
hop-conditioned bias, NLL/calibration, and relational accuracy against independent labels,
alongside the residual covariance.

## 5. Entropy/generality-weighted conductance (hypothesis, not licensed)

Entropy/generality weighting of conductance (suppressing hub leakage) is a NEW symmetric-conductance
hypothesis: freeze a topology-only definition, keep a positive floor, recalibrate α, and evaluate
under the prospective protocol. Its benefit is not automatic.

## 6. Standing levers (post-#3865)

- Candidate generation (recall@50 = 0.680 caps every ranker) — including candidate-skeleton /
  Kron / semantic-closure ideas, as a separate post-#3875 design + label-free validation study.
- Bounded domains / full-graph sparse diffusion (#3867): the 2-hop universe is boundary-dominated
  (22% boundary nodes; e-fold saturates), so h_s has not had a fair test. The e-fold recipe needs
  either a larger universe or a boundary-aware calibration.
- Independent semantic modifier: revision-pinned Nomic (MiniLM sensitivity) instead of e5 driving
  both candidates and conductance.
