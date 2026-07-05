# Cross-judge DIRECTION superposition — graph-discrimination ⊕ LLM-asymmetric-operators

*Proposal (user, 2026-07-05). The natural extension of the judge-superposition + judge-independence results
(`DESIGN_sym_estimation_integration.md`, `REPORT_truncated_lambda.md`) to the **directional** axis. Design first,
build after review.*

## The idea

Superpose two estimators of the **same latent — direction** (which node of a pair is the broader / container):
- **graph-judged discrimination** — direction read from *structure* (`a` is an ancestor of `b` ⇒ `b` is the
  narrower one; `up_hops` asymmetry);
- **LLM-judged asymmetric operators** — direction read from *semantics* via the `element_of` / `subcategory`
  operators (`μ_ELEM(b|a)`, `μ_subcat(b|a)` high ⇒ `a` is the container/broader), scored by the LLM.

They estimate the **same direction** by different means, so — per the judge-independence result — a cross-judge
superposition should learn a **direction invariant** that is robust to *which judge is right*. "They should give
the same direction information" is a **testable invariant**.

## Why this is the RIGHT superposition (resolves the earlier design concern)

Judge-independence only transfers when the superposed judges estimate **the same latent**. `element ⊕ subcategory`
would superpose *different relations* (different latents) → the invariant is ill-defined. **Direction is a shared
latent** both the graph and the LLM estimate — element and subcategory are different relations, but their
*asymmetry points the same way*. So this is a cross-**judge** superposition of one latent, exactly the condition
where the truncated-λ / trunk-generality logic applies.

*(Distinct from the existing operator superposition — the 30% inferred-blend / random-operator-embedding over the
SET operators, `DESIGN_inferred_operator_superposition.md`. That mixes WHICH operator applies; this mixes WHICH
JUDGE estimates a fixed thing — direction. Keep them separate; don't conflate.)*

## The two direction estimators (both buildable from existing data — no new scoring)

- **`d_graph(a,b)`** — from the DAG: `dir(up_hops(a→b), up_hops(b→a))`, e.g. `3/(1+up_hops(b→a)) −
  3/(1+up_hops(a→b))` (positive ⇒ `a` broader). 0 when neither is an ancestor (lateral) — a design point (below).
- **`d_LLM(a,b)`** — from `wiki_rel_scored.tsv` (gpt-5.5-low): **`E_mu_fwd − E_mu_rev`**, the LLM's forward-minus-
  reverse expected membership = its directional call. Already scored for the 880 pairs. (Restrict to the pairs
  whose LLM top relation is directional — `element_of`/`subcategory`/`subtopic`/`super_category` — since `d_LLM`
  is only meaningful there.)

## The superposition + how the model consumes it

Target `d_blend = (1−λ)·d_graph ⊕ λ·d_LLM`. The model already has a **directional** readout path — the asymmetric
operators (HIER/ELEM) give `μ(a|b) ≠ μ(b|a)`, and there's a directional-ranking loss (`--dir-rank-weight`,
`L_dir`). So the cross-judge direction is a **target for the directional discrimination**: train so the model's
`μ(a|b) − μ(b|a)` matches `d_blend`, under a **cross-judge blend judge tag** (parallel to the SYM `blend` judge).

## The tests (mirror what worked for SYM)

1. **Do the judges agree?** `corr(d_graph, d_LLM)` on held-out — validates the premise ("same direction info").
   If low, the whole idea is questionable; if high, they're two views of one direction.
2. **Judge-independence of the learned direction:** train on `d_blend`, then predict a held-out direction target
   read **with** vs **without** the blend judge input — expect the trunk to carry the direction (agnostic ≈ blend),
   as SYM did.
3. **Truncated-λ transfer:** does varying λ push the direction into the trunk (smaller judge-input gap), as it did
   for SYM?

## Premise check — RESULT (go/no-go, 2026-07-05): GREEN, use rank/sign

`corr(d_graph, d_LLM)` + sign-agreement on the 880 scored pairs (no training):

| set | corr | sign-agreement |
|---|---|---|
| all 880 | +0.839\* | 99% |
| directional subset (419) | +0.455 | **100%** |
| graph-directional only (394) | +0.259 | **100%** |

**On DIRECTION (sign) the graph and LLM agree ~100% — the premise holds.** On *magnitude* the correlation is weak
(+0.26–0.46): they agree *which way*, not *how asymmetric* (hop-count vs semantic-confidence, different scales).
(\* the +0.839 is inflated by lateral pairs where both ≈0 — trivial agreement; the directional subset is honest.)

**⇒ Build on the sign/RANK, not the magnitude** (resolves decisions #3 and #5): the invariant the two judges share
is the *direction*, so target the directional **rank** (existing `L_dir`), which also sidesteps the magnitude-
scale mismatch. The magnitude blend is the genuine cross-judge divergence — a secondary, optional term, not the
foundation.

## Open design decisions (need resolving before building)

1. **Lateral / no-ancestor pairs.** `d_graph = 0` when neither node is an ancestor (siblings, distant). Are those
   excluded (train only on genuinely-directional pairs) or included as `d≈0` (both judges say "no direction")?
   Cleanest: **restrict to directionally-labelled pairs** for the target, keep laterals as `d≈0` negatives.
2. **`d_LLM` source: LLM score vs model's ELEM/HIER readout.** `E_mu_fwd−E_mu_rev` is the *LLM's* direction (no
   feedback loop) — preferred. Using the model's own ELEM/HIER asymmetry would be a feedback loop (avoid, or
   detach as we did for the membership readouts).
3. **Readout carrier.** Reuse the existing `L_dir` directional-ranking machinery (fit the *rank/sign*) vs a new
   signed-magnitude target. Rank is more robust; magnitude carries "how asymmetric."
4. **Judge tag.** A new `direction-blend` judge row, or reuse `blend`? A distinct row keeps its calibration clean.
5. **Scale alignment.** `d_graph` (hop-based) and `d_LLM` (μ-difference ∈[−1,1]) need a common scale before the
   convex blend (normalise both to a comparable range, or blend the *signs/ranks* rather than magnitudes).

## Build plan (after this doc is reviewed)
1. Compute `d_graph`, `d_LLM` on the 880 pairs; **report `corr(d_graph, d_LLM)`** (the go/no-go premise check).
2. Construct `d_blend`, emit directional graded rows tagged `direction-blend`.
3. Fine-tune (fixed-λ + truncated-λ) from `model_prod`; eval judge-independence of the learned direction on a
   held-out set (with vs without the judge input, multi-λ, multi-seed — the discipline we've settled on).

## References
- `DESIGN_sym_estimation_integration.md`, `REPORT_truncated_lambda.md` (judge-superposition + judge-independence).
- `DESIGN_inferred_operator_superposition.md` (the *operator* superposition — distinct; §170 dir-rank).
- `wiki_rel_scored.tsv` (`E_mu_fwd`/`E_mu_rev` = the LLM direction); `_up_hops` (graph direction).
