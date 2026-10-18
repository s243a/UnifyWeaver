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

## The tests (mirror what worked for SYM)  — [SUPERSEDED: see "Purpose & the TRUE test" below]
> **Note:** this eval plan (in-distribution held set) was **superseded** by the novel-node TRUE test — read
> "Purpose & the TRUE test" before implementing anything from this section.

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

## Three-way finding (2026-07-05): direction is CONSENSUS on Wikipedia — magnitude is where they differ

Extended to **three** estimators (user's idea) — graph-discrimination, LLM-element, LLM-subcategory — the
sign-agreement is **100% pairwise, zero disagreements** (incl. element-vs-subcategory), on the 880 pairs. They
differ only in **magnitude**: corr(graph, subcat) +0.81, (graph, element) +0.44, (element, subcat) +0.28.

**Implication:** Wikipedia categories are a *clean taxonomy* — direction, when it exists, is unambiguous
(`model_prod` already gets 99.9% edge-order accuracy). So the "same direction" premise is confirmed but *trivially*
(nothing to resolve). Two consequences:
- A 3-operator random superposition here tests the **judge/operator-independence MECHANISM** (does varying the mix
  push a robust direction into the trunk — the truncated-λ transfer) and blends the **magnitude** differences
  (element vs subcat are genuinely different, +0.28). Valid, but it is a *mechanism* test, not a
  *disagreement-resolution* test.
- The **disagreement-resolution motivation needs direction-AMBIGUOUS data** — looser hierarchies (Pearltrees,
  pages, mindmaps) where the graph and LLM would actually flip. Wikipedia can't provide it.

**Decision pending (user):** (A) build the 3-operator superposition on Wikipedia = a clean mechanism/transfer
test on a consensus direction; or (B) source direction-ambiguous data first, so the superposition resolves real
disagreements. (Not mutually exclusive — A is cheap and reuses everything; B is the harder, more novel test.)

## Purpose & the TRUE test (user, 2026-07-05) — supersedes the framing above

- **Superposition = linear SUM, not product.** `d_blend = Σ wᵢ·dᵢ`. A *product* of operators is a *different
  (composite) operator*, not a superposition. Linearity is deliberate.
- **The purpose is to teach the model to SEPARATE the operator inputs; linearity of the TARGET makes it learnable
  — the NETWORK is not linear.** The mixing target is a linear sum, but the model learning it is a non-linear
  network that uses that capacity to represent each operator's contribution *separately* (so it can reconstruct
  *any* linear combination) — a source-separation / unmixing objective. This is *why* judge-independence emerges
  (the trunk holds the separable components); a *product* target wouldn't decompose this way.
- **The TRUE test: predict direction where NO operator has seen the nodes.** The eval above scored pairs the
  operators *cover* (graph knows them, LLM scored them) — not a real generalisation test. The decisive test is
  **novel-node pairs** (not in the graph, not LLM-scored in training) where all operators are *silent* — so the
  model must infer direction from **frozen e5 + the separated direction concept**, checked against an
  **independent** direction ground-truth (an eval-time LLM/human on those novel pairs). If the separation
  worked, the model predicts direction there; if it only memorised operator outputs, it fails.

### Build (the true test)
1. Sample **novel** category pairs — nodes *outside* the training graph (not in 100k_cats) and unscored in
   training — with a genuine direction. (`d_graph≈0`, and the ELEM/HIER readouts were never trained on them ⇒
   operators silent; the model has only e5.)
2. **LLM-score them at eval** for the direction ground-truth (`E_mu_fwd−E_mu_rev`) — independent of training.
3. `corr(model HIER-asymmetry, LLM direction)` for the dir-blend-trained model vs `model_prod` — does the
   superposition-trained model generalise direction to unseen nodes better? (multi-seed; agnostic vs dir-blend.)

## Open design decisions (need resolving before building) — all RESOLVED in the build

1. **Lateral / no-ancestor pairs.** — **RESOLVED:** kept as `d≈0` **negatives** (direction-requires-agreement).
2. **`d_LLM` source.** — **RESOLVED:** LLM score `E_mu_fwd−E_mu_rev` (no feedback loop), not the model readout.
3. **Readout carrier.** — **RESOLVED:** built on **rank/sign** (the premise check showed sign is the shared
   invariant, magnitude diverges); emitted as HIER fwd/rev asymmetry rows.
4. **Judge tag.** — **RESOLVED:** distinct **`dir-blend`** row (not reusing `blend`).
5. **Scale alignment.** — **RESOLVED by #3:** blend on sign/rank sidesteps the hop-vs-μ scale mismatch.

## Build plan (after this doc is reviewed)
1. Compute `d_graph`, `d_LLM` on the 880 pairs; **report `corr(d_graph, d_LLM)`** (the go/no-go premise check).
2. Construct `d_blend`, emit directional graded rows tagged `direction-blend`.
3. Fine-tune (fixed-λ + truncated-λ) from `model_prod`; eval judge-independence of the learned direction on a
   held-out set (with vs without the judge input, multi-λ, multi-seed — the discipline we've settled on).

## References
- `DESIGN_sym_estimation_integration.md`, `REPORT_truncated_lambda.md` (judge-superposition + judge-independence).
- `DESIGN_inferred_operator_superposition.md` (the *operator* superposition — distinct; §170 dir-rank).
- `wiki_rel_scored.tsv` (`E_mu_fwd`/`E_mu_rev` = the LLM direction); `_up_hops` (graph direction).
