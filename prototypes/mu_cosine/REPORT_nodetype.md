# Node-type token — implemented, and an honest collinearity finding

Finishes the second axis of `DESIGN_calibrated_judges.md` §7: a factored **per-endpoint** node-type token
(`category` / `page` / `mindmap_node` / `pearltrees_collection`), orthogonal *in principle* to the
operator. Added to each endpoint token (anchor = root's type, node = candidate's type, ancestors =
`category`), zero-initialised so it's a no-op at warm start. Architecture diagram in `TECHNIQUES.md` §1.

## A/B — node-type ON vs OFF

Both warm-started from `model_mathfields.pt`, same cumulative data, 3 layers, 600 steps, `--lr 1.5e-4`
(only difference = the node-type token). OFF arm = `model_gap_all.pt`.

| metric | OFF (`model_gap_all`) | **ON (`model_nodetype`)** |
|---|---|---|
| ELEM held-out centrality corr | **+0.620** | +0.470 |
| overall held-out SYM corr | +0.855 | +0.849 |
| discrimination | 87% | 87% |
| page strata (e.g. `linear_filters` / `linear_programming` / `non-equilibrium`) | +0.70 / +0.77 / +0.55 | +0.53 / +0.40 / +0.13 |

**Node-type *hurt*** — ELEM and the page strata dropped broadly; discrimination unchanged.

## Why — collinearity with the operator

In today's data the operator and the node-type are **not orthogonal, they're collinear**:

- `ELEM` ⟺ page-membership (node is always a `page`, root always a `category`),
- `SYM`/`WIKI` ⟺ category–category.

So "this endpoint is a `page`" is **already implied by the operator being `ELEM`** — the node-type token
carries **zero marginal information** given the operator. All it adds is extra zero-init parameters that,
under-trained in 600 steps, perturb the shared representation → a small but real regression. The
design's orthogonality assumption (operator ⊥ node-type) is an assumption *about the data*, and the data
doesn't satisfy it yet.

## Decision + when it pays off

- **Gated behind `--use-nodetype`, default OFF.** The implementation is correct and banked; normal training
  is byte-for-byte the prior behavior (no tags ⇒ `nodetype_of=-1` ⇒ the token is never applied;
  `nodetype_emb` receives no gradient). `model_gap_all` (no node-type) remains the deployed model.
- **Activate it once the data has within-operator type diversity** — i.e. when the *same* operator sees
  *different* endpoint types. Concretely:
  - **pearltrees collections**: `ELEM` over `collection` nodes *and* `page` nodes — now node-type
    distinguishes collection-membership from page-membership *within* `ELEM`;
  - **mindmap nodes** (`mindmap_node`) under the human-judge corpus;
  - any `page` appearing as an endpoint in a `SYM`-like relation.

  Then operator and node-type decorrelate, and the token can carry real information. Until then it's
  premature.

This is a **data-precondition** result, not an implementation flaw — the cheap experiment (one A/B run, no
new Haiku) saved us from baking a redundant axis into the trained model, and tells us exactly what data
unlocks it: the mindmap / pearltrees-collection expansion.
