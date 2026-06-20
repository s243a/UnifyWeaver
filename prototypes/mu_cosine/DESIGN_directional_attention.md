# Directional, multi-relational μ — role + operator attention (design)

Status: **design / not yet implemented**. The symmetric encoder (`train_cosine_mu_torch.py`, PR #3287)
is the baseline this extends. Implementation + training happens in an HF-egress session (frozen e5 +
a tiny learned head; minutes to train). WAM-Rust core untouched.

## Why this design

The merged encoder computes `μ(a,b) = cos(f(a), f(b))` with a single shared `f` — which is **inherently
symmetric** (`μ(a,b) = μ(b,a)`). But membership / subtopic is **order-dependent**: `μ(Optics | Physics)`
is high (Optics is a physics subtopic) while `μ(Physics | Optics)` is not. A single cosine cannot
represent that, no matter how good the input embeddings.

We also want **one** model to express several *relation sources* — symmetric relatedness, the Wikipedia
parent→child order, and LLM-judged membership under different prompts — rather than training a separate
model per source. The operator token below turns "which source of truth" from a *choice* into a
*conditioning input* (relation-conditioned scoring — the knowledge-graph-embedding trick of reading a
shared geometry through a relation-specific lens).

## Architecture

A tiny transformer over a short token sequence. With **≥3 tokens, attention is non-degenerate** — the
single-token degeneracy (softmax ≡ 1, Q/K dead) that ruled attention out for the symmetric encoder no
longer applies.

```
tokens = { operator(op), anchor(root), node(X)@gen0, {parents(X)}@gen1, {grandparents}@gen2, …, ⌀@gen≥k }
          (+ optional judge(j))

node(X)@gen0       = e5(X)      + gen_emb[0]          # the member candidate
ancestor@gen_d     = e5(anc)    + gen_emb[d]          # X's lineage, tagged by min-hop distance d
⌀@gen≥k            = absent_emb + gen_emb[d]          # learned "lineage ended / beyond depth" token
anchor(root)       = e5(root)   + role_emb[ANCHOR]    # the domain root — REQUIRED (see "Why keep the anchor")
operator(op)       = op_emb[op]                       # learned codebook row, no e5 content (an "instruction")
judge(j)           = judge_emb[j]                     # OPTIONAL, orthogonal axis — see below

μ(X | root; op) = sigmoid( W · pool( Attention(tokens) ) )   ∈ [0,1]
```

These are a **set of learned tags, not an ordered sequence** — with learned role/generation embeddings
the attention block is permutation-invariant, so "position" means *which learned tag a token carries*,
not its index. `gen_emb[d]` is a learned per-generation (per-hop) embedding; the graph is a **DAG**, so a
node can have several `gen1` parents — feed ancestors as a *set tagged by min-hop distance*, not a linear
chain. Depth `k` is an ablation knob (start at node + parent, i.e. `k=1`, then widen and measure).

- **e5 embeddings are frozen** (the `query:` / `passage:` asymmetry-friendly model — finally motivated
  here, where direction matters). Learned parameters are only: `role_emb` / `gen_emb` / `absent_emb` (a
  handful of rows), `op_emb` (a small codebook), the attention block (1–2 layers, a few heads), and the
  sigmoid readout.
- **"Learn the tags (operator / role / generation), not the nodes."** Every node is `frozen e5 + shared
  learned tags`, so **all 8,247 categories are covered (cold-start safe)**. A learned per-node *table*
  would only have rows for the ~324 nodes seen in training and would break the dense map — do not use one.
- **Why keep the `anchor(root)` token (don't go lineage-only).** μ computed purely from X's ancestry can
  only judge roots that *appear* in X's lineage — but the cases we most care about are exactly where the
  root is **not** an ancestor (`Music` vs `Physics`, the gate-leak at 0.52). A lineage-only model can't
  form that comparison. So the root is an explicit token, and the lineage is *context* for it.
- **The lineage is a soft cone.** Membership ≈ "root is an ancestor of X" ≈ the **downward closure** that
  `descendant_mu_mass` / `gated_ic` compute in the Rust core. Feeding X's ancestry lets the attention
  block learn that closure directly — "root sits near X's ancestor chain in e5 space ⇒ high μ" — while
  generalising past the graph's gaps/noise. This is the soft, embedding-smoothed version of the cone.
- **Asymmetry is structural**: the `anchor`/`node` tags differ, so swapping X and root changes the input
  and `μ(X|root) ≠ μ(root|X)`.
- **Sigmoid readout** gives `μ ∈ [0,1]` natively (cleaner than clamping a cosine; satisfies the Rust
  core's `μ ≥ 0`).

## Operators (the relation axis)

| op | meaning | label source | cost |
|---|---|---|---|
| `SYM` | symmetric relatedness / sameness | symmetric LLM labels (`mu_pairs_scored.tsv`); MiniLM as a weak teacher | done / cheap |
| `WIKI` | parent→child membership (order) | `data/benchmark/10k/category_parent.tsv` edges (margin loss) | **FREE, dense** |
| `LLM_<tmpl>` | directional membership under a prompt template ("is X a subtopic of Y", "is X used-in Y", …) | Haiku / Sonnet, boundary-focused | budget-capped |

## Judge axis — keep it ORTHOGONAL to the operator (recommended)

Do **not** fold "which model judged it" into the operator codebook. `relation ⊗ judge` is a *product*
space — you'll want "`WIKI` order *as Sonnet would score it*", which a single mixed axis can't express.
Add `judge(j)` as a separate optional token. **Start without it** (one implicit judge); add it only once
multi-judge labels exist (it then also lets you model judge bias / interpolate judges). Attention takes
an arbitrary number of tokens, so adding it later is free.

## Training

- **Multi-task over operators in one model**, with a per-operator readout sharing the attention backbone.
- **Balance the per-operator loss.** `WIKI` has ~millions of free edges; the `LLM_*` operators have
  hundreds of labels. Without inverse-frequency weighting / balanced batches, `WIKI` drowns the `LLM_*`
  operators (the same collapse mode as an unbalanced MoE gate). The free Wikipedia signal is a feature,
  but down-weight it to parity.
- **`WIKI` (free):** for an edge `child → parent`, margin loss `μ(child|parent) − μ(parent|child) ≥ m`,
  with in-batch uniform negatives (`gen_mu_pairs.py` already samples these). Zero LLM budget — this is
  the explicit wrong-order penalty.
- **`LLM_*`:** directional-μ MSE on the **cutoff band only**, budget-capped (the ~82k-Haiku-token/window
  rule; one inline subagent, items in the prompt, scores in the reply, no file I/O). The penalty here is
  *implicit* in the labels (the LLM's directional judgement).
- **`SYM`:** order-invariant MSE — feed both orders with the same target so the model learns symmetry
  under that operator.
- **Frozen e5 + AdamW weight decay** on the learned head (regularise toward identity) — the
  generalisation lever that made #3287 honest (without it the head memorises the few labels).

## Validation (per operator)

- `WIKI`: held-out edge order-accuracy (does `μ(child|parent) > μ(parent|child)`?); IC monotonicity /
  decision-flip on the gated cone.
- `LLM_*`: held-out directional-μ corr vs Haiku.
- `SYM`: matches relatedness; `μ(X|root) ≈ μ(root|X)`.
- Every operator's dense map → `check_feeds_rust.py` (100% coverage, IC general→specific, `lin ∈ [0,1]`,
  gate-leak count).
- **Control arm:** the MiniLM symmetric encoder (#3287) is the baseline the e5 multi-operator model must
  **beat on `SYM` and on gate-leak** — otherwise the extra machinery isn't earning its keep.

## Cost & coverage

Frozen e5 + tiny learned head ⇒ all 8,247 nodes covered, minutes to train on CPU. The only budget is the
bounded `LLM_*` boundary labels (the cutoff band, ~tens of k Haiku tokens, bought once and committed).

## Kickoff prompt (HF-egress session)

> In the UnifyWeaver repo (s243a/UnifyWeaver), implement the directional multi-relational μ model
> specified in `prototypes/mu_cosine/DESIGN_directional_attention.md`. Branch from `main`, open your OWN
> PR, don't touch the WAM-Rust core. GATING CHECK first (report + stop if it fails): `pip install -r
> prototypes/mu_cosine/requirements.txt` and confirm HuggingFace reaches `intfloat/e5-small-v2`.
> Then: (1) Add a `MuAttention` model — frozen e5 inputs + learned tags: `op_emb` codebook, `anchor`
> role, per-generation `gen_emb[0..k]` for the node + its ancestor **set** (min-hop distance on the
> `category_parent.tsv` DAG; start `k=1` = node+parent), and a learned `absent_emb` for masked/shallow
> generations — read by a 1–2 layer / few-head attention block (permutation-invariant; the tags carry
> position), with a sigmoid μ-readout. Keep e5 FROZEN; learn only the tags + attention + head (cold-start:
> every node = frozen e5 + shared tags). Keep the explicit `anchor(root)` token — lineage-only can't score
> out-of-domain roots. (2) Train
> multi-task over operators: `WIKI` from `category_parent.tsv` edges (free margin loss
> `μ(child|parent) − μ(parent|child) ≥ m`, in-batch uniform negatives) and `SYM` from
> `mu_pairs_scored.tsv` (order-invariant). Balance per-operator loss (WIKI down-weighted to parity).
> Frozen e5 + AdamW weight decay. (3) OPTIONAL, budget-gated, confirm first: add an `LLM_subtopic`
> operator from a small Haiku boundary set (cutoff band only, ≤82k Haiku tokens, one inline subagent).
> (4) Validate per operator: WIKI held-out edge order-accuracy, SYM relatedness + symmetry, each dense
> map through `check_feeds_rust.py`; report gate-leak vs the #3287 MiniLM-symmetric **control**. Do the
> judge axis later — start with one implicit judge. Report all per-operator numbers in the PR.
