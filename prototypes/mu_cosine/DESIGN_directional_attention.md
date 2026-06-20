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
tokens = [ member(X), anchor(root), operator(op) ]      (+ optional judge(j); + neighbour tokens later)

member(X)    = e5(X)    + role_emb[MEMBER]
anchor(root) = e5(root) + role_emb[ANCHOR]
operator(op) = op_emb[op]            # learned codebook row, no e5 content (a learned "instruction")
judge(j)     = judge_emb[j]          # OPTIONAL, orthogonal axis — see below

μ(X | root; op) = sigmoid( W · pool( Attention(tokens) ) )   ∈ [0,1]
```

- **e5 embeddings are frozen** (the `query:` / `passage:` asymmetry-friendly model — finally motivated
  here, where direction matters). Learned parameters are only: `role_emb` (3–4 rows), `op_emb` (a small
  codebook), the attention block (1–2 layers, a few heads), and the sigmoid readout.
- **"Learn the positions/operators, not the nodes."** Every node is `frozen e5 + shared learned tags`,
  so **all 8,247 categories are covered (cold-start safe)**. A learned per-node *table* would only have
  rows for the ~324 nodes seen in training and would break the dense map — do not use one.
- **Asymmetry is structural**: `role_emb[MEMBER] ≠ role_emb[ANCHOR]`, so swapping X and root changes the
  input and `μ(X|root) ≠ μ(root|X)`.
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
> Then: (1) Add a `MuAttention` model — frozen e5 inputs + learned `role_emb` (MEMBER/ANCHOR), a learned
> `op_emb` codebook, a 1–2 layer / few-head attention block, and a sigmoid μ-readout. Keep e5 FROZEN;
> learn only the tags + attention + head (cold-start: every node = frozen e5 + shared tags). (2) Train
> multi-task over operators: `WIKI` from `category_parent.tsv` edges (free margin loss
> `μ(child|parent) − μ(parent|child) ≥ m`, in-batch uniform negatives) and `SYM` from
> `mu_pairs_scored.tsv` (order-invariant). Balance per-operator loss (WIKI down-weighted to parity).
> Frozen e5 + AdamW weight decay. (3) OPTIONAL, budget-gated, confirm first: add an `LLM_subtopic`
> operator from a small Haiku boundary set (cutoff band only, ≤82k Haiku tokens, one inline subagent).
> (4) Validate per operator: WIKI held-out edge order-accuracy, SYM relatedness + symmetry, each dense
> map through `check_feeds_rust.py`; report gate-leak vs the #3287 MiniLM-symmetric **control**. Do the
> judge axis later — start with one implicit judge. Report all per-operator numbers in the PR.
