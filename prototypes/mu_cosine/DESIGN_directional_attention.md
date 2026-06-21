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
tokens = { operator(op), anchor(root), node(X)@gen0, {parents(X)}@gen1, {grandparents}@gen2, …, ⌀@gen≥k,
           provenance(corpus,judge) }

node(X)@gen0       = e5(X)      + gen_emb[0]          # the member candidate
ancestor@gen_d     = e5(anc)    + gen_emb[d]          # X's lineage, tagged by min-hop distance d
⌀@gen≥k            = noise + gen_emb[d]               # absent slot: OFF-MANIFOLD noise (no learnable content); gen_emb carries the depth
anchor(root)       = e5(root)   + role_emb[ANCHOR]    # the domain root — REQUIRED (see "Why keep the anchor")
operator(op)       = op_emb[op]                       # learned codebook row, no e5 content (an "instruction")
provenance         = corpus_emb[corpus] + judge_emb[judge] + prov_tag   # source of the label; MASKABLE → noise+prov_tag (agnostic). See "Provenance axis"

μ(X | root; op) = sigmoid( W · pool( Attention(tokens) ) )   ∈ [0,1]
```

These are a **set of learned tags, not an ordered sequence** — with learned role/generation embeddings
the attention block is permutation-invariant, so "position" means *which learned tag a token carries*,
not its index. `gen_emb[d]` is a learned per-generation (per-hop) embedding; the graph is a **DAG**, so a
node can have several `gen1` parents — feed ancestors as a *set tagged by min-hop distance*, not a linear
chain. Depth `k` is an ablation knob (start at node + parent, i.e. `k=1`, then widen and measure).

**Masking absent generations = off-manifold noise, not a learned token.** A shallow lineage (or anything
beyond depth `k`) fills its slot with **random noise** matched to the e5 magnitude, *plus* its `gen_emb`
tag. Why noise beats a learned `absent_emb`: a learned token is a fixed content vector the model can
*over-read* on a small label set; noise is off-manifold and **unreadable**, so the model learns to truly
*ignore* absent ancestry while the `gen_emb` tag still signals "lineage ended before gen-`d`" (the depth
cue is preserved without an overfittable vector). Use the *same* noise as a training regulariser —
randomly replace **present** ancestors with noise (noise-as-dropout) to stop over-reliance on a full
lineage and to help cold-start nodes with sparse real ancestry; this beats zero-dropout, whose zeroed
token is a readable, deterministic value. **Inference caveat:** random fill makes μ mildly stochastic, so
fix a per-node seed (or MC-average a few draws) to keep the dense map deterministic — the variance is
negligible once the model has learned to ignore noise slots.

- **e5 embeddings are frozen** (the `query:` / `passage:` asymmetry-friendly model — finally motivated
  here, where direction matters). Learned parameters are only: `role_emb` / `gen_emb` (a handful of rows;
  **no** learned absent token — absent slots are noise), `op_emb` (a small codebook), the attention block
  (1–2 layers, a few heads), and the sigmoid readout.
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

## Sampling & noise — starting defaults (all ablation knobs)

**Ancestor sampling.** Always include the **direct parents** (gen-1). For gen-2+, a **hub-down-weighted**
walk up — reuse `gen_mu_pairs.py`'s `1/deg^β` (β ≈ 1) — with per-step **stop ≈ 0.33**, capped at depth
~5 and ≤ ~8 ancestor tokens, deduped and tagged by **min-hop** (it's a DAG). The hub down-weighting is
essential: a plain uniform walk-up quickly hits the generic apex categories (`Main_topic_classifications`,
`Categories`, …) — the low-fan-in leak conduits found in the real-data measurement — which carry no
membership signal. Sample **stochastically per epoch** (free augmentation — the model sees different
ancestor subsets); use a **fixed seeded set at inference** for a reproducible dense map. Start at **k = 1**
(parents only, no walk) and widen only if gate-leak / held-out corr reward it.

**Noise-as-dropout rate.** During training, replace each *present* ancestor token with noise ≈ **0.2** of
the time, and noise out the **whole** lineage ≈ **0.1** of the time (cold-start guarantee — μ must still
work from `anchor + node` alone for nodes with sparse/missing ancestry). **No dropout at inference.**

**Noise magnitude.** L2-normalize the e5 embeddings, then fill a noise slot with a **random unit vector**
(Gaussian → normalize): same norm as a real token (≈ 1), random direction ⇒ off-manifold in high-D. Do
**not** scale it up (would dominate attention by magnitude) or down (a low-norm vector can be mistaken for
a real low-confidence embedding) — the "absent" signal must come from *direction*, not size.

## Provenance axis — keep it ORTHOGONAL to the operator (recommended)

Do **not** fold "which model judged it" into the operator codebook. `relation ⊗ judge` is a *product*
space — you'll want "`WIKI` order *as Sonnet would score it*", which a single mixed axis can't express.
Keep the source of a label on its own token, orthogonal to the relation. Attention takes an arbitrary
number of tokens, so adding it later is free.

**Generalize the "judge" into PROVENANCE (realized — PART B).** A label's source has *two* independent
parts, and both belong on this axis:

- **corpus** — which text the graph/embeddings came from (`simplewiki` now; `enwiki`, a domain dump,
  or a curated set later). Different corpora induce genuinely different μ.
- **judge** — who produced the label: a *bought LLM judgement* (`haiku` for the `SYM`/`LLM` targets) vs
  *the graph itself* (`graph` for `WIKI` edges, and for the free μ=0 `SYM` negatives = non-edges).

These form a `corpus ⊗ judge` product, but we don't want two more tokens diluting a 4–6-token set. So
represent provenance as **one token with a FACTORED embedding**

```
provenance = corpus_emb[corpus] + judge_emb[judge]   (+ prov_tag marking the slot)
```

— additive factors keep the product expressible (any corpus with any judge) while presenting as a single
input. `prov_tag` marks the *slot* in both states so the model can always locate "the provenance input".

**Make it MASKABLE, and mask by default.** Reuse the off-manifold-noise scheme of the absent-ancestor
slots: with probability `p_mask_prov` (training) — and *always* at the default inference path — replace
the factored embedding with a unit random vector (and drop the corpus/judge factors), leaving only
`prov_tag`. A masked provenance token = **provenance-agnostic μ**, i.e. *marginalize over sources* — the
sensible default when you just want "is X physics?" without committing to who said so. Revealing the
source (a 5-tuple `(node, root, op, corpus, judge)` at inference) conditions μ on that source; masking
(a 3-tuple) marginalizes it. So every existing call site stays a 3-tuple and automatically gets the
agnostic answer.

**Honest scope at first cut.** While the data is single-corpus (`simplewiki`) and the judge is
operator-correlated (Haiku⇄SYM/LLM, graph⇄WIKI), the token carries little *new* signal — it is near
constant. That's expected; validate it **structurally**, not by accuracy: (1) the slot + `corpus_emb`/
`judge_emb` exist and are wired, (2) masking flips the input and the model reads it, (3) an **ablation**
(`--prov-mask 1.0`, always-masked control) confirms the near-constant token does **not regress** the
Part-A results. The payoff is later: once `enwiki` labels (or a second judge) arrive, the same token
already carries the corpus⊗judge structure needed to model corpus shift / judge bias / source
interpolation — no architecture change, just new codebook entries.

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

**Primary metrics:** held-out directional/pairwise-μ corr, **gate-leak** (probe + large OOD sample),
decision-flip behaviour, and cold-start coverage. **Lin-agreement — revived by #3296, but compute it on
*node-gated* IC.** The control baseline (#3294) found pairwise `lin_from_ic` **96.7% saturated at 1.0**,
but that was the *path-gated* IC (`gated_ic`): μ-gating prunes ancestor cones, so gated IC is non-monotone
up the DAG (a common ancestor can out-IC the nodes below it ⇒ `2·IC(MICA)/(IC(u)+IC(v))` overshoots 1 and
clamps). **#3296's `gated_ic_node_filtered` (node-gated / downward-closed IC) fixes this** — on the same
fixture it drops saturation to **0.1%** with 431 distinct Lin values (`node_gated_ic.py`). So feed
`gated_ic_node_filtered` to `lin/resnik/faith_from_ic` and lin-agreement is a usable **secondary** signal
again (a taxonomic-structure check). Do **not** use the path-gated `gated_ic` for similarity (it saturates),
and do **not** validate against node-vs-root Lin (degenerate). Membership μ stays the **primary** in-domain
separator; node-gated lin-agreement is a complementary check, not the target.

- `WIKI`: held-out edge order-accuracy (does `μ(child|parent) > μ(parent|child)`?); decision-flip on the
  gated cone.
- `LLM_*`: held-out directional-μ corr vs Haiku.
- `SYM`: matches relatedness; `μ(X|root) ≈ μ(root|X)`.
- Every operator's dense map → `check_feeds_rust.py` (100% coverage, IC general→specific, `lin ∈ [0,1]`,
  gate-leak count). **Use `gated_ic_node_filtered`** when computing the Lin column.
- **Control arm:** the MiniLM symmetric encoder (#3287, numbers in `REPORT_control_baseline.md`) is the
  baseline the e5 multi-operator model must **beat on held-out μ corr, `SYM`, and gate-leak** (control:
  held-out +0.726, gate-leak 0/5 probe & 1.1% on 4280 OOD nodes) — otherwise the extra machinery isn't
  earning its keep.

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
> `category_parent.tsv` DAG; start `k=1` = node+parent), and **off-manifold random noise** (matched to e5
> magnitude, no learned token) for masked/shallow generations — read by a 1–2 layer / few-head attention
> block (permutation-invariant; the tags carry position), with a sigmoid μ-readout. Use the same noise as
> dropout (randomly replace present ancestors during training); fix a per-node seed at inference so the
> dense map is deterministic. Keep e5 FROZEN; learn only the tags + attention + head (cold-start: every
> node = frozen e5 + shared tags). Keep the explicit `anchor(root)` token — lineage-only can't score
> out-of-domain roots. (2) Train
> multi-task over operators: `WIKI` from `category_parent.tsv` edges (free margin loss
> `μ(child|parent) − μ(parent|child) ≥ m`, in-batch uniform negatives) and `SYM` from
> `mu_pairs_scored.tsv` (order-invariant). Balance per-operator loss (WIKI down-weighted to parity).
> Frozen e5 + AdamW weight decay. (3) OPTIONAL, budget-gated, confirm first: add an `LLM_subtopic`
> operator from a small Haiku boundary set (cutoff band only, ≤82k Haiku tokens, one inline subagent).
> (4) Validate per operator: WIKI held-out edge order-accuracy, SYM relatedness + symmetry, each dense
> map through `check_feeds_rust.py`; head-to-head vs the #3287 MiniLM-symmetric **control**
> (`REPORT_control_baseline.md`) on **held-out μ corr, SYM, and gate-leak** (the primary metrics).
> Lin-agreement is a usable *secondary* check **only when computed on `gated_ic_node_filtered`** (node-gated
> IC, #3296) — the path-gated `gated_ic` saturates it (96.7%); don't use that one. Do the
> judge axis later — start with one implicit judge. Report all per-operator numbers in the PR.

**Update (realized).** The judge axis is now built as the **provenance token** (`corpus ⊗ judge`,
factored + maskable; see "Provenance axis"). Masked = default = provenance-agnostic μ. While the data is
single-corpus it is validated structurally (slot/embeddings exist, masking works, `--prov-mask 1.0`
ablation shows no regression), ready to carry real signal once `enwiki`/a second judge arrives.
