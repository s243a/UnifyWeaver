# μ-cosine embedding prototype

A prototype for **generating dense fuzzy membership `μ` from learned category embeddings**, so the
μ-gated cone / similarity machinery (`descendant_mu_mass_gated`, `gated_ic`, `lin_from_ic` in the
WAM-Rust target) can run on domains where only a sparse set of categories has been LLM-scored.

This is a **separate project, prototyped on a branch** — it is Python/ML, not the Rust/Prolog core.

## The idea

`μ(X | root)` — how much category `X` belongs to a domain anchored by `root` (e.g. `Physics`) — is a
**relational** quantity: the same category has different `μ` under different domains. So it is not a
single per-node scalar but the **cosine between two learned vectors**, one for `X` and one for the
`root`:

```
μ(X | root) ≈ cos( encode(X), encode(root) )
```

Learn the encoder so that cosine matches the LLM-provided `μ`. Because the root is just another encoded
category, the *same* node embeddings serve any domain — swap the root vector, swap the domain. This
also unifies `μ` with the graph-similarity work: `μ(X | root)` is exactly `Lin(X, root)` measured in
semantic (vector) space instead of the category graph.

## Architecture (`mu_transformer.py`)

A small transformer encoder, with the design sizing from the discussion:

| knob | value | rationale |
|---|---|---|
| `d_model` | 384 | MiniLM / nanoGPT scale |
| `n_heads` | 16 | target ≈ 20 (`d_model / log₂(|categories|)` ≈ 384/19), but 384's divisors near 20 are 16 and 24; 16 is closest |
| `n_layers` | 1 (configurable) | "a lot with one layer" — the input is a single per-category vector, so attention is light |
| init | MiniLM per-category embedding, looked up by Wikipedia id | warm start; random fallback here |

`encode(id) = blocks(init_embedding[id])`; `μ = cosine(encode(a), encode(b))`. With a single-token
input the self-attention softmax is over one position (≡ 1), so MHA reduces to a linear map — the head
structure is kept anyway so the same code accepts a multi-token input later (name tokens, or the
category + its graph neighbours), where attention becomes non-trivial. `python3 mu_transformer.py`
runs the forward pass at full design scale.

## Training objective — proven runnable (`train_cosine_mu.py`)

`python3 train_cosine_mu.py` learns per-category vectors directly (the simplest "encoder") to isolate
and **prove the objective on the real Haiku-scored fixture**: minimise `(cos(v_X, v_root) − μ_X)²` with
the **distance-biased sampling** from the design (categories closer to the anchor in the Wikipedia
graph are sampled more often). Pure stdlib, closed-form cosine gradient (finite-difference checked).

Result on `tests/fixtures/wikipedia_physics_fuzzy_nodes.tsv` (90 nodes, anchor `Physics`):

```
initial  MSE 0.41   corr(cos, μ) +0.05
final    MSE 0.0002 corr(cos, μ) +1.00
  μ 0.40  cos +0.40  Arsenic_compounds
  μ 0.90  cos +0.90  Atoms
  μ 1.00  cos +0.96  Sound
```

So `μ` is faithfully expressible as the cosine of learned vectors. (With 16 dims for 90 nodes this is
over-parameterised and fits exactly — that is fine as a proof of the *objective*; the real test is
*generalisation* to unscored categories via the transformer encoder, which is the separate project.)

## What this prototype does NOT do (the separate project)

This environment has **no numpy / torch / network**, so:

- **MiniLM initialisation** needs the model weights (HuggingFace egress blocked here) — the encoder
  falls back to random init.
- **Training the full transformer** wants numpy/torch for speed (the pure-Python forward is for
  inspecting the architecture, not gradient descent at 384 dims).
- **Generalisation** needs more `μ` labels than the 90 single-anchor scores — cheap to generate with
  the existing Haiku pipeline (`scripts/physics_random_walk_candidates.py` + scoring), ideally
  **pairwise** `μ(a, b)` for varied anchors, sampled with the distance bias.

## Integration with the WAM-Rust core

Once the encoder produces dense `μ` for *all* categories (not just the scored 90), it removes the
density caveat on `descendant_mu_mass_gated` (gating no longer prunes through unscored connectors) and
feeds `gated_ic` / `lin_from_ic` directly — the same `μ` map, just dense and domain-swappable.
