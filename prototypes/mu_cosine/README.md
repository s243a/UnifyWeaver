# μ-cosine embedding prototype

A prototype for **generating dense fuzzy membership `μ` from learned category embeddings**, so the
μ-gated cone / similarity machinery (`descendant_mu_mass_gated`, `gated_ic`, `lin_from_ic` in the
WAM-Rust target) can run on domains where only a sparse set of categories has been LLM-scored.

This is a **separate project, prototyped on a branch** — it is Python/ML, not the Rust/Prolog core.

## Status & handoff (read this first)

**Branch:** `claude/mu-cosine-embedding-prototype` · **PR:** #3280 · last verified Python 3.11, stdlib only.

| piece | state |
|---|---|
| training objective (cosine ≈ μ) | ✅ **proven** on real data — `train_cosine_mu.py`, corr 1.00 |
| distance-biased sampler | ✅ implemented (in the trainer) |
| pairwise label *generator* | ✅ **done** — `gen_mu_pairs.py` (emits candidate pairs; no LLM cost) |
| transformer architecture | ✅ **forward pass** only — `mu_transformer.py`, runs at 384-dim |
| MiniLM init | ⬜ **not done** — random fallback (HuggingFace egress blocked in this env) |
| training the full encoder | ⬜ **not done** — needs numpy/torch (neither installed here) |
| **scoring the pairs (μ labels)** | ⬜ **not done** — the budget-spending step; `score_stub` in `gen_mu_pairs.py` |
| wiring dense μ back to the Rust core | ⬜ **not done** — see "Integration" below |

**Reproduce what exists (no deps):**
```bash
cd prototypes/mu_cosine
python3 train_cosine_mu.py        # learns cos≈μ on the 90-node fixture; prints corr → 1.00
python3 mu_transformer.py         # forward pass of the encoder at d_model=384
python3 gen_mu_pairs.py           # emits 1200 candidate pairs (200 pos / 1000 neg, 5:1) to mu_pairs.tsv
```
(`mu_pairs.sample.tsv` is a committed 30-line example of the generator's output.)

**To take it into the real ML environment (ordered):**
1. `pip install torch` (or numpy) — and obtain MiniLM (`sentence-transformers/all-MiniLM-L6-v2`, 384-d)
   for the per-category init embeddings. Wire it into `MuEncoder.embed` (currently random fallback).
2. **Score the candidate pairs.** `gen_mu_pairs.py` already emits them — a graded-word2vec-SGNS design
   (~5:1 negatives:positives; positives a hub-down-weighted random-walk *mesh* grown around the seed
   roots; negatives uniform noise). Fill the blank `μ` column with a Haiku subagent (`score_stub`
   shows the prompt/format; same discipline as `tests/fixtures/wikipedia_physics_*`). **Spends LLM
   budget — confirm with the user first.** Tune `--neg-ratio`, `--stop-prob`, `--restart-alpha`,
   `--seeds` and inspect the resulting μ histogram for boundary-band (0.3–0.7) coverage.
3. Port `mu_transformer.py`'s forward to torch (it is a faithful spec) and add the training loop — the
   objective and gradient shape are validated by `train_cosine_mu.py`.
4. Validate: held-out `μ` regression, and that the dense μ it predicts *agrees with the graph-side*
   `lin_from_ic` (the two should converge — `μ(X|root) == Lin(X, root)`, the unification below).
5. Emit a dense `μ` map (`category → μ`) for the whole graph and feed it to the Rust core.

**Compute & batching (so the GPU sizing is not guessed):**
- The sampler's **mesh size** (e.g. 324 nodes) is *coverage*, **not** a batch size — it sets label
  diversity, nothing about training memory. The training **batch** is a count of *pairs* per step, an
  independent knob.
- This prototype model is **not memory-bound**: 1 block (~1.77M params ≈ 7 MB) + the 8.2k-cat
  embedding table (~13 MB) + Adam (~25 MB); activations scale at **~24 KB/pair**. The whole 1.2k-pair
  set is ~30 MB — you can full-batch it. Choose batch (256–1024) for SGD noise vs. stability, not
  memory. Bigger batches also give more **in-batch negatives** (SGNS), so prefer larger on a big GPU.
- Activations/pair scale with `layers × seq_len × 2`, so a **500 MB activation budget** holds
  ~20k pairs for this 1-layer single-token model, ~3k for a 6-layer single-token one, ~200 for a
  6-layer name-token-sequence one — all comfortable batches. Don't assume the prototype's tiny
  footprint if you deepen the model or feed token sequences.
- The real memory cost is the **embedding table at full Wikipedia scale**: ~1M cats × 384 × 4B ≈
  1.5 GB (+~3 GB dense Adam). Use **sparse embedding gradients** (only the rows a batch touches) — the
  lever is the table, not the pair-batch (still 1k–4k).

**Open decisions a follow-up should know:**
- `n_heads = 16` (not the discussed ~20): 384 is not divisible by 20; its divisors near there are 16
  (24 dims/head) and 24 (16 dims/head). 16 is closest to the target head count — revisit if 24 (more
  literal "≈19 components/head") is preferred. Configurable in `Config`.
- Input is a **single per-category vector** (so attention degenerates to a linear map). If you want
  non-trivial attention, feed a *sequence* (the category's name tokens, or category + graph
  neighbours); the head structure is already in place for that.

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

**Exact pointers** (so a follow-up doesn't have to hunt):
- Consumers of `μ` live in `templates/targets/rust_wam/boundary_cache.rs.mustache`:
  `descendant_mu_mass` / `descendant_mu_mass_gated` (cone mass, gated), `gated_ic` (per-node IC),
  `resnik_from_ic` / `lin_from_ic` / `faith_from_ic` (similarity). They take `μ` as
  `HashMap<u32, f64>` keyed by the internal node id.
- The category graph (this prototype loads it): `data/benchmark/10k/category_parent.tsv`
  (`child<TAB>parent`). The sparse μ fixture: `tests/fixtures/wikipedia_physics_fuzzy_nodes.tsv`
  (`name<TAB>μ`). Candidate surfacing: `scripts/physics_random_walk_candidates.py`.
- Format the dense μ the same way (`name<TAB>μ`, '#'-comment header) so the existing Rust loaders
  (e.g. `wikipedia_gated_similarity_tracks_physics_relatedness`) consume it unchanged — just larger.
