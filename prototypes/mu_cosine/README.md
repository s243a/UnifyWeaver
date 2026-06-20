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
| encoder architecture (MLP/MoE, not a transformer) | ✅ **forward pass** only — `mu_encoder.py`, runs at 384-dim |
| MiniLM init | ✅ **done** — `mu_encoder_torch.py` `build_minilm_init` + `embed()` (ML env w/ HF egress) |
| training the full encoder | ✅ **done** — `train_cosine_mu_torch.py` (torch port; objective + held-out generalisation) |
| **scoring the pairs (μ labels)** | ⬜ **not done** — the budget-spending step; `score_stub` in `gen_mu_pairs.py` (awaiting user OK) |
| wiring dense μ back to the Rust core | 🟡 **emitter done** — `emit_dense_mu.py` (clamped, verbatim names, 100% coverage); Rust consumption not yet run |

### Progress — ML-environment port (this branch)

The handoff's "take it into the real ML environment" steps are now implemented in torch (steps that
spend **no** LLM budget). Gating checks passed first: `pip install -r requirements.txt` succeeds and
HuggingFace is reachable (MiniLM downloads + encodes, 384-d).

| new file | what it does |
|---|---|
| `mu_encoder_torch.py` | torch port of the MLP/MoE encoder; **MiniLM init wired into `embed()`** (`build_minilm_init`); MoE load-balancing aux loss; `to_membership` clamp |
| `train_cosine_mu_torch.py` | the cosine-μ objective in torch (Adam / SparseAdam). `--free-vectors` reproduces `train_cosine_mu.py` (**corr → 1.00**); `--minilm --holdout` is the generalisation test; `--mode pairs` for scored `gen_mu_pairs.py` (refuses unscored rows) |
| `emit_dense_mu.py` | README step 5 — dense `name<TAB>μ` for the whole graph from the trained encoder; **clamps [0,1]**, **emits names verbatim**, asserts coverage |
| `validate_lin_agreement.py` | README step 4b — faithful python port of `gated_ic`/`lin_from_ic` (threshold 0.3, Σμ denom), compares graph-side Lin vs semantic cosine |

**Validation results (no LLM budget spent):**
- **Objective**: `--free-vectors` → train MSE 0.0000, corr **+1.00** (torch matches the stdlib proof).
- **Generalisation** (frozen MiniLM init + shared 1-layer encoder, 0.2 held-out): **held-out corr +0.946**,
  MSE 0.023 — μ predicted for categories whose label the encoder never saw. Even the *untrained*
  MiniLM-cosine already scores held-out corr +0.88.
- **Dense μ**: emitted for all 8 247 categories, 100% name resolution; highest-μ are exactly physics
  topics (Light, Electromagnetism, Thermodynamics, Acoustics…); 262 in the [0.3,0.7] boundary band.
- **Lin agreement (a caveat, honestly)**: the literal `μ(X|root) == Lin(X, root)` is **degenerate** —
  node-vs-root Lin saturates at 1.0 for all 51 scored nodes (the root is the most-general node).
  The *pairwise* graph-Lin vs semantic-cosine correlation is **weak** for the single-anchor-trained
  encoder (Pearson ≈ −0.13): single-anchor μ(X|Physics) training collapses physics nodes onto the
  Physics direction, so pairwise cos saturates. **This is the concrete motivation for the pairwise
  scored labels** (`gen_mu_pairs.py`, the budget step) — the held-out *single-anchor* μ regression is
  strong, but capturing pairwise/Lin structure needs the varied-anchor pairwise μ.

**Next (needs the budget step):** score `mu_pairs.tsv` (`gen_mu_pairs.py score_stub`, ~1200 pairs) →
`train_cosine_mu_torch.py --mode pairs --minilm` → re-run `validate_lin_agreement.py` (expect the
pairwise agreement to improve). **Confirm with the user before scoring.**

### Progress — dense μ *without* training, model comparison (prompt A variant)

Instead of spending budget, the fastest path to unblock the graph work (README prompt **A**): emit a
dense μ map by **direct asymmetric embedding** (encode the root as a *query*, each category as a
*document*, cosine, clamp) — and pick the embedder that minimises future Haiku re-scoring.

| new file | what it does |
|---|---|
| `dense_mu_direct.py` | dense `name<TAB>μ` with no training, asymmetric query/doc prefixes; presets `minilm` / `e5` / `nomic`; `--compare` reports each model's **decision band** (μ near the 0.3 gate) + fixture discrimination |
| `check_feeds_rust.py` | confirms a chosen map feeds `gated_ic` / `lin_from_ic` (faithful port): names resolve verbatim, μ∈[0,1], gated IC finite + general→specific, membership separates physics/non-physics |
| `dense_mu_e5.sample.tsv` | committed illustrative 20-row sample of the chosen map (full maps are git-ignored, regenerable) |

**Decision-band metric & the trap.** The *budget* metric is the **decision band** — categories with
μ ∈ [0.2, 0.45] straddling the 0.3 gate, the ones a later Haiku pass must re-score. But the **raw**
band is misleading: asymmetric retrieval models have a high cosine *floor*, so they pile everything
above the gate (e5 raw band = **0** — but Music=0.84 ≈ Optics=0.87, no discrimination at all). The fix
(no budget): **calibrate** each model's cosine→μ against the existing 90-node Haiku fixture (linear
fit), then the band is comparable and reflects genuine ambiguity:

| model | dim | raw band | **calibrated band** | fixture r | gate leak (non-phys passing 0.3) |
|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | 876 | 2028 | +0.573 | 4/5 |
| **`e5-small-v2`** | **384** | 0 | **653** | **+0.665** | **3/5** (Cooking, Religious correctly out) |
| `nomic-embed-text-v1.5` | 768 | 4109 | 1049 | +0.647 | — |

**Pick: `intfloat/e5-small-v2`** — smallest calibrated band, best fixture correlation, cleanest gate
separation, and 384-d (lightweight). `check_feeds_rust.py` confirms its map feeds the Rust core
(IC general→specific: Physics 2.85 < Optics 5.12 < Thermodynamics 7.18). Caveat: pairwise `lin_from_ic`
**saturates** toward 1.0 for many pairs on this graph (all maps) — the membership μ, not pairwise Lin,
is the clean separator. Don't use BERT/ModernBERT (logit/entropy encoders, not sentence embedders).

```bash
python3 dense_mu_direct.py --compare                              # the table above
python3 dense_mu_direct.py --model e5 --out dense_mu_e5.tsv       # emit the chosen (calibrated) map
python3 check_feeds_rust.py --mu-file dense_mu_e5.tsv             # sanity-check it feeds the Rust core
```

**Reproduce what exists (no deps):**
```bash
cd prototypes/mu_cosine
python3 train_cosine_mu.py        # learns cos≈μ on the 90-node fixture; prints corr → 1.00
python3 mu_encoder.py             # forward pass of the MLP/MoE encoder at d_model=384
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
3. Port `mu_encoder.py`'s forward to torch (it is a faithful spec) and add the training loop — the
   objective and gradient shape are validated by `train_cosine_mu.py`. Start with `n_layers=1`,
   `n_experts=1` (plain MLP / projection head); add MoE experts or neighbour context only if it
   underfits. Notes: use **Adam, not SGD** (an embedding table wants per-row adaptive steps, with
   sparse gradients — see batching); **do *not* re-apply `train_cosine_mu.py`'s `--dist-bias`** when
   training on `gen_mu_pairs.py` output — those pairs already encode the distance preference via the
   explicit negatives, so a distance bias on top double-penalises far pairs.
4. Validate: held-out `μ` regression, and that the dense μ it predicts *agrees with the graph-side*
   `lin_from_ic` (the two should converge — `μ(X|root) == Lin(X, root)`, the unification below).
5. Emit a dense `μ` map (`category → μ`) for the whole graph — **clamp cosine to `[0,1]`
   (`to_membership`) and emit names verbatim** (both load-bearing — see Integration) — and feed it to
   the Rust core.

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

**Resolved design decisions (and the reasoning, so they aren't re-litigated):**
- **MLP/MoE encoder, not multi-head attention.** A single category vector ⇒ attention softmax ≡ 1 ⇒
  Q/K dead, V/O collapse to one Linear; the region-dependent computation is the MLP nonlinearity
  (and an optional gated MoE for explicit region routing). So no attention in the base model.
- **Attention is the future upgrade, gated on a *sequence* input** — feed the category + a few
  neighbour tokens and *learn* the pooling weights (vs the cheap fixed mean-pool already in
  `encode(..., neighbors=...)`). Only worth it if the fixed-weight context underfits.
- **Start at `n_layers=1`, `n_experts=1`** (plain MLP / projection head): few params generalise from a
  small label set; a *shared* encoder on (mostly-frozen) MiniLM embeddings is what produces dense μ for
  *unlabelled* categories — big/per-category capacity overfits the labels and leaves the rest at init.
- **If you turn on MoE (`n_experts > 1`):** the prototype's *soft* gate (all experts, softmax blend) is
  interpretable but can **collapse** all gate weight onto one expert without a load-balancing auxiliary
  loss (`aux = n_experts · Σ_i f_i·P_i`, fraction-routed × mean-gate) — add it. *Top-k* (sparse) routing
  is the throughput upgrade for production; soft routing is fine while prototyping.

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

## Architecture (`mu_encoder.py`) — an MLP, not a transformer

We worked out (see the design discussion) that **multi-head attention is the wrong tool here**: each
category is a single MiniLM-pooled vector, and self-attention over one token has softmax ≡ 1 — its
query/key projections become dead weight and value/output collapse to one linear layer (a redundant
`Linear` in an attention costume). The region-dependent behaviour we want — different units firing in
different parts of semantic space — is the **MLP nonlinearity**, optionally made explicit and
interpretable by a **gated MoE** (route to a region expert). So the encoder is an MLP/MoE, not a
transformer.

| knob | value | rationale |
|---|---|---|
| `d_model` | 384 | MiniLM scale |
| `n_layers` | 1 (configurable) | start small — few params generalise from few labels |
| `n_experts` | 1 (→ plain MLP; >1 → gated MoE) | explicit, inspectable region routing |
| init | MiniLM per-category embedding by Wikipedia id | warm start; random fallback here |

`encode(id) = blocks(init_embedding[id])`; `μ(a|root) = cosine(encode(a), encode(root))`. Each block
is `x + FFN(LN(x))` (FFN = MLP, or a soft gated mixture of experts). **Neighbour context** is folded in
cheaply by mean-pooling the category with a few neighbour vectors (a fixed-weight "sum of vectors");
*learning* those pooling weights is exactly attention-over-neighbours — the documented future upgrade,
the only place attention earns its keep. `python3 mu_encoder.py` runs the forward (MLP and MoE).

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

So `μ` is faithfully expressible as the cosine of learned vectors. **Read this as *capacity*, not
*inductive bias*:** 91 vectors × 16 dims = 1,456 free parameters fitting 90 targets (~16× over-
parameterised), so `corr → 1.00` proves the representation *can* encode μ, nothing about
generalisation. The real test — predicting μ for *unscored* categories — needs the shared MLP encoder
on MiniLM init, and is the separate project.

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

**Two things that will silently corrupt the integration if missed (review-flagged):**
1. **Clamp the cosine to `[0,1]` before emitting** (`to_membership` in `mu_encoder.py`). Cosine is in
   `[-1,1]`, but the Rust mass functions assume `μ ≥ 0`: `descendant_mu_mass` and `sketch_mu_mass` *sum*
   μ as mass, so a negative weight corrupts the mass / KMV estimate; only `descendant_mu_mass_gated`
   tolerates it (a negative just fails the gate). Training targets `μ ∈ [0,1]` so the model is only
   *asked* for non-negative, but an unlabelled very-dissimilar pair can produce a negative cosine at
   inference — hence the clamp at emission.
2. **Category-name strings must match `category_parent.tsv` exactly** (case- and underscore-sensitive).
   The Rust loaders intern names → integer ids; a name absent from the graph hits `unwrap_or(0.0)` and
   silently becomes `μ=0`, **re-activating the density caveat this whole project exists to close.**
   Emit names verbatim from the TSV, and assert your coverage (how many emitted names resolved).

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

## Starting a new session (ML environment)

The pure-stdlib pieces run anywhere. To train the real encoder you need an environment with
`pip install -r requirements.txt` (torch, numpy, sentence-transformers) **and** HuggingFace egress
(for `all-MiniLM-L6-v2`). On Claude-Code-on-the-web that means an environment whose **network policy**
allows `huggingface.co` and whose **setup script** installs the requirements (an HF MCP server is
optional — it's an interface, not a bypass of the network policy). Large artifacts (full-graph
embeddings ~1.5 GB, weights) should be hosted externally and downloaded at runtime, **not** committed
to git — the cloud container is ephemeral, so they re-fetch per session. Cross-session coordination:
post progress on **PR #3280**.

Two ready-to-paste kickoff prompts:

**A — dense μ *without* training (fastest path to unblock the graph work):**
> In the UnifyWeaver repo, produce a dense μ map by running an embedding model directly — no training.
> For every category in `data/benchmark/10k/category_parent.tsv`, MiniLM-encode its name
> (`sentence-transformers/all-MiniLM-L6-v2`), cosine to the `Physics` anchor embedding, clamp to
> `[0,1]` (`prototypes/mu_cosine/mu_encoder.py:to_membership`), and emit `name<TAB>μ` in the format of
> `tests/fixtures/wikipedia_physics_fuzzy_nodes.tsv` (names verbatim from the TSV). Then sanity-check
> it feeds `gated_ic` / `lin_from_ic` in the Rust core. Deps: `pip install -r
> prototypes/mu_cosine/requirements.txt` + HF egress. Report coverage on PR #3280.

**B — train the encoder (the prototype's payoff):**
> Pick up the self-contained ML sub-project in the UnifyWeaver repo: read
> `prototypes/mu_cosine/README.md` (a complete handoff), branch
> `claude/mu-cosine-embedding-prototype`, discuss on PR #3280. First verify `pip install -r
> prototypes/mu_cosine/requirements.txt` succeeds and HuggingFace is reachable; if not, report exactly
> what's blocked and stop — don't fake it. Then follow the README's ordered steps: port
> `mu_encoder.py`'s forward to torch, wire MiniLM init into `embed()`, generate+score training pairs
> (`gen_mu_pairs.py` emits candidates; **scoring spends LLM budget — confirm with me before running**),
> train the cosine-μ objective (validated by `train_cosine_mu.py`), validate generalisation (held-out
> μ, and agreement with the graph-side `lin_from_ic`). Heed the integration guards: clamp cosine to
> `[0,1]` and emit names matching `category_parent.tsv` exactly. Keep changes on the prototype branch;
> do not touch the merged WAM-Rust core.

### Persistent storage (rclone + Dropbox)

The cloud container is **ephemeral** and the ~30 GB disk doesn't survive between sessions, so large
artifacts that outgrow git — MiniLM cache, model checkpoints, full-graph embeddings (~1.5 GB), scored
label sets — should live in external storage and be pulled per session. The agreed approach is
**`rclone` against a Dropbox *app-folder* app** (real VM egress, unlike a connector, which is
context-only). `cloud_setup.sh` is a credential-free template for the environment's setup-script field.

- **Security boundary is the Dropbox app, not rclone.** Create it as *Scoped access — App folder* so
  the token physically can't reach anything outside `/Apps/<YourApp>/`. A config path-prefix is **not**
  a restriction (any command could still type `dropbox:` and reach a full-Dropbox app's whole account).
  Scope permissions to `files.content.read` (+ `.write` only if you persist results).
- **Credentials**: set `DROPBOX_APP_KEY` / `DROPBOX_APP_SECRET` / `DROPBOX_REFRESH_TOKEN` in the
  env-vars field yourself (visible to env editors, no secrets store → app-folder scoping is the safety
  net). Generate the refresh token once via `rclone config` on your own machine. **An agent must not
  enter credentials for you.**
- **Workflow**: `rclone copy "dropbox:datasets" ./data` (down) / `rclone copy ./outputs
  "dropbox:outputs"` (up). `copy` never deletes — prefer it over `sync`. Pre-pull stable files in the
  setup script so they bake into the snapshot; fetch only changing data at runtime.
- **Parallel sessions** (prompts A and B at once) can share one app folder — coordinate writes by
  subfolder (`dropbox:dense-mu/` for A's output, `dropbox:checkpoints/` for B) and via PR #3280.
- *Alternative:* dedicated object storage (S3/GCS/R2) is arguably a better fit for machine-readable
  blobs (cleaner credential scoping, no token-refresh fragility); `rclone` supports those backends too,
  so the same workflow applies — just a different `[remote]` in `rclone.conf`.
