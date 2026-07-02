# PATH operator — multi-path ancestor membership over the DAG (design / future-work)

## Empirical conclusion (2026-07-01, 3-seed CI) — multipath does NOT beat single-path on Pearltrees

Built the deterministic shallow merged-multipath passage (`merged_ancestors.py`, `train_lineage --merged --dag`)
and tested it on the RDF+API **assembled** Pearltrees DAG (which recovers 396/880 multi-parent filing folders — the
API-only 20 was an artifact of the RDF-export truncation bug, see [[project_pearltrees_rdf_export_bug]]).

**Result: multipath is not a win on Pearltrees.** A 3-seed CI (seeds 7/13/42) of `mu-lineage` PATH(merged) vs
single-path LINEAGE:
- *Full eval:* mean deltas positive but **within noise** (stdev ≥ mean; seed 42 reverses). Looked promising at
  seed 7 alone — walked back by multi-seed.
- *Multi-parent subset* (where the passage actually differs — 82% of queries, dilution was minor): the
  branch-recovery metrics are **negative** — `ov|miss` mean −0.027, `depth|miss` mean −0.123. The full-eval
  positivity was **training-side variance**, not the multipath passage doing work.

**Decision — corpus-driven operator choice:**
- **Pearltrees → single-path `LINEAGE`.** It has a *principal* parent (the RefPearl "inside" precedence = the actual
  human filing decision — real signal), and multipath gives no measurable benefit over it. Simpler and cheaper.
- **Wikipedia → multipath `PATH`.** Categories have *no* principal parent (every parent link equal), so single-path
  is an arbitrary pick — multipath is the only meaningful ancestor representation. NB: this is "what Wikipedia
  affords," **not** proven-superior — a proper Wikipedia branch-recovery eval (avoiding the mis-framed
  context-on-candidate setup in `eval_path_wiki.py`) is still open if we need it.

The machinery below is built and correct; it's the right tool for the multi-parent Wikipedia DAG, not for Pearltrees.
The rest of this doc is the (still-unbuilt) full design.

---

Design note (nothing built yet). Follows the finding that single-path LINEAGE is redundant to HIER, but a
**multi-path** ancestor operator is *not* — because a DAG node has many ancestor chains, and the harvested
single-path collapse (`parent_map` precedence) threw all but one away.

## 1. Motivation

`LINEAGE` (as built) scored `μ(node | ONE materialized ancestor chain)` — the jsonL picks a single parent per node
(RefPearl "inside" over AliasPearl "cross-link"), collapsing the **DAG** (multi-parent: Pearltrees aliases, Wikipedia
categories) to one arbitrary path. A **PATH** operator instead scores the node against a **superposition over all
valid ancestor chains** — the full set of hierarchical contexts it belongs to ("this AI paper is under both
`CS ▸ ML` and `Research ▸ ToRead`"). Keep LINEAGE (the canonical chain); add PATH (the multi-path superposition).

## 2. Unification: one operator, a METHOD parameter (a higher-order operator)

Ancestor-path membership is a *single relation*; LINEAGE and PATH differ only in **how the paths are aggregated**.
So factor it: **operator = ancestor-path membership (what); method = aggregation/sampling strategy (how).**
`operator × method` is a **higher-order operator** — the operator takes a *strategy* as a parameter:

| name today | = |
|---|---|
| `LINEAGE` | `ancestor-path(method = canonical)` — the single primary chain |
| `PATH` | `ancestor-path(method = random-walk)` — uniform-parent superposition |
| (future) | `ancestor-path(method = edge-weighted / PPR)` — richer variants |

This unifies lineage/path into one operator and generalizes the pattern (any operator with multiple computation
strategies can be method-parameterized).

*Clarification (per review) — LINEAGE does NOT stay a separate codebook row.* The redundant experiment grew a
distinct index-4 `LINEAGE` op; going forward that is **retired**. "Keep LINEAGE" means *keep the canonical-chain
behaviour as a **method** (`method=canonical`)* of the single `ancestor-path` operator — a method selection at
data-gen / inference time, **not** a new `op_emb` slot. So **`n_ops` does not grow**: `ancestor-path` is one operator
(encoded via the `MLP(spec)`/attention-lookup once that exists; a single row meanwhile), and `canonical` vs
`random-walk` vs `PPR` are its methods, not codebook entries.

## 3. "Random superposition" = sampling (cheap; reuses the variant cache)

Realize the superposition **stochastically**: each training step, **sample ONE ancestor path** from the method's
distribution, embed it, apply wildcard masking (id-dropout + prefix-dropout), and use it as the passage. The model
learns the *expectation over paths* across steps — no need to embed every path and weight-average explicitly. The
wildcards are an orthogonal masking augmentation on top of the path sampling.

*Gradient-variance note (per review):* the 1-sample estimator has variance set by the *spread* of a node's ancestor
chains — high-branching DAG nodes with divergent parents give dissimilar samples ⇒ noisy gradients. Two cheap
mitigations (both reuse the pre-built node cache): **sample k paths per step and average**, and/or **warm up with the
edge-weighted method** (lower-variance, canonical-path-dominated) before annealing toward uniform.

## 3b. Passage representation — per-node tokens, NOT a fixed whole-path embedding

Once paths are *sampled* (random walk × stop-β × multi-path × wildcards), the space is **combinatorial** — you cannot
precompute a fixed e5 embedding for every sampled path (the variant-cache trick that worked for LINEAGE's ~6
variants/folder breaks entirely). **Resolution: stop embedding the whole path as one e5 string.** Represent the path
as a **set of per-node embeddings** — embed each ancestor node's *title* **once** (cached, linear in #nodes) — and
let the model **attend / aggregate over the sampled subset**. Then all the sampling (which nodes, how deep, which of
the DAG's paths, wildcard-masked) is just **selecting which cached node-tokens enter the passage**:
- No re-embedding, ever — the combinatorial path space is handled by *token selection over a linear node cache*
  (this also kills the re-embed-per-step compute concern that dogged the whole-path-string approach).
- The μ-attention block already supports a **multi-token passage**, so this is a representation change, not an
  architecture change.
- The materialized-id line / wildcards become per-node attributes rather than substrings of one blob.

*Three implementation points the review flagged (resolve before building):*
- **Ordered set → depth positional encoding.** The passage is a *set* of node tokens, but ancestor paths are
  **ordered** (child→parent→grandparent). Attention over an unordered set would discard depth — which is exactly the
  graded-depth signal we want. So **add a depth-position embedding** to each node token (0 = the node itself, 1 =
  direct parent, …) before aggregation, so "direct parent" vs "grandparent" is distinguishable. (This is the natural
  home for the graded-depth membership target.) *Max depth:* stop-β/multi-path give variable, possibly-deep paths, so
  fix a **max-depth slot count with truncation** (simplest — deepest ancestors beyond it are dropped, acceptable as
  branch signal is shallow-heavy), or use **relative** depth PEs if arbitrary depth must be exact.
  **EMPIRICAL (dense-DAG calibration, wide_enwiki_math):** on Wikipedia (dense multi-parent) the *deterministic*
  merged closure **explodes** — `max_depth 12–15` from any node reaches all of science/humanities via structural
  connectors (even after admin-filtering) — an incoherent grab-bag. At **`max_depth 2`** it's small and coherent
  (`Abstract algebra` → `Algebra` + `Fields of mathematics`; `People` → `Humans` + `Anthropology`). So the merged
  list must be a **shallow LOCAL context (parents + grandparents), NOT the full closure** — the branch signal lives
  there anyway. `max_depth 15` only worked for *shallow Pearltrees*; the stochastic stop-β must likewise be
  **shallow-mean** on dense DAGs (not the mean-10 that suited Pearltrees). Plus admin-filter the few structural
  connectors (`Container categories`, `Categories requiring diffusion`) that appear as grandparents.
- **Padding masking.** Variable-length paths (from stop-β / multi-path) are padded to a fixed slot count; the
  attention block must **mask the padding tokens** so they don't pollute the aggregate. Standard, but must be wired.
- **Cache invalidation.** The per-node embedding cache assumes static titles. On a graph refresh (Pearltrees/Wikipedia
  snapshot), do **incremental update** — re-embed only new/changed nodes — rather than a full rebuild.

So the path operator factors on **three** axes: **operator** (ancestor-path membership) × **method** (sampling incl.
stop-β) × **representation** (per-node tokens, attention-aggregated). The whole-path-string embedding was scaffolding
that doesn't scale; per-node composition is the structure that does — and it's what the decoder's per-node /
optimizer view wanted anyway (§0 of the decoder sketch: the optimizer reads a per-node partial-path state).

## 4. The three path-sampling methods (document all three)

**(1) Uniform random walk (+ stop-β) — default, cheapest.** Going up the DAG, pick a parent **uniformly** at each
node, and **terminate with probability β at each step**. Path prob ≈ `∏ 1/|parents(level)|` × the geometric
stopping term. Parameter-free but for β — literally "sample a parent at each level, sometimes stop early." Two
bonuses: **(a)** the stop gives geometric-depth (variable-length) paths ⇒ a *principled depth sampler* (graded-depth
ancestor membership — a better-motivated prefix-dropout); **(b)** a uniform up-walk with per-step stop is **closely
analogous to discrete PPR** for depth weighting — β plays the role of the PPR restart — so methods (1) and (3)
largely collapse into one knob. *Hedge (per review):* it is **not numerically identical** to PPR — a stop-walk
*terminates* at a node, whereas PPR *teleports back to the source* and keeps circulating mass; treat them as the same
*family* (geometric depth decay), not the same estimator, when comparing against a PPR baseline. *Direction
convention (state once so the knob isn't swapped):* **β = the per-step stop probability, so larger β → shallower**
paths (β→0 walks to the root; β→1 stays at the leaf). Start here.

**(2) Edge-weighted — canonical strength.** Weight edges by **type/confidence** — RefPearl (inside/owned) ≫
AliasPearl (cross-link), or graph-edge confidence — path probability = product of its edge weights. **Normalize
LOCALLY, per step** (a softmax over a node's parent edges at each hop), *not* globally over all root-paths — global
path normalization requires enumerating every path (intractable for large DAGs), while local per-step normalization
is a cheap, differentiable random-walk transition. Down-weights incidental cross-link lineages; encodes "primary home
vs cross-reference." The right refinement once uniform over-weights incidental multi-parenting.

**(3) PPR / flux — principled continuous.** Personalized PageRank from the node over the **reverse (ancestor)**
graph; each path's mass = the PPR flow along it, giving exp-decay / power-law weighting by depth × branching for
free. Reuses the scan-strategy flux machinery (the Green's-function / KCL framing). The continuous version of (1)+(2).

**Weighting semantics.** (1) = structural likelihood, (2) = lineage strength, (3) = both / continuous. Default (1);
move to (2) (effectively *structural × confidence*) because a node's *true* contexts are its canonical ancestors,
and pure branching over-weights incidental multi-parenting; (3) if/when the full flux is worth computing.

## 5. Method as a model input — higher-order, but adopt lazily

Conceptually `method` is a **factored token** like `corpus_emb`/`judge_emb`: the model conditions on
`(operator, method)`, and PATH becomes **queryable by method at inference** ("give me the PPR-weighted membership").
But **with only one method live, a method token is constant = no signal.** So:
- **Now:** method is a **data-generation choice** (how we sample the training paths) — *not* a model input.
- **Promote to a model input (`method_emb`)** only when **(a) ≥2 methods are trained** *and* **(b) inference needs to
  select/distinguish them.** Then the higher-order operator is realized.

(Same conclusion as the source-type discussion: design for it; adopt the machinery when the instance count justifies
it, not before.)

**Correction / the real mechanism — a learned MLP encoder, not more tokens.** A `method_emb` *token* (and the fixed
`op_emb` **codebook** generally) fits a *small discrete* set — fine for `{SYM, HIER, ELEM}`. But once operators are
**higher-order and parameterized** (`ancestor-path × method × stop-β × edge-weights`) the space is **continuous and
compositional**, and no lookup table — nor a handful of factored tokens — can span it (you can't *index* a real-valued
β, nor enumerate compositions). The mechanism that can: **compute the operator encoding with a small MLP from an
operator SPECIFICATION** — `op_encoding = MLP([relation_emb, method_emb, params…])` — instead of looking it up. The
MLP is the *higher-order function realized in the architecture*: the "operator constructor" that takes the parameters
(method, β, edge-weights) and emits the `d_model` operator token.
- **Parameterized:** β / edge-weights enter as real-valued inputs (a codebook holds points, not a continuum).
- **Compositional + interpolable:** unseen (method, β) configs map smoothly; the base relations `{SYM, HIER, ELEM}`
  become special *points* the MLP can still produce — the current codebook is the degenerate "lookup for one-hot
  specs" case.
- **Hypernetwork-flavored:** a small net generating the conditioning vector from a spec — the honest way to feed a
  *function-with-parameters* to the model. Spec source = learned per-component embeddings (relation/method) + scalar
  params; optionally grounded in e5-of-description (but that reintroduces prompt-rewriting; the MLP mediates it into a
  *learned* encoding).
- **Scope:** overkill for 3 discrete relations; it is the **enabling piece** for parameterized/higher-order operators
  — it's what turns "operator × method × representation" from a *factoring* into something the model can actually
  consume. Adopt it *when building* PATH-with-methods, and the codebook stays valid for the base relations meanwhile
  (the MLP can be introduced as `op_encoding = MLP(spec)` with the current ops as its first, one-hot-spec outputs).

**Preferred realization — a key–value *attention* lookup that reuses `op_weights` (not a plain MLP).** Rather than a
generic MLP, make the encoder a content-addressable attention over the *existing* operators:
- **keys** = e5(operator *names/descriptions*), one per existing operator;
- **values** = the existing learned `op_emb` rows (SYM/HIER/ELEM), used as *initialized* values (warm-start, still
  trainable);
- **query** = e5(the desired operator's spec/description);
- **encoding** = `softmax(query·keysᵀ / τ) @ values`.

The softmax **IS `op_weights`**, and the model already has the blended-operator pathway `op_weights @ op_emb.weight`
— so this doesn't add machinery, it **generates the superposition weights from the operator *description*** instead
of hand-setting them (the operator-superposition design, made content-addressed). It degenerates cleanly: a query
matching one key → one-hot `op_weights` → the current codebook lookup. And it **sidesteps the prompt-rewriting
worry**: e5 is only the *key* (routing); the *learned value* carries the semantics — new operators are *addressed*
by language but *encoded* by learned embeddings. Caveat (naming): the key is the name/description, so **naming/description
quality matters** (close descriptions → close keys → blended operators — right when they're related, under-resolved
when they aren't; use fuller descriptions than the bare short name). *Two more the review flagged:* **(cold-start)**
a genuinely-new operator's query spreads softmax mass over existing keys → a blend biased toward neighbours; early
gradients can then *pull the existing op embeddings*. Mitigate by **freezing `values` while a new operator's
representation converges**, or treating the new operator as an *additive residual* on top of the blend.
**(diagnostic)** on registration/startup, print `softmax(query·keysᵀ)` for each known operator — you want it
**near one-hot** for `SYM/HIER/ELEM`; a diffuse row flags a silent description collision.

**Numeric parameters (β, edge-weight scalars) — Fourier-encode, then FiLM-modulate.** The attention lookup routes
the *discrete/semantic* selection (which relation, which method); it cannot carry a **continuum** — β=0.3 isn't a
word and has no key. *(Note, per review: the lookup temperature **τ** is NOT one of these — it's a sharpness
hyperparameter of the router, not a path-sampling semantic; keep it a **fixed** scalar, or at most a single learned
scalar, and do NOT Fourier/FiLM it, else the model can learn to blur its own routing.)* So numeric params are injected on a **separate path**, in two steps:
1. **Fourier-feature encode each scalar** — `β → [sin(ω₁β), cos(ω₁β), …, sin(ω_kβ), cos(ω_kβ)]` at several
   frequencies. A *raw scalar is a poor MLP input* (one dimension; nets learn ~linear-only functions of it or ignore
   it); sinusoidal features lift it into a space where a small net can express a **rich, smooth** function of β
   across its range (same trick as transformer positional encodings / NeRF).
2. **FiLM-modulate the base encoding** — a tiny MLP maps the Fourier features → a **scale γ and shift**, applied to
   the attention-lookup output: `op_encoding = γ(fourier(β)) ⊙ base + shift(fourier(β))`.

So: **attention lookup = the discrete router (relation/method); Fourier + FiLM = the continuous parameter injector.**
This matches the semantics — β controls the path *depth distribution* (small β → deep paths, large β → shallow), so
the encoding should move *smoothly but expressively* as β sweeps, which a raw-scalar concat won't give but Fourier+
FiLM will. (Vector params like per-edge-type weights: Fourier-encode each scalar, or, if the *choice* of weighting is
discrete — uniform vs type-weighted vs confidence — that choice is semantic and goes in the attention query; only the
actual weight *values* take the Fourier+FiLM path.)

Notes: frequencies are indexed by the *dimension* (each dim a fixed ωᵢ, geometrically spaced), and the value is fed
to all — so the wavelength lives on the *β-axis*; set the lowest ω so its wavelength covers β's range (distinct,
non-wrapping codes), higher ω for resolution. The sin/cos *pair* handles sign automatically (β as an angle on the
unit circle) — no phase-shift term needed. **Use FIXED (not learned) frequencies** — chosen deliberately for
*reduced training* (consumer-grade hardware): fixed makes the encoding **parameter-free** (only the small FiLM MLP
trains), deterministic, and stable. Since β has a **known bounded range** (a probability/restart in [0,1]), the
frequency band is set **analytically** to that range, so learned frequencies — which mainly help when the relevant
scales are *unknown* — buy nothing here while adding parameters and mild instability. The only knob is the band
(set-once hyperparameter), not a learned weight. *Prior art:* sinusoidal positional
encoding (Vaswani et al. 2017, §3.5); Fourier features & the MLP spectral-bias motivation (Tancik et al., NeurIPS
2020; Rahimi & Recht, NeurIPS 2007); continuous-coordinate encoding (NeRF, Mildenhall et al. 2020); learned scalar
encoding (Time2Vec, Kazemi et al. 2019); FiLM modulation (Perez et al., AAAI 2018).

## 6. Why PATH (multi-path) is *not* redundant to HIER, though LINEAGE was

`HIER` = the single category-hierarchy **edge**. Single-path `LINEAGE` = one chain of `HIER` edges ⊂ (`HIER` +
transitive closure) — hence redundant. But the **multi-path superposition** is the DAG's *ancestor set / closure
with weights* — it expresses multi-context membership that a single chain (or a bare edge) does not emphasize. That
is the part our experiments did **not** test, and the reason to build PATH rather than conclude "paths don't help."

## 7. Start

`ancestor-path` operator, `method = random-walk`, sampled per step with wildcard masking, method kept as a
data-generation knob (no token yet). Add the `method` token when a second method (edge-weighted or PPR) is trained
and inference wants to select. Eval with the graceful-degradation metrics already built (path-overlap / matched-depth
on the paired hard subset) — the question is whether multi-path recovers ancestor branches better than HIER alone.
*Interpretability guard (per review):* the `matched-depth` metric is only meaningful once the **depth PEs (§3b) are
active** — a per-node baseline *without* them makes the model depth-blind, so a flat matched-depth would reflect the
missing PEs, not a lack of PATH signal. Wire depth PEs before reading matched-depth results.
