# Phase Binding — Philosophy

## What this is

A design for **norm-preserving token tagging** in the filing system's
`MuAttention` models: replace additive tag composition
(`token = e5(X) + gen_emb[d] + role_emb[...]`) with **phase binding** —
rotating the frozen e5 embedding by a per-tag phase vector in a fixed
unitary (Fourier) basis. The tagged token stays exactly on the unit
sphere, identically-tagged tokens keep their pairwise cosines exactly,
tags compose as a group, and depth becomes a *continuous* translation
instead of a lookup table.

The theory behind this design is worked out in
`docs/proposals/spherical_phase_normalization.md`: unit-modulus spectra
live on the Clifford torus `T^n ⊂ S^{2n-1}` (the flat torus inside the
unit sphere), composition by phase addition is a rotation, and the
construction is the maximally-commuting special case (a maximal torus)
of the bivector/rotor framework already established in
`docs/design/ORTHOGONAL_CODEBOOK_DESIGN.md` and
`prototypes/mu_cosine/DESIGN_model_applications.md`. RoPE and FHRR are
the two well-known instances.

The deliverable, in one sentence: a drop-in **`tag_mode="phase"`**
alternative to the additive tags in `prototypes/mu_cosine/mu_attention.py`,
warm-start-exact, A/B-evaluable under the existing filing-ranker
protocol.

## Why now

Three pressures motivate this work:

1. **Additive tags leave the sphere.** e5 embeddings are unit-normed by
   construction (`build_e5_tables`, `normalize_embeddings=True`); the
   entire ranking backbone (`RANK-001` in
   `docs/design/ARCHITECTURE_filing_engine.md`) is cosine geometry on
   that sphere. But the tokenizer's composition step adds learned tags
   (`gen_emb`, `anchor_tag`, `nodetype_emb`, `prefix_emb`, provenance
   embeddings) directly to the frozen e5 vector, producing tokens of
   uncontrolled norm at uncontrolled distances from the manifold the
   backbone's geometry lives on. LayerNorm inside the transformer papers
   over the norm but not the distortion of *relative* geometry among
   tokens with different tags.

2. **Depth wants to be continuous.** `gen_emb` is a per-generation
   lookup table (`max_gen + 1` rows). Min-hop depth is genuinely ordinal
   and approximately metric, but the table gives the model no notion
   that gen-2 sits between gen-1 and gen-3. Phase binding encodes depth
   `d` as `d` fractional applications of one step rotation
   (`ψ_d = d·ψ_step`) — interpolation, extrapolation, and the
   relative-offset property (similarity between two depth-bound tokens
   depends only on `d₁ − d₂`) fall out of the group structure. This is
   the fractional-power-encoding / Spatial-Semantic-Pointer
   construction, and RoPE's position handling, applied to lineage depth.

3. **Tags should be removable.** Additive tags are not invertible: the
   model cannot recover the untagged e5 vector, and neither can we in
   analysis. Phase binding is exactly invertible by the conjugate
   phase — useful for probing (unbind the depth, ask what's left), for
   the HRR-style compositional experiments of
   `prototypes/mu_cosine/DESIGN_tree_position_encoding_theory.md`, and
   for keeping the off-manifold-noise contract (`⌀ = unit noise`) clean:
   a phase-bound noise token is still unit noise.

## The core insight

**The group that preserves the sphere is rotations, and the cheap
commuting rotations are phase translations in a fixed unitary basis.**

For a real vector `x ∈ R^384`, take its real FFT (193 complex bins),
multiply bin `k` by `e^{iψ_k}`, invert. The resulting map `U_ψ` is an
orthogonal circulant matrix:

- `‖U_ψ x‖ = ‖x‖` exactly — tagged tokens never leave the sphere;
- `⟨U_ψ x, U_ψ y⟩ = ⟨x, y⟩` exactly — a tag is an isometry, so the
  e5 geometry among identically-tagged tokens is untouched, while
  *cross-tag* geometry depends only on the phase *difference* — the
  relative-position property;
- `U_ψ U_φ = U_{ψ+φ}` — tags compose; `U_ψ^{-1} = U_{−ψ}` — tags
  unbind; `U_0 = I` — **zero-init phases are an exact warm-start
  no-op**, matching the repo's zero-init convention for every recent
  addition (`nodetype_emb`, `prefix_emb`, `account_emb`,
  `struct_lambda`);
- cost `O(d log d)` per token via FFT, no learned matrices.

The DFT basis is arbitrary but fixed — e5 coordinates are not "really"
frequencies. What matters is that *some* pinned unitary basis makes a
family of commuting rotations diagonal and cheap. (A learned orthogonal
basis is the non-commuting generalization; that is the rotor codebook,
already built, and deliberately out of scope here.)

## Design principles

1. **Warm-start exactness over cleverness.** Every mode change must
   have a configuration that reproduces the current model bit-for-bit
   at initialization (zero phases = identity). This is the same
   discipline as `struct_lambda = 0`, `NameFunctionCond` residuals
   `r = 0`, and zero-init `nodetype_emb`.

2. **Spend commutativity deliberately.** Phase tags commute — binding
   depth then role equals role then depth. That is correct for
   depth/role slot tags (the set-transformer is permutation-invariant
   anyway) and *wrong* for ordered path composition (`a/b` vs `b/a`).
   Ordered composition stays with depth-indexed keys or escalates to
   the non-commuting rotor layer. The spec draws this line explicitly.

3. **The tag is a prior, not the estimator** (the bitter-lesson caveat
   of `DESIGN_model_applications.md` §"Theory note"). Additive tags
   into a trained transformer may work fine in practice — LayerNorm and
   attention can learn around geometric distortion. The claim here is
   geometric hygiene with exact invariants, **to be tested, not
   assumed**: the A/B under `PROTOCOL_filing_ranker_eval.md` is part of
   the design, and the additive path remains the default until the
   phase path wins on the ledger.

4. **Everything pinned in the manifest.** Basis (rfft, size 384, bin
   ordering), which slots are phase-bound vs additive, frequency
   parameterization, and init scheme are run-manifest entries, per the
   repo's reproducibility rules
   (`docs/proposals/reproducible_embedding_datasets.md`).

## What this is not

- **Not a replacement for the rotor codebook.** `rotation_transformer.py`
  and the orthogonal codebook solve cross-model *mapping* with general
  rotations; this design tags tokens with commuting rotations inside one
  model. They meet in the theory (maximal torus ⊂ Spin(2n)) and stay
  separate in code until an experiment demands otherwise.
- **Not a new similarity backbone.** e5 cosine remains the coarse
  ranker (`RANK-001`); phase binding changes how μ-model *inputs* are
  composed, nothing upstream.
- **Not a claim that phases make pairs recoverable.** The known-key
  caveats of `DESIGN_tree_position_encoding_theory.md` §5 stand:
  unbinding needs the key.

## Relationship to existing documents

| Document | Relationship |
|---|---|
| `docs/proposals/spherical_phase_normalization.md` | The theory this design operationalizes (Clifford torus, FHRR, RoPE, hypercomplex/rotor connection). |
| `prototypes/mu_cosine/DESIGN_directional_attention.md` | Defines the token set whose composition step this design modifies. |
| `prototypes/mu_cosine/DESIGN_tree_position_encoding_theory.md` | Composition-family theory (§3) and HRR unbinding limits (§5) that scope what phase binding may claim. |
| `docs/design/ORTHOGONAL_CODEBOOK_DESIGN.md` | The non-commuting generalization ("orthogonal planes commute" is the shared identity). |
| `prototypes/mu_cosine/DESIGN_path_operator.md` §Fourier features | The read-out (coordinates-as-features) use of the same torus; phase binding is the operator (group-action) use. |
| `PHASE_BINDING_SPECIFICATION.md` | The precise operator, parameterization, and invariants. |
| `PHASE_BINDING_IMPLEMENTATION_PLAN.md` | Phased rollout with gates. |
