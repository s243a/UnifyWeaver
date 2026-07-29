# Phase Binding — Specification

Precise definition of the phase-binding operator, its parameterization,
its invariants, and its integration points in
`prototypes/mu_cosine/mu_attention.py`. Motivation and scope are in
`PHASE_BINDING_PHILOSOPHY.md`; theory in
`docs/proposals/spherical_phase_normalization.md`.

## 1. The operator

### 1.1 Definition

For `x ∈ R^D` (here `D = 384`, even) and a phase vector `ψ`, define

```
U_ψ(x) = irfft( rfft(x) ⊙ e^{iψ} ,  n = D )
```

where `rfft(x) ∈ C^{D/2+1}` (bins `k = 0 … D/2`) and
`ψ ∈ R^{D/2+1}`. Reality of the output requires the DC and Nyquist
bins to stay real:

```
ψ_0 = 0,   ψ_{D/2} = 0            (hard constraint, not learned)
```

leaving `n_free = D/2 − 1 = 191` free phases. Implementations MUST
enforce the constraint structurally (parameter of size 191, zeros
concatenated at positions 0 and D/2), not by penalty.

### 1.2 Invariants (normative)

For all `x, y ∈ R^D` and phase vectors `ψ, φ`:

| # | Invariant | Statement |
|---|---|---|
| I1 | Isometry | `‖U_ψ x‖₂ = ‖x‖₂` (exact; `U_ψ` is orthogonal circulant) |
| I2 | Cosine preservation | `⟨U_ψ x, U_ψ y⟩ = ⟨x, y⟩` |
| I3 | Relative binding | `⟨U_ψ x, U_φ y⟩ = ⟨U_{ψ−φ} x, y⟩` — depends on `ψ − φ` only |
| I4 | Composition | `U_ψ U_φ = U_{ψ+φ}`; the tags form an abelian group |
| I5 | Unbinding | `U_ψ^{-1} = U_{−ψ} = U_ψ^T` |
| I6 | Identity | `U_0 = I` (exact warm-start no-op) |
| I7 | Noise closure | `x` uniform on `S^{D-1}` ⇒ `U_ψ x` uniform on `S^{D-1}` (off-manifold-noise contract preserved) |

Unit tests MUST assert I1–I6 to within fp32 FFT round-off
(`atol ≤ 1e-5` on unit vectors) and are the acceptance gate for P0 of
the implementation plan.

### 1.3 Fractional / continuous binding

For a scalar parameter `t ∈ R` (depth, position, β):

```
U_step^t = U_{t·ψ_step}
```

One learned (or fixed) `ψ_step` encodes the whole axis. By I3, the
similarity between `U_step^{t₁} x` and `U_step^{t₂} y` depends on
`t₁ − t₂` only (the RoPE relative-offset property). Non-integer `t` is
well-defined; no table lookup, no `max_gen` cap.

**Frequency/periodicity policy.** The effective frequencies are
`ψ_step` itself. Two admissible schemes, pinned per run:

- `geometric` (default): `ψ_step,k = θ · g^{−k/n_free}` with base
  `θ` and decay `g` (RoPE-style, e.g. `θ = 1`, `g = 10000`).
  Incommensurate in practice ⇒ no exact collisions over any usable
  depth range.
- `learned`: `ψ_step` is a free parameter, zero-init (I6 ⇒ warm-start
  no-op; depth signal is learned). Optionally initialized to
  `geometric` when warm-start exactness is not required.

The LCM analysis (`spherical_phase_normalization.md` §1, §5) governs
deliberate periodic wraps; if a wrap at period `P` is wanted, choose
`ψ_step,k = 2π m_k / P`, `m_k ∈ Z`.

## 2. Tag algebra for the MuAttention token set

Current additive composition (`mu_attention.py`, `Tokenizer.build` +
`MuAttention.forward`):

```
token = content(e5 or noise)
      + gen_emb[d]·1[gen] + anchor_tag·1[anchor] + nodetype_emb[τ]
      + prefix_emb[ρ] + op/prov embeddings (own slots or added)
```

Phase-bound composition (`tag_mode="phase"`), per token:

```
token = U_{ψ(token)} ( content )
ψ(token) = d·ψ_step                    (lineage generation, d = min-hop)
         + ψ_role[r]                   (r ∈ {anchor, node, anc, prov_slot})
         + ψ_nodetype[τ]               (τ ∈ NODETYPE, row 0 pinned to 0)
         + ψ_prefix[ρ]                 (ρ ∈ {query:, passage:, none})
```

Notes (normative):

- **Slot coverage.** Phase 1 scope binds the *per-token structural
  tags*: generation, anchor/role, node-type, prefix. The operator
  token, provenance token content (`corpus_emb + judge_emb + …`), and
  readout are unchanged: they are *content-bearing* slots (whole
  learned vectors, not tags on frozen content) and stay additive.
  `prov_tag` (the slot marker) MAY be migrated to `ψ_role[prov_slot]`
  since it marks frozen-noise content.
- **All ψ tables zero-init** ⇒ by I4+I6 the entire tagged token equals
  raw `content` at init. Combined with `tag_mode="additive"` retained
  as the default, warm starts and old checkpoints are exact.
- **Commutativity is intended here.** The token set is unordered
  (permutation-invariant encoder); tags identify slots, they do not
  encode sequence. Ordered path composition (root-to-node role paths,
  `DESIGN_tree_position_encoding_theory.md` §4) MUST NOT be encoded as
  a bare sum of role phases — `a/b` would collide with `b/a`. Ordered
  paths use depth-indexed keys `ψ_role[r] + t·ψ_step` per edge
  (distinguishes which level a role occupied) or the rotor layer.
- **Noise slots** (`⌀`, masked provenance) are phase-bound like any
  content (I7 makes this a no-op distributionally); the deterministic
  per-node seeds are unchanged.

## 3. Similarity and readout

Phase binding changes token construction only. The encoder, CLS-style
op-token readout, per-operator heads, dual-judge blends, and μ ranges
are unchanged.

For analysis tooling (not the model), the induced closed forms are:

- cross-tag cosine: `⟨U_ψ x, U_φ y⟩ = Σ_k Re( X_k Ȳ_k e^{i(ψ_k−φ_k)} )`
  with `X = rfft(x)`, `Y = rfft(y)`;
- phase-only (FHRR) variant, if amplitudes are flattened:
  cosine reduces to `(1/n) Σ_k cos(Δφ_k)`. Amplitude flattening of e5
  content is **out of scope** for the model path (it changes the frozen
  content, violating `RANK-001`); it is available in analysis tools
  under an explicit flag.
- superposition coherence: for summed phase-bound bundles, per-bin
  magnitude of the sum is the resultant length of the contributing
  phasors — exported as an optional diagnostic
  (`coherence(bundle) ∈ R^{D/2+1}`) for the fusion heads
  (`DESIGN_amortized_fusion_heads.md`).

## 4. Parameters, manifest, and compatibility

### 4.1 New parameters (all zero-init)

| Parameter | Shape | Replaces (additive) |
|---|---|---|
| `psi_step` | `[n_free]` | `gen_emb` (D·(max_gen+1) params → 191) |
| `psi_role` | `[n_roles, n_free]` | `anchor_tag` (+ optionally `prov_tag`) |
| `psi_nodetype` | `[n_nodetype, n_free]` | `nodetype_emb` |
| `psi_prefix` | `[3, n_free]` | `prefix_emb` |

Additive tables remain in the module (checkpoint compatibility;
`tag_mode` selects the path in `forward`, mirroring the
`judge_name`-bypasses-`judge_emb` pattern).

### 4.2 Run-manifest entries (required when `tag_mode="phase"`)

```
tag_mode: phase | additive          (default: additive)
phase_basis: rfft384                (pinned; the only defined value)
phase_step_scheme: geometric | learned
phase_step_theta/g:                 (iff geometric)
phase_slots: [gen, role, nodetype, prefix]   (which tags are bound)
```

Silently changing any of these changes the model class
(`DESIGN_tree_position_encoding_theory.md` ground rules apply).

### 4.3 Numerical notes

- fp32 rfft/irfft round-trip error ~1e-7 relative; tolerated, no
  re-normalization step is permitted in the model path (it would mask
  I1 violations — assert instead).
- Batched implementation: one `rfft` per `[B,T,D]` content tensor, one
  phase-sum per token, one `irfft`; cost ≈ two FFTs per forward,
  negligible against the encoder.
- CPU-only training (the repo's consumer-hardware constraint) is
  respected: `torch.fft` requires no GPU.

## 5. Explicitly out of scope

- Learned non-commuting rotations (rotor codebook — exists,
  `ORTHOGONAL_CODEBOOK_DESIGN.md`).
- Amplitude flattening / phase-only e5 content in the ranking path.
- Hyperspherical-coordinate normalization (rejected;
  `spherical_phase_normalization.md` §2).
- Any change to e5 caching, the DAG, sampling, or eval protocols.
