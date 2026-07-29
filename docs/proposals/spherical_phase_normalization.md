# Spherical Normalization via Fourier Phases — Tori, Hopf Fibrations, and Hypercomplex Structure

**Status:** Theory note / proposal
**Context:** The filing-system work (`prototypes/mu_cosine/`) composes tokens additively
(`node(X)@gen0 = e5(X) + gen_emb[0]`, etc.), but e5 embeddings live on the unit sphere
`S^{n-1}` and additive composition leaves the sphere: the sum of unit vectors is not unit,
and renormalizing after the fact distorts relative geometry. This note works out a proposed
alternative — represent a signal by the phases of its finite complex Fourier transform,
normalized over the least common multiple of the component periods, and treat each phase as
a coordinate on a unit sphere — and identifies exactly what that construction gives, what it
costs, and how it connects to hypercomplex numbers.

**TL;DR.** Phases-as-angles gives you a *torus*, not a sphere, unless you force them through
hyperspherical coordinates — and that forcing is precisely the "scaling of the frequency
components" one notices: it couples the amplitude spectrum to the phase spectrum in an
ordering-dependent way, with coordinate singularities. The construction that *does* work is
older and better: unit-modulus spectra (phase-only representations, FHRR), which live on the
**Clifford torus** `T^n ⊂ S^{2n-1}` — a flat torus sitting *inside* the unit sphere, where
normalization holds by construction for every phase assignment, and composition by phase
addition is exactly norm-preserving. The hypercomplex hunch is correct and lands in a
specific place: unit phasors are unit complex numbers; a *pair* of Fourier bins with unit
total energy is literally a unit quaternion (with the Hopf fibration `S^1 → S^3 → S^2`
separating common phase from relative structure); and beyond dimension 4 the division-algebra
ladder stops (Hurwitz: only ℝ, ℂ, ℍ, 𝕆), so the correct general framework is the one this
repo already adopted — bivectors/rotors (`DESIGN_model_applications.md` §"Theory note",
`docs/design/ORTHOGONAL_CODEBOOK_DESIGN.md`). The DFT is exactly the change of basis in
which circular-convolution rotors become commuting 2-plane rotations: **FHRR phase-binding
is a maximal torus of Spin(2n) expressed in the Fourier basis.**

---

## 1. What the DFT already normalizes, and what the LCM buys

Take a real signal `x ∈ R^N` with DFT coefficients `X_k = r_k e^{iφ_k}`. Two facts frame
everything below.

**Parseval gives a sphere on amplitudes, not phases.** `‖x‖² = (1/N) Σ_k |X_k|²`, so
L2-normalizing the *signal* puts the amplitude vector `(r_0, …, r_{N-1})` on (the positive
orthant of) a sphere. The phases are extra coordinates: the space of unit-norm spectra is a
torus bundle over that amplitude sphere — each point of the amplitude simplex carries a torus
of phases. (For the complex-vector case, quotienting by global phase makes this the standard
toric/moment-map picture of `CP^{n-1}`.) So "the DFT of a normalized signal" already lives
on a sphere — but the sphere constrains amplitudes, while the information we want to treat
as coordinates is in the phases.

**The LCM of periods is the fundamental domain of the phase torus.** Bin `k` of an `N`-point
DFT has period `N / gcd(k, N)` samples; the LCM over the retained bins divides `N`, and for
the full DFT equals `N`. Normalizing time to that common period means the joint phase vector
`φ(t) = (2πk₁t/N, …, 2πk_nt/N) + φ⁰` is a *closed winding line* on the torus
`T^n = (R/2πZ)^n`: after one LCM period every phase returns. This is the right formalization
of the LCM intuition — it selects the torus `R^n / (period lattice)` as the state space. Two
consequences worth noting:

- A **circular time shift** by `τ` maps `φ_k ↦ φ_k + 2πkτ/N` — a *linear-in-k phase ramp*,
  i.e. a specific one-parameter subgroup of torus translations. Shift-invariant comparison =
  quotienting by that subgroup = phase-only correlation (computable by FFT; see §5).
- With **irrationally related** frequencies (the continuous analog, as in RoPE or the fixed
  geometric frequencies of `DESIGN_path_operator.md` §"Fourier feature encoding"), the
  winding line never closes and is dense on the torus — the quasi-periodicity that makes
  geometrically spaced frequencies give quasi-orthogonal codes over long ranges.

## 2. Phases as hyperspherical coordinates: what the forcing costs

The proposal under examination: take the `n` phases `φ_1, …, φ_n` and use them as the
angular coordinates of a point on a unit sphere, so normalization holds by construction.
Hyperspherical coordinates on `S^n ⊂ R^{n+1}`:

```
x_1 = cos θ_1
x_2 = sin θ_1 cos θ_2
x_3 = sin θ_1 sin θ_2 cos θ_3
...
x_{n+1} = sin θ_1 ⋯ sin θ_{n-1} sin θ_n
```

Substituting `θ_k = φ_k` indeed lands on the unit sphere for every phase vector. But three
structural problems follow, and the second is exactly the "scaling of frequency components"
one senses in this construction:

1. **Topology obstruction.** The phase space is a torus `T^n`; the target is a sphere `S^n`.
   These are different manifolds (`π₁(T^n) = Z^n`, `π₁(S^n) = 0` for `n ≥ 2`), so no
   continuous map between them is invertible: the map must collapse distinct phase vectors.
   Concretely, all hyperspherical angles except the last range over `[0, π]`, while phases
   range over `[0, 2π)` — the substitution double-covers, and whenever any `sin θ_j = 0`
   (a "pole"), *all later phases are annihilated* regardless of their values. These are
   gimbal-lock singularities: representational dead zones and gradient pathologies.

2. **Induced amplitude coupling.** The round metric on `S^n` in these coordinates is

   ```
   ds² = dθ_1² + sin²θ_1 dθ_2² + sin²θ_1 sin²θ_2 dθ_3² + ⋯
   ```

   so the effective sensitivity ("amplitude") of phase `k` is `∏_{j<k} sin φ_j` — the
   amplitude spectrum is no longer free but *determined by the phase spectrum*, through an
   arbitrary product-of-sines rule that depends on how the bins are ordered. In Fourier
   terms, amplitude and phase are independent quantities; this construction welds them
   together. That is the precise sense in which "we ensure normalization by scaling the
   frequency components."

3. **Ordering dependence.** Permuting the bins changes the geometry. The DFT has no
   preferred bin ordering for this purpose, so the representation acquires structure the
   signal does not have.

**When the coupling might be a feature.** The chain `dθ_1² + sin²θ_1 dθ_2² + …` is a
coarse-to-fine hierarchy: the first angle always matters; the k-th matters only when all
earlier angles sit near the equator. If the bins are deliberately ordered coarse-to-fine
(low to high frequency), this resembles the filing hierarchy's generation structure — an
early "coarse" phase gates the relevance of later "fine" phases, similar in spirit to the
depth-tagged lineage tokens of `DESIGN_directional_attention.md`. That would be a design
choice to make explicitly, with the pole singularities acknowledged — not a free
normalization trick.

## 3. The construction that works: the Clifford torus in `S^{2n-1}`

There is a way to get *unconditional* unit norm from phases with none of the above costs:
represent each phase as a unit phasor and concatenate.

```
Φ(φ) = (1/√n) (cos φ_1, sin φ_1, cos φ_2, sin φ_2, …, cos φ_n, sin φ_n) ∈ R^{2n}
```

Then `‖Φ(φ)‖ = 1` identically — the image is the **Clifford torus** `T^n ⊂ S^{2n-1}`, the
flat torus isometrically embedded in the unit sphere. This spends 2 real dimensions per
phase instead of 1, and in exchange:

- **No singularities, no ordering dependence, no amplitude coupling.** The induced metric is
  the flat metric `(1/n) Σ dφ_k²`; every phase enters symmetrically.
- **Cosine similarity has a clean closed form.** For two phase vectors,

  ```
  ⟨Φ(φ), Φ(ψ)⟩ = (1/n) Σ_k cos(φ_k − ψ_k)
  ```

  — the *mean resultant* of the phase differences, i.e. circular-statistics agreement
  (a von Mises–type similarity). Standard cosine ranking infrastructure applies unchanged.
- **Norm-preserving composition.** Adding a phase offset vector `ψ` (binding) maps the
  torus to itself: `φ ↦ φ + ψ` is a rotation of `R^{2n}` (block-diagonal, one Givens
  rotation per phasor plane). Composition never leaves the sphere — this is the direct
  replacement for additive token composition that motivated this note.

This is not a new object: it is exactly **FHRR** (Fourier Holographic Reduced
Representations, Plate) — HRR carried out in the frequency domain with all amplitudes pinned
to 1. `DESIGN_tree_position_encoding_theory.md` §5 already works with the Fourier-domain
form `DFT(z) = DFT(d) ⊙ DFT(b)`; pinning `|DFT(·)| ≡ 1` turns that elementwise product into
pure phase addition, makes binding exactly invertible by the conjugate (`Φ(−φ)` — the
known-key caveats of §5 still apply for *unknown* keys), and makes it exactly
norm-preserving. Amplitude flattening also has independent classical support: phase carries
most of the structural information in signals and images (Oppenheim & Lim 1981), and
phase-only ("whitened") correlation is a standard registration technique.

**Superposition amplitude is free evidence coherence.** When phase-bound bundles are
*superposed* (summed), the per-bin magnitude of the sum before re-projection onto the torus
is the resultant length of the contributing phasors — a direct measure of phase agreement
among the superposed items. Re-projecting to unit modulus is the "cleanup" step, but the
discarded magnitudes are exactly the per-component confidence signal that the evidence-fusion
machinery wants (`THEORY_evidence_fusion.md`'s complaint that linear blending drops
covariance; `DESIGN_amortized_fusion_heads.md`'s Kalman framing — resultant length plays the
role of an innovation-consistency weight). Keep them.

## 4. The hypercomplex connection — real, and it lands on the repo's existing rotors

The connection is genuine and has a precise shape. Climbing the ladder:

**ℂ (complex).** A unit-modulus Fourier coefficient `e^{iφ}` *is* a unit complex number;
a unit-modulus spectrum is an element of the torus group `U(1)^n ⊂ C^n ≅ R^{2n}`, and
`(1/√n)`-scaling embeds it in `S^{2n-1}` — that is §3. Binding = the group operation.

**ℍ (quaternions).** Take just *two* complex bins with unit total energy:
`(z_1, z_2) ∈ C²`, `|z_1|² + |z_2|² = 1`. That is a point of `S^3`, and
`q = z_1 + z_2 j` is a **unit quaternion** — a pair of Fourier coefficients with unit energy
literally *is* a unit quaternion. The **Hopf fibration** `S^1 → S^3 → S^2` then performs an
interpretable decomposition: the fiber is the common phase; the base `S^2 ≅ CP^1` (Bloch
sphere) encodes the amplitude ratio and *relative* phase of the two bins. This is the
2-bin instance of the general statement: quotienting a unit-norm complex spectrum by global
phase gives `CP^{n-1}` with the Fubini–Study metric, where the natural similarity is
`|⟨x, y⟩|` — fidelity, invariant to common phase. (For real signals, conjugate symmetry
`X_{N-k} = X̄_k` means the free phases are the ~`N/2` positive-frequency bins, and the
physically meaningful quotient is by the *phase-ramp* subgroup of §1 — time shift — rather
than global phase; both are `U(1)` subgroups of the torus.)

**𝕆 and the obstruction.** The ladder stops where it always stops: by Hurwitz/Adams, the
normed division algebras are ℝ, ℂ, ℍ, 𝕆 (dimensions 1, 2, 4, 8), and the only
parallelizable spheres are `S^1, S^3, S^7`. This is why "make the phases coordinates on one
big unit sphere *with a multiplication*" cannot work in general dimension — there is no
hypercomplex algebra whose unit sphere is `S^{n}` for other `n`. The same theorem family
underlies the observation already recorded at `DESIGN_model_applications.md`
("Why bivector, not cross product": the cross product exists only in 3D/7D). Beyond
dimension 8 the two viable generalizations are exactly:

- the **torus** `U(1)^n` — commuting phase rotations (this note, FHRR), and
- the **rotor group** `Spin(2n)` — general bivector exponentials (the repo's existing
  framework: `DESIGN_model_applications.md` §"Theory note", `ORTHOGONAL_CODEBOOK_DESIGN.md`,
  `rotation_transformer.py`).

**And they are the same picture in different bases.** Circular convolution by a
unit-spectrum filter is an orthogonal circulant matrix; the DFT diagonalizes all circulants
simultaneously into 2×2 rotation blocks — one Givens rotation per conjugate bin pair. In
Clifford terms: the bivector generator is supported on `n` mutually orthogonal coordinate
planes `e_{2k-1} ∧ e_{2k}`, and because **orthogonal planes commute**,
`exp(Σ_k φ_k B_k) = ∏_k exp(φ_k B_k)` — precisely the key identity of
`ORTHOGONAL_CODEBOOK_DESIGN.md`, with the Rodrigues-formula cost benefit. So:

> **FHRR phase-binding is a maximal torus of the rotor group, written in the Fourier basis.**
> The unit-spectrum representation is not an alternative to the bivector framework — it is
> its maximally-commuting special case, with the DFT as the basis in which that torus is
> coordinate-aligned. RoPE is the same object with position-proportional phases.

The trade is expressiveness vs. cost and commutativity: general rotors (`O(Kn)` blades,
non-commuting, order-sensitive — appropriate for directional operators like `subcategory`)
vs. the torus (fully commuting, `O(n)`, exactly invertible — appropriate for role/slot
binding and positional structure where order-independence of the *bindings themselves* is
fine and the sphere must be preserved exactly).

## 5. Practical implications for the filing system

1. **Replace additive tags with phase binding where norm preservation matters.** The current
   additive scheme (`e5(X) + gen_emb[0]`, `+ role_emb[ANCHOR]`) can be swapped, per slot,
   for binding by a fixed per-role phase vector in a fixed unitary basis (FFT of the
   embedding, multiply by `e^{iψ_role}`, inverse FFT — an orthogonal transform, so unit norm
   and pairwise cosines among identically-bound tokens are preserved *exactly*). Roles are
   unbound by the conjugate. This is a controlled experiment: same token count, same
   dimension, additive vs. rotational tagging, evaluated under the `PROTOCOL_filing_ranker_eval.md`
   harness.
2. **Do not use the hyperspherical-coordinate map as a normalization device.** Use it, if at
   all, only as a deliberate coarse-to-fine gating prior with bins explicitly ordered by
   scale (§2), and expect pole pathologies.
3. **Similarity menu, by invariance.** Plain cosine on the Clifford torus =
   `(1/n) Σ cos Δφ_k` (no invariance); fidelity `|⟨x,y⟩|` (global-phase invariance,
   Fubini–Study/`CP^{n-1}`); max over circular shifts via FFT = phase-only correlation
   (shift invariance). Choose per operator — SYM plausibly wants an invariant form,
   HIER/ELEM plausibly do not.
4. **Harvest superposition magnitudes as coherence weights** for the fusion heads (§3),
   instead of discarding them at renormalization.
5. **Frequency choice.** For continuous parameters (depth, `β` of `DESIGN_path_operator.md`),
   keep fixed geometric frequencies; the LCM/winding analysis of §1 says commensurate
   (rational) frequency ratios give exactly-periodic codes with a known collision period,
   while incommensurate ratios trade that for dense non-repetition. Both are torus
   statements; pick per operator and pin in the run manifest.

## References

- Plate, T. — *Holographic Reduced Representation* (FHRR: unit-modulus frequency-domain HRR).
- Oppenheim, A. & Lim, J. (1981) — *The importance of phase in signals.* Proc. IEEE.
- Tancik et al. (2020) — *Fourier features let networks learn high frequency functions.* NeurIPS.
- Su et al. (2021) — *RoFormer: Enhanced transformer with rotary position embedding.*
- Baez, J. (2002) — *The octonions.* Bull. AMS. (Hurwitz theorem, Hopf fibrations, parallelizable spheres.)
- Hestenes, D. — *New Foundations for Classical Mechanics* (rotors, bivectors).
- In-repo: `prototypes/mu_cosine/DESIGN_model_applications.md` (§"Theory note — μ is a
  geometric (scalar + bivector) object"), `prototypes/mu_cosine/DESIGN_tree_position_encoding_theory.md`
  (§5 HRR unbinding), `docs/design/ORTHOGONAL_CODEBOOK_DESIGN.md` (commuting orthogonal
  planes), `prototypes/mu_cosine/DESIGN_path_operator.md` (Fourier features + FiLM),
  `prototypes/mu_cosine/THEORY_evidence_fusion.md`, `prototypes/mu_cosine/DESIGN_amortized_fusion_heads.md`.
