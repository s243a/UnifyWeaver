# Tree Position Encodings for Expression ASTs — Theory Note (FUTURE WORK)

**Status: theory capture, 2026-07-24 (owner + Perplexity/Sonnet-5 discussion).** Companion to
`DESIGN_expression_encoder_future.md`; same gating (post-P1, post-P3-plateau). This note records
the mathematical structure so a future implementer (possibly a separate agent) does not
re-derive it.

## The problem

A node in an expression AST (`e5(routing(e5, sonnet.lineage, ...))`) is doubly positioned:
**depth** k (operator vs argument vs sub-argument — abstraction level) and **breadth** j
(which argument at that depth — role/order). P3's current plan collapses this to fixed scalar
per-position weights. The question: how should a positional embedding represent (k, j) jointly?

## The factorization and the combination operator

Give depth and breadth their own embedding vocabularies, d_k and b_j, and combine. The candidate
combinations form a strict information hierarchy:

| operation | output dim | information preserved | Gaussian analogy |
|---|---|---|---|
| Kronecker d ⊗ b | D_d · D_b | exact, fully separable joint | full joint covariance Σ |
| circular convolution d ∗ b | D | Fourier-basis projection of the Kronecker | low-rank/diagonal Σ approximation |
| learned bilinear W(d ⊗ b) | D | learned projection of the Kronecker | fitted low-rank Σ |
| quadratic form dᵀA b | 1 | one interaction scalar | log-likelihood of the joint |
| additive d + b | D | marginals only, no interactions | independence, Σ = diag |

**Key identity (the owner's question, answered affirmatively): circular convolution IS a
dimensionality reduction of the Kronecker product** — a fixed linear projection. In the Fourier
domain, F(d ∗ b)_n = F(d)_n · F(b)_n: the D²-dimensional outer product is contracted to D
dimensions using the DFT matrix as the projection basis. Fixed (no training, approximately
invertible — the holographic reduced representation / HRR construction), but the Fourier basis
is not necessarily optimal for a given vocabulary.

**The quadratic-form / joint-Gaussian analogy, made precise:** for zero-mean joint Gaussian x,
−½ xᵀΣ⁻¹x = −½ vec(xxᵀ)ᵀ vec(Σ⁻¹) — the outer product xxᵀ is the rank-1 Kronecker self-product,
and contracting against the precision matrix is a linear projection of the Kronecker product to a
SCALAR: maximal compression, keeping exactly the information a score needs and nothing
recoverable. Attention's qᵀk is this same object; multi-head attention learns D_head such
projections in parallel — i.e., a learned partial recovery of the Kronecker structure. This
places the whole design space on one axis: how much of the joint (depth × breadth) interaction
structure do you keep, and is the projection fixed (Fourier) or learned (W)?

## Separability requirement

Depth and breadth vocabularies must be DISTINCT (independent initializations; distinguishable
subspaces). With a shared vocabulary the combination degenerates: the network cannot tell
(depth=k, breadth=j) from other pairs producing the same product. This was the owner's own
caveat and it is load-bearing — it is what makes the joint representation identifiable.

## Recommendation for THIS grammar

Expression trees here are shallow and narrow (depth ≤ ~4: operator → argument → sub-argument /
kwarg; breadth ≤ ~8). Therefore:

- Full Kronecker is FEASIBLE (small vocabularies ⇒ manageable D_d·D_b) but likely wasteful.
- **Learned bilinear W(d ⊗ b) is the recommended sweet spot**: a few hundred parameters,
  exactly separable, discovers which depth×breadth interactions matter for this grammar instead
  of assuming the Fourier basis. P3's fixed scalar position weights are its rank-degenerate
  special case, so it slots into the existing ablation ladder rather than replacing it.
- Preregistered ablation ladder when implemented: fixed-weight sum (P3 baseline) → additive
  d + b → circular convolution (HRR) → learned bilinear → full Kronecker. Determinism contract
  carries over: any stochastic element seeded by SHA-256 of the canonical AST.

## Cautions

- With ~922 labeled rows, NO learned position scheme is trainable from task labels alone — this
  only becomes feasible on the synthetic-pretraining path of the encoder–decoder design
  (grammar-generated expressions, reconstruction + e5-alignment objectives).
- Current art references: `process_cards.py` (P0 — canonicalization/token identification only,
  no embeddings composed today), `DESIGN_process_expression_implementation.md` P3 (deterministic
  superposition), `ARCHITECTURE_filing_engine.md` OPENQ-012 (cross-corpus transfer — the
  evaluation any position scheme ultimately serves).
