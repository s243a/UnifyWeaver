# Expression Encoder–Decoder (FUTURE WORK — do not schedule before the P1 gate)

**Status: future-work design capture, 2026-07-24.** Not part of the current P0–P4 ladder and not
to interrupt the frozen P1 protocol (#3982's rigor contract). Sits at "P3.5": pursue only if
P1 passes (expression conditioning earns its keep) AND P3's deterministic composition plateaus.
Self-contained enough to hand to a separate agent with spare inference; the handoff needs only
this doc, the companion `DESIGN_tree_position_encoding_theory.md`, and the current-art references
below.

## Current art (what exists today, and its limits)

- `DESIGN_process_expression_{language,philosophy,implementation}.md` + `process_cards.py` (P0,
  PR #3980): the process-expression grammar with registry-driven lexing. **Today the grammar is
  used ONLY for canonicalization and token identification** — lossless identity (canonical AST +
  ast_sha) and deterministic card rendering (V0–V3). "Self-parsing" in #3980 means only that the
  parser must parse the spec's own examples (acceptance test), not a formal novelty.
- Embedding today: rendered card string → frozen e5 → NameFunctionCond residual. The
  implementation doc flags e5-on-formal-strings as a leap of faith at V2/V3; P2 adds an
  NL-template arm to test it.
- P3 (planned): DETERMINISTIC composition — node embedding = card-e5(operator) + Σ fixed
  position-weighted child embeddings; stochastic elements seeded by SHA-256 of the canonical AST.
  No learning anywhere in the composition.
- `ARCHITECTURE_filing_engine.md` rows this feeds: FUSE-003 (name-conditioned identity — the
  motivation), OPENQ-012 (cross-corpus transfer of process conditioning — the forest question).

## The proposal (owner's design, from the 2026-07-24 Perplexity/Sonnet-5 discussion)

A small transformer over the PARSED, TOKENIZED, CANONICALIZED expression, trained with a dual
objective:

```
canonical AST ──tokenize──▶ transformer encoder ──▶ latent z
                                        ├─▶ MLP alignment head  → ê5(card)   (match frozen e5 of the rendered card)
                                        └─▶ cotrained decoder   → canonical string (reconstruction)
```

1. **Alignment head:** keeps z anchored to e5's semantic space — the μ-stack's conditioning
   pathway (NameFunctionCond's W) continues to work unchanged, consuming ê5 instead of e5.
2. **Reconstruction decoder:** forces z to preserve the compositional structure e5 may collapse
   (the V2/V3 leap-of-faith risk). Structural echo worth preserving in any implementation: the
   decoder enforces exactly the language spec's LOSSLESS-IDENTITY vs LOSSY-CARD split — z is the
   identity, the alignment head's output is the card.
3. **Variant:** CLIP-style contrastive training between (formal-expression encoding, NL-template
   rendering fed to e5) instead of a fixed MLP map — learns WHICH structural distinctions are
   semantically meaningful to e5 rather than assuming a mapping.

## Why training cost is smaller than it looks

The encoder/decoder pretrain ENTIRELY SYNTHETICALLY: the grammar + registry generate unlimited
valid expressions (combinatorial composition of operators, judges, kwargs); e5-small alignment
targets cost one cheap inference pass per sample. Judge labels are needed only for the downstream
μ-head fine-tune, which is the small real ledger P1 already uses. The expensive open question is
coverage: how many synthetic samples before the compositional space is dense enough that unseen
real expressions interpolate.

## Why this could matter (the generalization argument)

e5 generalizes over strings by surface similarity; a trained expression encoder generalizes over
PROGRAMS by structure. If OPENQ-012's cross-corpus arm shows expression conditioning transfers,
this encoder is the natural next step — compositional generalization to unseen process
combinations (`kalman(sonnet.D, luna.S)` never seen, both parents seen) is precisely what the
reconstruction objective buys and what surface-string e5 cannot.

## Gates and sequencing (explicit, to protect the current lanes)

1. P1 frozen primary must pass (expressions vs flat tokens). If it fails, this direction is moot.
2. P3 deterministic superposition must be tried first; this design activates on plateau.
3. The position-encoding component is specified separately in
   `DESIGN_tree_position_encoding_theory.md` (depth⊗breadth factorization; Kronecker vs
   convolution; learned bilinear recommendation) — the encoder should adopt whichever variant
   that doc's ablation ladder selects.
4. Evaluation inherits the amended P3 protocol: multi-process LOCO + cross-corpus LOCO
   (OPENQ-012), against the four controls (frozen-string e5, additive bag-of-nodes, flat token,
   unconditional) — plus one new control: NL-template e5 (P2's arm), which is the strongest
   no-new-architecture baseline this design must beat to justify its cost.
