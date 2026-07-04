# SYM is a dual judge: superposition of e5-semantic ⊕ graph-structural

Empirical finding (`fit_sym_from_graph.py`, on 1500 judge-scored SYM positives, `model_prod.pt`, 100k_cats graph).

## Result

Fit `μ_sym ≈ σ(w·features)` against the judge-scored SYM targets:

| features | R² | corr |
|---|---|---|
| **graph judge** `3/dist` (inverse-radial, `1/r`) | +0.43 | +0.66 |
| **e5 judge** (the model's SYM operator readout) | +0.34 | +0.60 |
| **DUAL** `3/dist` ⊕ e5-SYM | **+0.57** | **+0.75** |
| struct-embed only (16–64 landmark BFS, `3/‖Δvec‖`) | +0.14 | +0.40 |
| DUAL (struct-embed ⊕ e5-SYM) | +0.39 | +0.63 |

## What it means

- **SYM is a superposition of two judges, and they're complementary.** The dual (+0.75) beats *either* alone
  (graph +0.66, e5 +0.60) by a real margin — they capture different residuals. Not "e5 OR graph"; **e5 ⊕ graph.**
- **The graph judge = `3/dist`** (inverse-radial `1/r`, the power-law variant in `COST_FUNCTION_PHILOSOPHY.md`),
  and it **dominates the directionals** — `μ(a|b)`, `μ(b|a)` add ~nothing (recip-only ≈ full fit). The residual
  form `σ(3/dist − μab − μba)` is *order*-correct (corr +0.61) but needs one fitted scale on `3/dist` for
  calibrated magnitude (R²<0 unfitted).
- **Graph distance alone beats the trained e5 SYM operator** (+0.66 vs +0.60/+0.41-baseline) — the structural
  signal is not something e5 already knows. This is why a **pure-e5 distillation of SYM is lossy**, and it
  **re-explains the SYM "regression"**: SYM is ~half structural, so when the broad fine-tune shifted the graph,
  the pure-e5 operator lost the structural half it never had.

## Architecture

`μ_sym = blend( judge=graph [3/dist], judge=e5 [SYM operator] )` — exactly the **calibrated-judges /
operator-superposition** machinery (`judge_emb`, the blend regularizer). SYM is its first real use.
- **Inference:** e5 is free (model input); the graph half must be a **cheap O(1) structural signal**, not a BFS.
- **Structural embedding (step 2, `structural_embedding.py`):** landmark heuristics fail (corr ~+0.33). A
  learned metric embedding trained on the **RECIPROCAL target** `3/d` (not distance — far pairs' small `3/d`
  auto-weighs down, so capacity goes to the near scale where SYM lives; user's insight) recovers +0.405 corr
  with true `3/dist` on the SYM pairs → **DUAL(struct-embed ⊕ e5) corr +0.652**, beating e5-only (+0.60) but
  below the raw-BFS ceiling (+0.76). So the O(1) signal adds real value; residual fine-scale loss remains
  (higher dim / better training / the exact scoring graph are the levers).

## Step 3 — wiring the dual judge into the model (`mu_attention.py`)

The finding is a **linear blend of one scalar (`3/dist`, now the O(1) struct-embed proxy) and the e5-SYM
readout**. Step 3 bakes that blend into the model as a **learnable, SYM-gated structural channel** — the
faithful, minimal operationalisation of the measured dual judge:

```
logit_SYM = pooled·w_SYM + b_SYM  +  sym_struct_w · struct_feat        # struct_feat = 3/(1+‖Δ struct-emb‖)
```

- **`self.sym_struct_w` (scalar, zero-init)** — the learned scale on the structural channel. Zero-init ⇒ an
  **exact warm-start no-op** (unit-tested: identical μ to the no-struct path), so it's safe to add to any
  checkpoint; the scale is learned only during SYM training.
- **SYM-gated** — added to the logit *only* where `op == SYM` (via `op_of`, or `op_weights[:,SYM]` in the
  blended path). Unit-tested: perturbing the scale moves the SYM row and leaves HIER/others untouched.
- **`struct_feat`** — computed in `Tokenizer.build` from a `{name: struct-emb vec}` table (`--struct-emb`,
  the `structural_embedding.py` `.pt`). Off by default (no table ⇒ key omitted ⇒ old behaviour byte-for-byte).
  On the cumulative SYM pairs, **57 %** have both endpoints in the table (the rest fall back to pure e5,
  `struct_feat = 0` — the same graceful degradation as inference on out-of-graph pairs).

**Why the pairwise scalar (not a per-endpoint struct token).** The validated finding is a function of the
*pairwise* distance, so injecting the single scalar `3/(1+‖Δ‖)` into the SYM logit reproduces the +0.652 dual
**by construction** and is O(1). A per-endpoint structural token (project each node's struct-vec into `d_model`,
let attention re-derive the distance) is a strictly more ambitious generalisation — deferred as a follow-up; it
*might* exceed +0.652 but is not what the measurement supports, and it costs a projection + more plumbing.

**Validation status.** The *value* of the channel is already established (step 2: DUAL +0.652 > e5-only +0.60).
The *wiring* is unit-verified (no-op at warm start, SYM-gated). The remaining step — an **end-to-end SYM retrain
with `--struct-emb`** confirming a trained model realises the lift — must use the full production recipe that
produced `model_prod.pt`; a throwaway `--sym-only --steps 600` warm-start collapses to μ=0 (a known,
struct-independent failure of that reduced recipe, not of this change). Launch with the real recipe + a held-out
SYM-corr A/B (`--struct-emb` on vs off), multi-seed before believing.

## Caveats
- R²≈0.43 for the graph half is *moderate* — `3/dist` explains ~half of SYM; the rest is semantic nuance +
  judge noise + the fact that 100k_cats may not be the exact graph the pairs were scored on (true corr could be
  higher on the right structure).

See `DESIGN_calibrated_judges.md`, `COST_FUNCTION_PHILOSOPHY.md`, `fit_sym_from_graph.py`.
