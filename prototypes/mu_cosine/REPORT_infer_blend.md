# Infer-blend — random operator embedding from the joint posterior (the design, applied)

Wiring the fitted joint `P(relation | μ_vec)` into the trainer as a **random operator embedding** for inferred
rows (`--infer-blend`), per `DESIGN_inferred_operator_superposition.md` §5b. This is the payoff increment:
the estimator → model-input pipeline running in the training loop.

## What it does (§5b, the #3359-corrected spec)
- **Enabler:** `MuAttention.forward(op_weights=…)` — a blended operator overriding the op token *and* the
  readout head (one-hot ≡ indexed, Δ=0; gradients reach `op_emb`/`readout_w`).
- **Curriculum:** tagged-only until `--blend-warmup`, then the blend turns on (a warm-up gate, not a switch).
- **Refresh** (every `--blend-refresh`, stop-grad target-network): measure the 6-readout vector with the
  *current* model, fit `JointPosterior` on tagged rows, predict for inferred → **operator-marginal** `P(op)` +
  a **relation-level direction-specific** blended target `Σ P(rel)·target_dir(rel)` (so SYM's bridge/see_also/
  assoc, which share an operator but differ in μ, are not collapsed).
- **Per step:** `op_weights ~ Dirichlet(α·P(op))` on an **isolated RNG** → blended forward → MSE to the
  blended target.

## A/B (warm-start `model_nodetype.pt`, graded6, 700 steps, bs 128, seed 1, clean isolated-RNG harness)

| arm | discrimination | SYM held-out | WIKI order-acc | ELEM corr |
|---|---|---|---|---|
| no-switch | 89% (32/36) | +0.830 | 99.8% | +0.698 |
| v1 fixed-breadth switch | 94% (34/36) | +0.838 | 99.8% | +0.702 |
| **infer-blend (posterior)** | **94% (34/36)** | **+0.839** | **99.9%** | **+0.706** |

## Honest verdict
The principled posterior-driven blend **matches the v1 heuristic** on the headline metric (94%, +5 over
no-switch) with only a **marginal** edge on SYM/ELEM. At this scale it does **not** dramatically outperform
the cheap fixed-breadth switch. What it buys is **not** a big number here but:
- **Generality** — v1 only switches `element_of→subcategory`; the blend handles *every* inferred relation via
  `P(relation|μ_vec)`, including the see_also↔membership reconsideration μ enables.
- **Principle** — the operator is a calibrated random superposition (the joint head beats the corrected PoE,
  PR #3359), with the noise decomposition's knobs (α, out-of-set) instead of a hand-tuned breadth rule.
- **Headroom** — only **350** inferred rows here (mostly typo'd `Subtoipcs` fallbacks, which are *really*
  element_of). The blend should matter more with **more diverse inferred data** (more fused mindmaps, the
  fuzzy/LLM section categoriser surfacing genuinely-ambiguous relations) — that is where a fixed
  element→subcategory rule breaks and the posterior earns its keep.

So: the design works end-to-end and is at parity with the heuristic now; its value is the principled,
general framework that scales with data diversity, not a win at this (small, low-diversity) inferred set.

## Next
- Grow the inferred set (more mindmaps; fuzzy/LLM section categorisation) and re-measure — the regime where
  the blend should separate from v1.
- Tune α / out-of-set noise; try `--blend-hidden` (MLP joint head) vs LR.
- Deterministic-mean blend at inference (already specified; the eval path uses one-hot operators).
