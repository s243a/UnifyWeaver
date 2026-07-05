# Does the constructed blend judge teach generality? — multi-seed test

*Tests the hypothesis (user, 2026-07-05): training on judge *superpositions* teaches generality, so base
held-out (never touched by the blend pairs) should rise. 3 arms × 3 seeds, fine-tuned from `model_prod` (800→600
steps, full recipe, `--pairs mu_pairs_scored_cumulative` for the base SYM held-out, `UW_MU_GRAPH=100k_cats`).
Metric: **base simplewiki SYM held-out corr** (1074 positives, disjoint from the Wikipedia blend pairs).*

## Arms
- **A `LLM only`** — graded round = the 880 Wikipedia pairs under `judge=gpt-5.5-low` (no constructed blend).
- **B `fixed-λ`** — A + 880 `judge=blend` SYM rows, `blend = 0.5·μ_e5_sym ⊕ 0.5·P(SYM|1/d,asym-ops)`.
- **C `random-λ`** — A + 880 `judge=blend` rows with `λ ~ U(0,1)` per pair (the blend regulariser).

## Result

| arm | seed 1 | seed 2 | seed 3 | mean |
|---|---|---|---|---|
| A `LLM only` (control) | +0.792 | +0.788 | +0.789 | **+0.790** |
| B `fixed-λ blend` | +0.790 | +0.755 | +0.815 | +0.787 |
| C `random-λ blend` | +0.785 | +0.782 | **+0.092** | +0.553 |

*(model_prod, no fine-tune: +0.41. Within-seed blend−control: seed1 −0.002/−0.007, seed2 −0.033/−0.006, seed3
+0.026 / **−0.697 collapse**.)*

## Verdict — the generality-via-constructed-blend hypothesis is NOT supported

1. **The base-SYM lift is real and multi-seed-stable (+0.41 → ~+0.79) — but it is the DATA + fine-tune, not the
   blend.** The control (`A`, no constructed blend) matches the blend arms (+0.790 vs +0.787). Fine-tuning on the
   LLM Wikipedia round — which itself spans multiple judge tags (`gpt-5.5-low`, and the base pairs' `haiku`/
   `graph`) — is what generalises the base SYM.
2. **The constructed blend judge adds nothing over the LLM data** (B−A ≈ 0 across seeds).
3. **Random-λ is risky** — seed 3 collapsed to +0.09. Forcing a per-pair random blend can destabilise training.

## Takeaways
- **Multi-seed caught a wrong single-seed story:** the +0.781 first-run lift was real *as a lift* but *not*
  attributable to the superposition. This is exactly why single-seed deltas aren't believed.
- **Data/judge diversity may buy the generality for free** — you may not need a hand-constructed blend judge;
  training across several real judge tags already spans views.
- **If you do use a constructed blend, it must beat a no-blend control across seeds** before you credit it, and
  **random-λ needs a stability guard** (warmup, λ schedule, or a floor).
- Not overturned: the *estimator* wins from earlier (joint head > PoE), and the base fine-tune's +0.79 SYM is a
  genuine, reproducible improvement over model_prod's +0.41 — just not for the reason hypothesised.

Data: `gen_wiki_relation_pairs.py` → `score_with_codex.py` → `convert_scored_to_graded.py` + `emit_blend_judge.py`.
Sweep: `/tmp/mu_data/blend_sweep.sh`.
