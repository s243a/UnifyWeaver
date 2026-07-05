# Truncated-λ blends ⇒ a more judge-INDEPENDENT generality

*Follow-up to the blend-judge work (`REPORT_blend_judge_sweep.md`). Tests whether **varying λ** in the training
blend (truncated-normal, mean 0.5, std 0.15, resampled to [0,1]) changes what the model learns, vs a **fixed
λ=0.5**. Metric: predict the held-out judge superposition `T` (`eval_blend_prediction.py`), read **with** the
`judge=blend` input and **agnostically** (no judge input). 3 seeds each, fine-tuned from `model_prod`.*

## Result (corr(SYM, T) at λ_eval=0.5)

| training | agnostic (judge-independent) | blend | **judge-input gap** | agnostic/blend |
|---|---|---|---|---|
| fixed-λ=0.5 (s1/s2/s3) | .787 / .713 / .799 → **.766** | .847 / .832 / .845 → **.841** | **+0.075** | 0.911 |
| truncated-λ (s1/s2/s3) | .818 / .743 / .778 → **.780** | .848 / .806 / .821 → **.825** | **+0.045** | **0.945** |

## Two findings (both hold on all 3 seeds)

1. **Varying λ ⇒ more robust to the judge input.** The gap between reading *with* vs *without* the `judge=blend`
   tag **shrinks** from +0.075 (fixed) to +0.045 (truncated) — per-seed 0.060/0.119/0.046 → 0.030/0.063/0.043.
   The truncated model relies on the judge tag *less*.
2. **⇒ the generality is judge-INDEPENDENT.** The fraction of the superposition signal that survives *without*
   the judge input (agnostic/blend) rises from 0.911 to **0.945** (per-seed 0.929/0.857/0.946 → 0.965/0.922/0.948)
   — more of what the model learned lives in the **shared trunk**, not the judge head. (User's framing, verified:
   *"if we vary λ the model becomes more robust to judge input, and finds a generality that is judge-independent."*)

## The trade-off (honest)
Truncated's **peak** (blend-read) is marginally *lower* — mean +0.825 vs +0.841 (fixed higher on 2/3 seeds). So
randomness buys **robustness / judge-independence, not peak accuracy**: spreading the blend family pushes the
signal into the trunk (general, judge-agnostic) and makes the per-judge head less sharp. Which you want depends
on the use: fixed-λ if you'll always supply the right judge tag; truncated-λ if you want a model that behaves
well **without** the tag (robust deployment, or as a regulariser toward judge-independent structure).

## Caveats
- `T` is graph-dominated on this held-out set (`e5_ref` std 0.032) — confirms the structural half most strongly.
- The *absolute* agnostic level barely moves (+0.766→+0.780, mixed across seeds); the robust, all-3-seed effect
  is the **gap-shrinking / judge-independence**, not a peak gain.
- Single held-out set, one λ-distribution (mean 0.5 / std 0.15); a std sweep would map how far the effect goes.

Repro: `emit_blend_judge.py --lam-dist truncated --lam 0.5 --lam-std 0.15` → combined round → fine-tune ×3 seeds →
`eval_blend_prediction.py --lam-eval 0.5`.
