# Does `1/d` earn its keep in the calibrated joint posterior? — the honest test

> **SCOPE CORRECTION (user, 2026-07-05).** This test used the **LLM's argmax relation as the target** and asked
> whether `1/d` beats e5 at *predicting the LLM label*. That is **not the SYM judge**. The SYM judge is a
> *constructed superposition* — `judge = e5 ⊕ confidence_weighted(1/d ⊕ asymmetric memberships)` — and the
> model's loss is against **that blend**, in which `1/d` is a **constituent** (weighted by the superposition
> ratio), not a competitor. So "`1/d` is redundant with e5" here means only "redundant *for predicting the LLM
> relation label*"; it says **nothing** about `1/d`'s role in the constructed judge. Two consequences: (a) the
> `corr(1/d, e5)=+0.50` is a *positive* signal — `1/d` captures semantic distance while tracking our graph more
> closely; (b) the loss-relevant validation of `1/d`+graph is the *original dual-judge finding* (LLM SYM-
> **relatedness** as validation ground truth → DUAL(graph⊕e5) +0.75 vs e5 +0.60), which this does not overturn.
> **What still stands unambiguously: the JOINT head beats the factored PoE.** The "`1/d` doesn't earn its keep"
> headline is retained below only for the narrow LLM-relation-classification framing.


*Result of the `DESIGN_sym_estimation_integration.md` build plan (steps 1–3): add `1/d` to the JointPosterior
source vector, fit on held-out, ablate with vs without. Data = 880 Wikipedia category pairs, both endpoints in
the struct embedding, LLM-labelled (gpt-5.5-low, §14 rubric; argmax relation = the classification target). Model
readouts from `model_prod.pt` with ancestors resolved on 100k_cats. 2026-07-05.*

## Headline

1. **The JOINT head beats the factored product-of-marginals — confirmed (#3359).** Much better *calibrated*:
   log-loss **1.076** vs **1.55** (equal-weight PoE) / **1.26** (separability-weighted); ECE **0.085** vs
   **0.16 / 0.12**. Accuracy comparable (~62–65%). The course-correction's core — a learned calibrated combiner
   over the correlated vector beats naive independence — holds on this data.
2. **`1/d` does NOT earn its keep.** Refuted on both splits by the pre-registered criterion (with-`1/d` AURC CI
   must sit below the without-`1/d` point estimate):

   | split | held | without `1/d` AURC(margin) | with `1/d` AURC(margin) | acc Δ | ECE Δ |
   |---|---|---|---|---|---|
   | node-disjoint | 60 | 0.169 [0.091, 0.289] | 0.176 [0.095, 0.305] | 0.0 | +0.055 |
   | random | 220 | 0.183 [0.135, 0.241] | 0.179 [0.132, 0.235] | −1.4 | +0.011 |

   The AURC intervals massively overlap; accuracy/ECE are flat-to-slightly-worse with `1/d`. No earn.

## Why — `1/d` is redundant with e5 here

- **Correlation matrix:** `dist` correlates **+0.50 with e5**, +0.46 with `sym`, +0.32 with `elem_fwd`. On
  Wikipedia *categories*, graph distance ≈ topical similarity ≈ e5, so `1/d` largely re-reads e5.
- **Separability 0.041** — barely above e5's 0.037; it discriminates relations weakly.
- The joint head *does* give `dist` a small positive weight for the lateral relation (`assoc` dist +0.06…+0.18)
  — consistent with the sibling intuition — but it's not enough to improve the estimate over e5+readouts.
- Reconciles with the earlier sibling finding (`corr(1/d, SYM)=+0.136` on membership-free pairs): that signal
  was **real but not *additional* to e5**, which already covers the lateral/semantic axis the vertical
  membership operators miss. `1/d`'s unique contribution was over the *memberships*, not over *e5*.

## Caveats
- Target is the **gpt-5.5-low** argmax relation (one judge). `see_also` was tiny (6); the LLM lumped siblings
  into `assoc` (161), so the lateral class = `assoc`. A different judge / a graded (not argmax) target could
  shift it.
- Held sets are small (node-disjoint drops 328/880 cross-split pairs → 60 held; random → 220). AURC CIs are
  wide; this establishes "no clear win," not a razor-sharp null.
- The struct embedding is 48-d on 100k_cats; a higher-fidelity / less-e5-correlated distance *might* change the
  verdict. But the redundancy-with-e5 mechanism is robust across both splits.

## What this validates (the methodology, even though the hypothesis lost)
The workflow did its job: it took a **conceptually-motivated source** (`1/d`, the sibling/lateral axis) and,
via the calibrated joint head + a with/without ablation + AURC/ECE on held-out data, showed it is **redundant
with e5** rather than assuming it helps. That is exactly what `DESIGN_uncertainty_estimation_playbook.md` is for.
The joint-head-beats-PoE result stands as the reusable win.

Run: `UW_MU_GRAPH=…/100k_cats/category_parent.tsv python3 mu_posterior.py --pairs wiki_rel_graded_pairs.tsv
--e5-cache wiki_rel_e5.pt --model model_prod.pt --struct-emb struct_emb_recip.pt --split node-disjoint --boot 500`.
Data pipeline: `gen_wiki_relation_pairs.py` → `score_with_codex.py` (gpt-5.5-low) → `convert_scored_to_graded.py`.
