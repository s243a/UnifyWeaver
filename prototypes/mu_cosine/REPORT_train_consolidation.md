# Consolidation train + eval — the full cumulative (cyber/systheory/page/pearltrees rounds)

First training run since `model_mathfields.pt`: everything gathered this sprint (stats, sysinfo, alggeom,
boundary, cyber, systheory, page-frontier, pearltrees) had accumulated **untrained**. Fine-tune-with-replay
from `model_mathfields.pt` on the full cumulative (24.7k rows), replay = prior (eng+enwiki), `--lr 1.5e-4
--steps 500`, graph = `wide_enwiki_math`. A **placebo** (same warm start + replay, but `--pairs = prior`, no
new data) is the churn control. Both evaluated on the **same** cumulative held-out split (1,135 positives).

Setup fix (committed): `train_mu_attention.py` and `eval_per_stratum.py` now **union every `--pairs`/
`--replay`/LLM node into the e5 build**, so cold-start endpoints (cross-slice, page, pearltrees nodes) get
a frozen e5 embedding instead of being silently dropped by the `in idx` filters.

## Headline (fine-tune vs placebo, same held-out split)

| metric | fine-tune (`model_all`) | placebo (`model_all_placebo`) | read |
|---|---|---|---|
| overall held-out SYM corr | **+0.837** | +0.811 | new data adds ~+0.026 |
| multi-domain discrimination | 35/39 (90%) | 34/39 (87%) | saturated; +1 = noise |
| top-2 discrimination | 100% | 100% | — |
| WIKI edge order-acc | 99.8% | 99.9% | no forgetting |
| gate-leak (SYM / WIKI) | 0/5 | 0/4 | no leak |

So at the **category** level the run confirms the whole arc's pattern: **cross-domain discrimination is
saturated** (held, not improved), and the new category data's contribution to **within-field ranking is
modest and partly churn** (+0.026 over placebo). The misses are all genuine multi-membership
(Mechanics→Engi, Machine_learning/Neural_networks→Comp, Thermodynamics→Chem).

## The real finding — page-membership is NOT absorbed as SYM

Per-stratum held-out corr, fine-tune vs placebo, **on the same split** (both models; placebo never trained
on page data):

| stratum | fine-tune | placebo | n |
|---|---|---|---|
| `pos_network_theory_down` | +0.97 | — | 5 |
| `pos_chem` / `pos_modphys` / `pos_foundations` | +0.91 / +0.91 / +0.92 | — | 49 / 22 / 14 |
| `pos_dynamical_syst_down` | +0.85 | — | 17 |
| **`pos_pageof_ergodic_theory`** | **+0.39** | **+0.50** | 14 |
| **`pos_pageof_nonlinear`** | **+0.12** | −0.05 | 7 |
| **`pos_pageof_emergence`** | **+0.31** | +0.30 | 15 |
| **`pos_pageof_systems_analysis`** | **+0.68** | +0.58 | 14 |
| **`pos_pageof_holism`** | **+0.63** | +0.52 | 8 |

Two things stand out:

1. **Page strata rank far worse than category strata** (+0.0–0.68 vs +0.85–0.97), *despite* having clean
   Haiku-graded targets — the model can't capture page-centrality the way it captures category membership.
2. **Training on the page data barely moved page-centrality over the placebo that never saw it**
   (fine-tune ≈ placebo per page stratum; placebo is even higher on ergodic). So the page signal, fed as
   undifferentiated SYM positives, is **essentially inert** — the model is reading whatever it gets from
   frozen e5, not learning from the page labels.

This is the empirical case the design predicted (`DESIGN_calibrated_judges.md` §7): **page-membership is a
different relation** (article-is-about-topic) than subcategory membership (subfield-is-part-of-field), and
conflating them in one `SYM` operator wastes the page labels. (Caveat: page held-out sets are small,
n=3–15; the per-stratum numbers are noisy, but the direction is uniform across all six and corroborated by
fine-tune≈placebo.)

## Conclusion → what this argues for next

- The **category vein has plateaued**: discrimination saturated, ranking gains churn-dominated. More
  category rounds will keep returning ~this.
- The **page/pearltrees data is currently under-used** — it trains but doesn't teach, because it has no
  distinct relation operator. The highest-value next lever is therefore **implementing the element-of
  operator + node-type token** (design §7), then re-running this exact comparison: the prediction is that
  page-centrality corr jumps once the page labels train as their own function instead of riding SYM.

Artifacts (gitignored `*.pt`): `model_all.pt`, `model_all_placebo.pt`, `e5_tables_train_all.pt`.
Reproduce: `UW_MU_GRAPH=…/wide_enwiki_math/… UW_E5_CACHE=e5_tables_train_all.pt train_mu_attention.py
--pairs mu_pairs_scored_cumulative.tsv --replay-pairs mu_pairs_scored_prior.tsv --init-from
model_mathfields.pt --pairs-corpus enwiki --lr 1.5e-4 --steps 500 --save model_all.pt` (+ placebo with
`--pairs mu_pairs_scored_prior.tsv`), then `eval_per_stratum.py --model model_all.pt --pairs
mu_pairs_scored_cumulative.tsv`.
