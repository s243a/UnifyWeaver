# Targeted page supplementation + the quality×diversity selection question

Two things in one round: (a) act on the **data-gap policy** — supplement *model-weak, category-thin,
in-region* nodes with page data; (b) answer **"what fraction of a category's pages to keep?"** with an A/B
instead of a guess.

## Round: 10 finder-selected gap categories

`find_data_gaps.py` ranked systems/dynamical-region nodes by THIN (subcats≤3) ∧ WEAK (low e5
domain-margin) and surfaced targets that the trained model also ranks worst (it independently flagged
`Ergodic_theory`, the stratum stuck at +0.25 across every config). Harvested 10 — `Linear_filters`,
`Wavelets`, `Linear_programming`, `Non-equilibrium_thermodynamics`, `Packing_problems`,
`Network_flow_problem`, `Model_checkers`, `Filter_frequency_response`, `Organizational_cybernetics`,
`Bargaining_theory` — 326 member pages, Haiku-graded for centrality (element-of), tagged
`relation=element_of` (`mu_pairs_scored_pages_gaps_*.tsv`). Retrained at 3 layers.

The new categories rank well (`linear_programming` +0.77, `linear_filters` +0.70, `wavelets` +0.66), and
the policy paid off on the weak node: **`ergodic` +0.25 → +0.31–0.43** — supplementing a model-weak
category with page data improved it. ELEM held-out corr also rose (+0.575 → +0.620) — the operator is
data-hungry; more page examples help it generally.

## Selection A/B — keep ALL vs a diverse subset

Question (user): keep all pages? best-by-Haiku? random? The intuition — **best ranks + e5 diversity**, with
SVD/DPP as the tool. Built `select_diverse.py`: drop μ<0.4 junk, then greedy **quality-weighted
farthest-point** selection (cheap DPP-MAP proxy) per category — keep count = category's intrinsic e5
diversity, not a fixed %. It pruned 326 → **183** (dense cats collapse: `Linear_filters` 66→37,
`Wavelets` 54→30). Trained all-326 vs diverse-183 at 3L, evaluated on the **same** held-out:

| stratum | all-326 | diverse-183 |
|---|---|---|
| `linear_filters` | **+0.70** | +0.12 |
| `non-equilibrium` | **+0.55** | +0.30 |
| `wavelets` | **+0.66** | +0.45 |
| `linear_programming` | **+0.77** | +0.65 |
| `packing` | +0.64 | **+0.89** |
| `ergodic` | +0.31 | **+0.43** |
| ELEM held-out | **+0.620** | +0.541 |
| overall | +0.853 | +0.845 |

**Conclusion: at this scale, keep all (prune only μ<0.4 junk).** All-326 wins most strata and the ELEM
held-out; diverse wins only the already-diverse `packing` and the hard `ergodic`.

Two honest caveats on *why* this isn't a refutation of the diversity idea:
1. **Eval bias toward "all":** held-out = 20% of *all* pages, so for dense categories it is full of the
   near-duplicates the diverse model dropped → the all-model, having trained on similar pages, predicts
   them better. That inflates all's edge exactly on the dense strata. The unbiased signals (overall +0.853
   vs +0.845; ELEM +0.620 vs +0.541) say all is *modestly* better, not dramatically.
2. **The lesson:** "redundant" filter pages aren't redundant for centrality (Butterworth vs Chebyshev vs
   Bessel differ), the ELEM operator is data-hungry (~300–500 example pool), and frozen e5 means it *can't*
   overfit duplicates — they're just more supervision on the manifold. Quality×diversity pays off at
   **larger scale** (redundancy + compute bite) and needs a **disjoint held-out** (excluding pruned pages)
   to measure fairly — not at few-hundred-page scale where quantity dominates.

**Method, precisely (documenting what was actually run).** `select_diverse.py` is **greedy
quality-weighted farthest-point** on e5 cosine (drop μ<0.4, seed highest-Haiku, then add the page
maximising `centrality·(1 − max cos to picked)`) — a cheap **DPP-MAP proxy, NOT a PCA/SVD decomposition**.
PCA/SVD was the *suggested* framing. It is now **implemented as `select_svd_coverage.py`**: take the top-K
SVD axes of the page-embedding matrix and keep a subset covering the **μ distribution along each axis**
(both ends × low/mid/high μ), ensuring enough negatives — coverage of the *(e5-axis × μ)* joint, with the
kept-count set by the SVD **variance elbow** (intrinsic dimension) rather than a fixed `--frac`. It adds a
**sufficiency threshold** (`--min-pages`: keep a category whole below it) and pairs with **subcategory
augmentation** (`fetch_category_pages.py --recurse-subcats`) to push a thin category over that threshold.
Both selectors remain the larger-scale "must filter" knob; not used at current scale.

**Page-sampling policy (the operative conclusion).** There is far more page than category data, but because
page-membership trains on its **own `ELEM` operator**, the imbalance is *contained* in `ELEM` rather than
drowning the category operators — so **don't filter page data for `ELEM`; use all**. Instead, **monitor**
whether growing `ELEM` learning degrades the other operators (shared-trunk interference), and if so
**throttle `ELEM`'s gradient** (`--elem-weight`, or a per-operator learning rate / reduced backprop — not
yet implemented) or **add capacity** (the 3rd layer already removed the interference) — rather than cutting
the data. Diversity-pruning (`select_diverse.py`) is reserved for the larger-scale "must filter" regime.

`select_diverse.py` is kept as the knob (`--frac`, `--min-mu`) for that larger-scale regime. The **full**
326-page set is folded into the cumulative (25,651 rows). Reproduce: `find_data_gaps.py` →
`fetch_category_pages.py` → Haiku centrality → `gen_page_pairs.py`; A/B via two `train_mu_attention.py
--layers 3` runs on cumulative+all vs cumulative+`select_diverse.py` output, eval on the all-pairs split.
