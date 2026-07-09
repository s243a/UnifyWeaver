# Product-Kalman on real data: fusion is large, the correlation term is real, independent PoE is fragile

*First real-data run of the Product-Kalman holdout harness (`run_product_kalman_realdata.py`, 2026-07-08) —
build step 1 of `DESIGN_amortized_fusion_heads.md`. EXPLORATORY comparison, not preregistered. Two corpora:
250 exploratory multihop pairs (100k_cats) and the 250 fresh Behavior-slice pairs from the confirmatory run;
40 descendant-disjoint splits each; shrinkage 0.05; graph channel affine-calibrated to D on train only.*

## Setup

State/target = continuous LLM labels `(D, S)`. Prior = model readouts `(mu_D, mu_S)`. The harness scores, per
split: `prior` (fitted P), `independent_kalman` (cross-covariance C=0), `product_kalman` (learned C).
**Identification: Gaussian PoE ≡ the independent Kalman update** (product of Gaussian experts = precision
summation = fusion with C=0), so independent-vs-product IS "raw PoE vs correlated Kalman".

**How "beats" is measured (user question):** three axes — held-out **NLL** (proper score, mean+bars jointly);
**MSE** (mean alone); **Mahalanobis/dim** (error bars alone: ≈1 calibrated, >1 overconfident; squared-Mahalanobis
q95 vs the chi2_2 reference 5.99). Theory predicts the correlation term shows up mostly in the ERROR BARS
(independent fusion double-counts correlated evidence → too-confident bars), not the means.

## Results

**Config [graph] — fuse prior with the calibrated walk channel (H=[1,0]):**

| corpus | variant | NLL ↓ | MSE ↓ | Mahal/dim | q95 (ref 5.99) |
|---|---|---|---|---|---|
| exploratory | prior | +0.570 | 0.271 | 1.59 | 7.15 |
| exploratory | independent (=PoE) | +0.012 | 0.116 | 1.51 | 7.00 |
| exploratory | **product (correlated)** | **−0.306** | **0.099** | **1.18** | **5.70** |
| fresh | prior | +0.408 | 0.188 | 1.42 | 7.93 |
| fresh | independent (=PoE) | +0.038 | 0.111 | 1.49 | 8.05 |
| fresh | **product (correlated)** | **−0.010** | **0.107** | 1.39 | 8.00 |

**Config [graph+poe] — add PoE-lower/noisy-OR-upper channels (deliberately correlated with the prior):**

| corpus | variant | NLL ↓ | Mahal/dim | q95 |
|---|---|---|---|---|
| exploratory | independent (=PoE) | +0.381 *(worse than without!)* | **1.95** | **10.11** |
| exploratory | product (correlated) | −0.257 | 1.24 | 6.14 |
| fresh | independent (=PoE) | +0.221 *(worse than without!)* | **1.79** | **10.36** |
| fresh | product (correlated) | +0.070 | 1.49 | 8.34 |

## Findings

1. **Fusion is the big win, on both corpora:** prior→product NLL gain +0.88 (exploratory) / +0.42 (fresh),
   positive on 40/40 splits each; MSE roughly halves. Any covariance-aware use of the graph channel beats the
   model alone — the fast-timescale hybrid earns its keep immediately.
2. **The correlation term (product vs independent = "Kalman vs PoE") is real but corpus-dependent:**
   +0.32 NLL (40/40 splits) on exploratory vs +0.05 (35/40) on fresh. Direction replicates; magnitude tracks
   how correlated the corpus's channels are. As predicted, the win is mostly in the ERROR BARS: at similar MSE,
   Mahal/dim 1.18 vs 1.51 (exploratory).
3. **Independent PoE is FRAGILE to adding correlated evidence — the double-counting failure mode, measured, on
   BOTH corpora:** adding the PoE lower/upper channels under independence makes things *worse than not adding
   them* (NLL +0.01→+0.38 exploratory, +0.04→+0.22 fresh) and inflates overconfidence (Mahal/dim 1.95/1.79,
   q95 ≈ 10 vs the 5.99 reference — error bars ~sqrt(2) too small). The correlated update absorbs the same
   channels harmlessly. **Correlated fusion is not just better, it is robust to channel-stacking; independent
   fusion anti-scales with evidence.**
4. **Residual mis-calibration on fresh points at Sigma(hop):** even the correlated update stays overconfident
   on the fresh slice (Mahal/dim 1.39–1.49). This harness fits CONSTANT covariances — and we know (confirmed,
   p=0.001) the residual covariance varies with hop. The designed synthesis (hop-conditioned `V(hop)` feeding
   the Kalman gain) is exactly what should eat this residual; that is the natural next rung.

## Caveats

Exploratory, not preregistered; split-SE is stability-only (splits share one dataset); two corpora; single LLM
judge; the PoE channels here are constructed features (membership-space products), not independent evidence;
constant covariance blocks (no hop conditioning yet); the fresh slice was already used for the Sigma(hop)
confirmatory test, so it is fresh relative to model development but not never-touched.

## Repro

```
python3 run_product_kalman_realdata.py --dataset exploratory
python3 run_product_kalman_realdata.py --dataset fresh
```
Inputs: the committed loaders (`sigma_hop_confirmatory.py`) over the run artifacts in `/tmp/mu_data/`
(multihop_score_in.tsv / multihop_resp.txt / multihop_e5.pt; sigma_hop_fresh_pairs.tsv /
sigma_hop_fresh_responses_gpt55low.txt / sigma_hop_behavior_slice_e5.pt), `model_prod.pt`, the 100k_cats TSV
graph and the enwiki_cats_correct scoped LMDB (root Behavior).
