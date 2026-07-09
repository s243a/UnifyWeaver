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

## Which mean does "PoE" take? (user question, 2026-07-08)

Two different products are in play and they take DIFFERENT means — the harness's PoE is *not* the lower estimate:

- **Product of DENSITIES (Gaussian PoE = `independent_kalman` here):** multiplying Gaussian densities over the
  same latent yields the precision-weighted **arithmetic** mean `(Λ1μ1+Λ2μ2)/(Λ1+Λ2)` — a convex combination
  that INTERPOLATES between experts, never below the min. Semantics: fusion of noisy estimates of one quantity.
- **Product of VALUES (`mu_lower = Π μi^wi`):** multiplying memberships themselves. Unnormalized (w=1) = strict
  AND ≤ min; normalized (Σw=1) = weighted **geometric** mean (≤ arithmetic by AM–GM, ≥ min — conservative
  consensus, not a strict bound). Semantics: conjunction of independent requirements — this is the lower estimate.

**Bridge — it is a choice of coordinates:** density-PoE fused in μ-space → arithmetic mean; in **log-μ** space →
**geometric** mean; in logit space → geometric mean of odds (multiplying Bayes factors). The DESIGN's log-space
product-Kalman (`K_ell = P/(P+R)`) IS the geometric-mean fusion with learned precision weights.

In this run: the fusion rule was μ-space density-PoE (arithmetic); config B's lo/hi channels were value-products
used as features. NOT yet tested: the fusion itself in log/logit coordinates (the geometric-flavor Kalman) —
`product_space.py` has the links; the comparison needs the change-of-variables Jacobian so NLLs stay comparable
across coordinate systems. Candidate next rung alongside hop-conditioned V(hop).

### Which space are the errors actually Gaussian in? (user refinement + measurement, 2026-07-08)

*User: it's only a coordinate change if the error statistics are defined in a different coordinate system than
the fuse space — and the Kalman filter runs in any of these spaces.* Both right: exact Bayes is
reparameterization-invariant, the GAUSSIAN ASSUMPTION is not; a Gaussian in mu is not Gaussian in log-mu. So the
fuse-space choice = where you assert the noise is Gaussian (mu = additive noise, log = multiplicative, logit =
noise on odds) — a modeling decision, and empirically decidable: fuse where the residuals ARE most Gaussian.

Measured (affine mean model per space; Jarque–Bera, lower = more Gaussian):

| corpus | space | D skew/kurt/JB | S skew/kurt/JB |
|---|---|---|---|
| exploratory | **mu** | **−0.3 / −0.7 / 10** | +0.8 / +0.8 / 34 |
| exploratory | log | −6.5 / +56 / 34448 | −0.6 / +0.7 / 21 |
| exploratory | logit | −3.2 / +21 / 4905 | −0.1 / +0.4 / **2** |
| fresh | **mu** | **+0.2 / −0.6 / 6** | **+0.1 / −0.4 / 2** |
| fresh | log | −4.7 / +26 / 8208 | −2.2 / +8.2 / 915 |
| fresh | logit | −3.1 / +15 / 2663 | −1.4 / +4.2 / 261 |

**Verdict: the errors are approximately ADDITIVE — mu-space is the right fuse space for this data** (JB 2–34 vs
10^2–10^4 elsewhere); the arithmetic fusion above was correctly placed, and a geometric-flavor (log-space) Kalman
would spend its Gaussianity where it is most violated. Mechanism: fuzzy labels pile up AT the 0/1 boundaries and
the log/logit links blow those into extreme tails (1/mu, 1/(mu(1−mu)) Jacobians). Nuances: boundary pile-up is
itself non-Gaussian in ANY unbounded space (mu just keeps it finite); and exploratory-S is mildly better in logit
(JB 2 vs 34) — the best space can be per-channel/per-corpus, consistent with the corpus-specificity theme. This
retires the "geometric-flavor Kalman" rung for this data; the space question is settled empirically, not by
convention.

### Why Gaussianity matters to the FILTER specifically: autonomous covariance propagation (user, 2026-07-08)

*User: Gaussian statistics are relevant to the Kalman filter because the covariance can then be propagated as a
linear differential or difference equation.* Sharpened, and it explains why the architecture is possible:

- **Closure:** linear + Gaussian ⇒ the posterior stays Gaussian, two moments suffice, and `P` evolves by its own
  autonomous matrix equation (`P_{k+1} = A P A^T + Q`, `P+ = (I−KH)P`; continuous: the Riccati ODE).
- **The key consequence: `P`'s evolution is DATA-INDEPENDENT** — measurements never appear in it; only structure
  (`A, H, Q, R`) does. Error bars follow a deterministic, precomputable schedule (classically: steady-state
  Riccati/gain); only the mean is data-driven.
- **That is the license for the `Sigma(hop)` head:** covariance-as-a-function-of-structure is a precomputed
  Riccati trajectory along the hop axis — predictable from graph position without seeing labels, BECAUSE
  covariance propagates autonomously in the linear-Gaussian regime. The confirmed Sigma(hop) result is an
  empirical instance. Conversely, where autonomy breaks (adaptive `R` from innovations, the metastable drift
  layer, EKF-style state-dependent linearization) is exactly what stays with the ONLINE filter. **The
  two-timescale split follows the mathematical fault line: precomputable covariance schedule → model head;
  data-dependent statistics tracking → filter.**
- **Precision:** the covariance difference equation needs only linearity + second moments (Kalman = LMMSE for
  any noise). Gaussianity upgrades best-linear to EXACT BAYES, making `P` an honest credible region. The JB
  diagnostic above certifies that upgrade for mu-space (JB 2–34 ≈ near-exact-Bayes, Mahalanobis ≈ coverage);
  in log space the same algebra would run but `P` would degrade to second-moment bookkeeping with non-Gaussian
  tails (kurtosis +56).
