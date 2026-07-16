# The cheap-judge pipeline: volume scoring with luna, calibrated fusion, escalation — design + baseline

Co-designed (user proposal + measured grounding), 2026-07-10. The scheme: if luna is ~k× cheaper than
gpt-5.5, collect ~k× the luna data, score a small overlap with 5.5, fit the fusion on the overlap, and use
FUSED targets on the luna-only bulk. Grounding: `REPORT_luna_campaign.md` (luna ≈ 90% of 5.5's
self-agreement pooled; fused targets recover half-to-all of the D-quality gap, 6/6 seed cells;
`REPORT_multi_judge_fusion.md` (channel values; non-degeneracy). Diagrams: `figures/pipeline_dataflow.png`,
`figures/fusion_architecture.png` (`make_pipeline_diagrams.py`).

## 1. Budget & data flow (the user's scheme, refined)

For a scoring budget with price ratio k (luna k× cheaper):

1. **Sample structurally for D coverage (user):** pick a node at random, walk D parent hops — per-hop
   quotas (h1..h5) plus sibling/cousin/random strata give coarse coverage of the D axis BEFORE any scoring
   (`sample_channel_campaign.py` already does this). Caveat: hop is a proxy — deep transitive pairs split
   directional-vs-unrelated (the class-mixture result) — so structure can't precisely fill the ambiguous
   middle. Fix: luna scores FIRST (the cheap pass), then top up thin D-bins using luna's D.
2. **Luna scores everything** (the bulk).
3. **5.5 scores a RANDOM overlap core** (calibration set). Size: the fusion parameters are ~20 numbers
   (5×5+ covariance blocks, per-channel affine bias corrections, optionally per-stratum) — a few hundred
   rows suffice (our fits were stable at 500–700 across seeds). Random, NOT conflict-selected: fitting R
   on selected-hard rows biases it upward (selection effect).
4. **Routed 5.5 calls** on the bulk's high-conflict rows (|graph − prior| innovation, Lever A's best
   policy) — extra expensive labels where they're worth most; these do NOT enter the covariance fit.
5. **Escalation tiebreaker (user): claude sonnet 5, low reasoning effort** on luna-vs-5.5 disagreement
   rows. Cross-family (e5 card distance ~0.83 from the GPT cluster) — an independent-ish error process
   that partially breaks the same-family circularity every result so far carries. Sonnet's own R is fit
   on a small RANDOM slice, not the disagreement slice. Card:
   `"LLM judge anthropic claude sonnet 5, low reasoning effort"` — onboards at r=0 with the Claude-family
   name prior.
6. **Fit the fusion on the overlap** → explicit correlated Kalman posteriors on the luna bulk → distill
   into the `kalman-fused` name head (+ channel heads keep their raw supervision).

The escalation ladder, priced: luna everywhere → 5.5 on the random core + routed rows → sonnet 5 on
disagreements. Each rung is a judge with a card, a fitted (R, bias), and a cost.

## 2. Fusion form: correlated Kalman with the prior as an expert

Not naive independent PoE (it anti-scaled with evidence in the original ladder). Channels and H rows:

| channel | observes | source |
|---|---|---|
| model prior (μ_D, μ_S) | state | agnostic readout — free |
| graph walk d (calibrated) | D | `hit_prob` — free |
| **graph symmetric distance (user)** | **S** | lateral/common-ancestor structure — free; makes the S fusion non-trivial for the first time (validated below) |
| luna D, S | D, S | cheap judge |
| [5.5 D, S / sonnet D, S] | D, S | routed rows only |

Bias first: luna's tilt (−S universal, +D transitive-specific) is an affine correction fit on the overlap,
not folded into R as variance (the Mahal-1.55 overconfidence fix). All cross-channel correlations are FIT
on the overlap and priced by the correlated update `K = (PHᵀ + C)S⁻¹`.

**The fused-judge exclusion rule:** the posterior is a linear combination of its inputs, so `kalman-fused`
is near-deterministically correlated with luna/graph/prior. It REPLACES its inputs downstream; it must
never be fused alongside them as independent evidence (naive PoE would double-count; a correct joint fit
would discount it to zero new information).

## 3. Two Kalman forms — judge vs principles (user question), resolved as the two timescales

- **`kalman-fused` as a JUDGE (amortized, slow timescale):** distillation bakes the fusion map — gain and
  correlations included — into a name-conditioned head. One forward pass; works for pairs with NO
  measurements at inference (the luna-only bulk's downstream use). It behaves as if it knows the
  correlations but can't report or adapt them.
- **The EXPLICIT filter (principles, fast timescale):** when measurements exist at inference, run the real
  correlated update — exact where the head is an approximation. The graph measurement is ALWAYS available
  (the walk is cheap), so filter-at-inference over {prior head, live graph} vs head-only is a real open
  comparison (DESIGN_amortized_fusion_heads build-order step 5 — still unrun).
- Maintenance asymmetry: judge drift (version bump) → re-fit ~20 block numbers on a fresh overlap slice
  (fast) vs re-distill the head (slow). Fast knowledge in the blocks, slow knowledge in the weights.

## 4. Batch statistics: how they work, and why batch is the baseline (user: "my guess is stability")

**How it works now.** Per corpus, on the train/overlap split only: stack per-row residuals
`E = [y − prior (2 cols), meas − y (per channel)]`, take the empirical covariance, shrink toward its
diagonal (`fit_residual_covariance`, λ=0.05 — a Ledoit–Wolf-style ridge that guarantees SPD and damps the
off-diagonals the sample supports least). The result is ONE stationary joint Gaussian: constant blocks
P₀ (prior error), R (measurement errors), C (cross-correlations). The filter is then time-invariant — the
same gain algebra for every row. Affine calibrations (graph d→D, judge bias) are fit on the same split.

**Why batch over dynamic — stability is right, and three more reasons:**
1. **Stability / estimator variance (the user's guess).** ~20 numbers from 500–700 rows is
   well-conditioned; a windowed/online estimate from dozens of rows is noisy, and gain is a RATIO of these
   estimates — a noisy R̂ near zero briefly makes the filter collapse onto that channel. Batch + shrinkage
   keeps the gain smooth.
2. **No time axis.** Our rows are exchangeable pairs, not a sequence — there is no within-campaign
   dynamics for a dynamic filter to track. The real non-stationarity is ACROSS campaigns/judge
   versions/corpora, which is slow and discrete — piecewise-constant blocks (re-fit per corpus, per
   campaign) match that structure better than continuous adaptation.
3. **Evaluation honesty.** Constant blocks make splits/seeds/replication clean (fit on train, apply to
   held). Adaptive statistics leak sequencing choices into results and make A/Bs path-dependent.
4. **The epistemic limit.** Error statistics are statistics OF the model/judges — estimable only from
   outside. A batch fit on held data is the cleanest external estimate; the more "dynamic" the estimator,
   the closer it drifts toward the model estimating its own error in-distribution (the feedback loop the
   design forbids).

**The alternatives ladder (increasing dynamism — try in this order, each needs a trigger):**

| # | scheme | what it buys | risk / when justified |
|---|---|---|---|
| 0 | **batch per corpus (baseline, built)** | stable, honest, cheap | stale under drift |
| 1 | piecewise-constant re-fit (per campaign / per stratum / per D-bin) | matches the actual drift structure | needs enough rows per cell |
| 2 | EMA across re-fits | smooth slow adaptation, still stable | lag vs window trade |
| 3 | innovation-based adaptation (Mehra-style R estimation; χ² consistency test → adaptive fading) | detects drift ONLINE (judge version change) without labels | chases noise if the χ² trigger is loose; needs a sequence (deployment stream, not batch scoring) |
| 4 | state augmentation: bias + volatility as states (the metastable design) | luna's tilt priced as BIAS, not variance; per-regime R | more state to identify; wants regime labels or heavy tails |
| 5 | model-learned HETEROSCEDASTIC noise: Σ(context) head predicting per-pair R/P (the Σ(hop) head generalized) | per-pair precision weighting — the statlin/hetero-MLE lesson says known per-row noise helps | must be trained on EXTERNAL error data (innovations/held-out) with stop-grad — never on its own fit |
| 6 | fully joint: differentiable Kalman layer, model + filter co-learned end-to-end | maximum expressiveness | the tautology trap at full scale; gradients through the gain are unstable (this is where square-root/UDUᵀ forms stop being optional) |

Division of labor for 5–6 (answering "could the model learn the statistics"): per-pair VARIANCES are a
good model target (context-dependent, lots of signal); cross-CHANNEL correlations should stay external
constants (few numbers, weakly identified per-pair, and exactly the self-knowledge the epistemic limit
says the model can't have from inside).

## 5. Baseline validation on existing data (zero new scoring)

Two experiments accompany this doc (results in `REPORT_cheap_judge_baseline.md`):
1. **Symmetric-graph S channel** (`run_sym_channel_fusion.py`): the user's distance-based symmetric graph
   judge (common-ancestor lateral distance + shared-parent/grandparent structure, regression-calibrated to
   S on train) added as an S measurement row — does the S ladder improve the way D's did?
2. **Matched-cost simulation** (`sim_matched_cost.py`): learning curves — n pure-5.5 labels vs k·n
   fused-luna targets at equal cost, sweeping n on the 1,700 dual-judge pairs — locates the break-even
   price ratio k before any budget is committed.

## 6. Build items when a real campaign runs this scheme

Sampler: add luna-first D-bin top-up to `sample_channel_campaign.py`. Scorer: sonnet-5 tiebreaker path
(Claude-side scoring, card above). Fusion: per-D-bin affine bias corrections (uniform-per-D overlap gives
the support). Bookkeeping: overlap slice doubles as the drift monitor (innovation χ² across campaigns —
alternatives ladder #3's trigger, computed offline).
