# Bias-state augmentation: per-judge × per-distance bias states, and the dual-space question

Co-designed (user proposal, 2026-07-16) — the state-space form of what the record already demands:
judge bias VARIES ACROSS STRATA (REPORT_luna_campaign §1: luna D bias +0.090 transitive vs −0.05 lateral;
S bias −0.15 → −0.06 by stratum — a stratum-dependent result; a smooth trend WITHIN h1–h5 is plausible but
not yet measured, which is one thing the binned states will show), bias handling is first-order (the
#3648 blocker-3 correction flipped a headline claim), and "per-stratum/per-D-bin bias models" is the
declared follow-up in the corrected luna report. This doc gives that follow-up a proper estimator, and
works through the user's dual-space (logit ⊕ direct) state proposal with the Lagrangian coupling.

Scope note: this is an implementation spec for a PRACTICAL labeling/fusion system. Where a choice arises
between inferential ceremony and a decision-oriented check, we take the decision-oriented check (held-out
node-disjoint improvement) — the system's job is filing bookmarks, not hypothesis testing.

## 1. The state model

**States.** For each judge j ∈ {luna, 5.5, sonnet-5, graph, prior-as-expert…}, each distance bin
k ∈ {h1..h5, sib, cous, rand} (strata classes; d_sym bins for corpora without clean hops), and each
channel c ∈ {D, S}: a bias state b_{j,k,c}. Plus the core state x = (D, S) per row.

**Calibration order (important — the #3648 lesson protected).** Bias states sit ON TOP of the existing
GLOBAL per-channel affine calibration (slope + intercept, `affine_calibrate`, fit train-only), never in
place of it: the states are additive offsets and cannot express a slope error, so dropping the affine
would reinvite exactly the tilt that flipped the #3648 headline. Pipeline per judge/channel:
global affine first → per-bin residual offset states on the calibrated reading. (Per-bin GAIN states are
a possible later refinement; not in scope.)

**Gauge fixing (identifiability).** The core state x and the biases share a gauge: shifting x by Δ and
every co-observed bias by −Δ leaves the likelihood unchanged. Fix: the OPERATING judge (gpt-5.5-low, the
target frame every report already uses) has its bias pinned to 0; all bias states are estimates
RELATIVE TO the operating judge, and are labeled as such — fidelity/bias vs gpt-5.5-low, never "semantic
accuracy". (An absolute frame needs the human-verified gold subset — the known, deferred upgrade.)
Implementation prints cheap per-fit diagnostics and FAILS CLOSED on them: unregularized design rank
(kernel columns overlap, so Σw is support mass, not effective sample size — rank is the honest check),
per-state conditional posterior variance, and the design condition number; a state whose conditional
information is below floor falls back to its prior. Rows with NO usable distance signal get their own
explicit `missing` basis state rather than being silently mapped into `rand`.

**Measurement equation.** A row with soft distance weights w and judge j's affine-calibrated reading z:

    z = H_core·x + Σ_k w_k · b_{j,k,c} + v,        v ~ N(0, R_j)

- **w is a deterministic, outcome-blind kernel basis** (graph features only — never labels): bin centers =
  the strata classes, bandwidth a train-tuned kernel width over d_sym/hop features, rows with no usable
  distance getting their own explicit `missing` basis state (not silently mapped to `rand` — "no signal"
  and "measured unrelated" are different facts). Hard switching is the bandwidth→0 special case. To be precise about
  what the softness buys (review point): overlapping w columns make the bias ESTIMATES share information
  across neighboring bins (coefficient coupling through the design — kernel smoothing, thin bins borrow
  strength). It is NOT correlated measurement noise, and we explicitly reject the alternative reading of
  w as "probability the row belongs to bin k" — marginalizing that uncertainty gives a variance-inflated
  mixture and a Gaussian-mixture filter for no practical gain.
- **Linear-combination form, not mixture form.** Treating Σ w_k b_k as a linear functional of the states
  keeps the update exactly linear-Gaussian: one Kalman update.

**Zero-bias prior as a weak pseudo-measurement (user).** Observe 0 on each b with large R_prior:
mathematically identical to the Gaussian prior b ~ N(0, σ²_prior) and to ridge shrinkage. This is the same
object as the name-architecture's ‖r‖→0 residual pull: shrink to the family default, deviate only where
data insists. Priors and constraints are BOTH pseudo-measurements — they differ only in R (weak vs strong).

**Transition.** Identity within a campaign (rows are exchangeable — there is no time axis to track).
Consequence worth stating plainly: with identity transition and no process noise, the sequential filter's
posterior over the biases equals ONE batch hierarchical (partial-pooling) fit — per-(j,k,c) biases with
shared shrinkage — PROVIDED both use the same joint linear-Gaussian likelihood, including any cross-row /
repeated-judge covariance (rows sharing a judge are not independent given an uncertain bias; that shared
component is what the bias state represents, and Codex's Stage-A campaign is measuring exactly this
repeated-judge dependence). So the batch implementation (a small ridge regression in the target factory)
and the recursive implementation are the same estimator under the same model. The recursion buys
something only ACROSS campaigns: add small process noise Q and the biases become the random walk of the
metastable-drift design, tracking judge version changes. Batch now; recursive when there is a stream.

**Counting.** ~3 judges × 8 bins × 2 channels ≈ 50 bias states on ~1,700 dual-judge rows → ~35 rows/state:
identifiable WITH the shrinkage prior and the soft-w smoothing, thin without. The parametric fallback if
bins stay thin: b_j(h) = a_j + c_j·h per channel (the Σ(hop)/chol_of_hop idiom) — fewer parameters,
monotone in distance; the binned form is its nonparametric generalization.

## 2. Coordinate choice: why logit is the natural state home

The bias states live in some coordinate. The candidates, with the measured record:

- **Logit space**: unbounded support (a Gaussian state is not fighting its domain — the original argument
  for logit in the fusion arc); second-order statistics propagate LINEARLY (covariance through the
  identity-plus-bias measurement map stays exact); the interior-only tests favored logit normality
  (JB, logit rung wins 3/4 interior cells). Pathology: the boundary — logit stretches [0,1] endpoints to
  ±∞, which is why dequant (Q_HALF interior clipping) and bin-mass scoring exist.
- **Direct (μ) space**: the labels, the lattice (0.05 quantization), and the filing decision all live
  here, and boundary MASS is representable. Pathology — the user's point, made precise: there are no clean
  propagation equations for second-order statistics. On [0,1] the variance is CAPPED by the support
  (Var ≤ μ(1−μ), the Bhatia–Davis/Bernoulli bound — a ceiling the actual variance need not follow, but
  which a constant-R Gaussian cannot even respect near the edges); pushing a logit Gaussian to μ space
  multiplies by σ'(ℓ) = μ(1−μ) per row (heteroscedastic Jacobian), and the mean acquires a Jensen
  correction (E[σ(ℓ)] ≠ σ(E[ℓ])). Second order in direct space is inherently operating-point-dependent.

Neither space wins uniformly — that is itself a measured result (the champion predictive is the μ⊕logit
MIXTURE, and the mean-vs-distribution split showed the μ expert carries points while the mixture carries
uncertainty shape). So "logit only" would contradict our own record at the OUTPUT layer. The question is
what the STATE layer needs.

## 3. The dual-state proposal (user), analyzed

Proposal: maintain bias states in BOTH spaces, b_ℓ (logit) and b_μ (direct), tied by a strong Lagrangian
constraint; expect their error correlation to be near — but not exactly — one, the gap coming from
nonlinearity.

**The constraint, defined operationally (empirical-first).** Within bin k the operating point is
approximately stable (that is what the bins are for: transitive-h1 sits near μ≈0.9, laterals low). The
exact map b_μ = E[σ(ℓ + b_ℓ) − σ(ℓ)] depends on the within-bin distribution of BOTH ℓ and the bias, so an
analytic slope is only as good as those distributional assumptions. The implementation therefore defines
the coupling EMPIRICALLY, per bin, on train rows only:

    L*_k    = slope of the regression of direct-space residuals on logit-space residuals within bin k
    R̂_row,k = the ROW-LEVEL residual variance of that regression

— measured numbers, no quadrature assumptions, averaged over the bin's actual (ℓ, bias) distribution. The
GL-3 statlin machinery (run_product_kalman_statlin.py — note its existing routine transports direct→logit,
the OPPOSITE direction) serves as an analytic cross-check on small bins, not as the definition.

**Two levels, not one (review distinction, adopted).** R̂_row,k is row-level observation scatter — it
contains target noise and errors-in-variables from BOTH residual axes — and is therefore an UPPER BOUND
on, not an estimate of, the state-level constraint noise R_c,k (the discrepancy between the bin-level
bias states themselves). The state-level R_c,k, the associated correlation formula
corr = L*_k σ_ℓ / sqrt(L*²_k σ²_ℓ + R_c,k) and its "→ 1 as bins narrow" limit are PROVISIONAL pending the
statistical follow-up (Codex/rigor lane: cross-fitted or replicated bin/campaign effects that separate
observation noise from state discrepancy). Until then R̂_row,k must NOT be used as a tight state
constraint — δ̃'s prior scale uses a deliberately conservative (loose) placeholder, which only weakens the
coupling, never fabricates precision.

**Lagrangian = pseudo-measurement.** The strong coupling is implemented in the same machinery as the weak
zero prior: observe 0 on the constraint residual c = b_μ − L*_k·b_ℓ with SMALL R_c (statlin supplies R_c —
it is measured, not hand-set). Constrained-Kalman theory offers three standard treatments: (i) posterior
projection onto the constraint manifold, (ii) the soft pseudo-measurement (→ (i) as R_c → 0), (iii)
reparameterization that eliminates the constraint. All three are available; (ii) is the literal Lagrangian.

**The numerical hazard — and why it lands in Codex's lane.** Near-unit correlation between paired states
makes the joint covariance nearly singular: naive inv() updates degrade exactly here. This is the concrete
in-house instance of Codex's third question ("how much correlation can we safely apply in
inverse-square-root propagation") — the dual-state filter is a natural stress test for the QR/whitening
conditioner, and the whitened condition number is the observable safety margin.

## 4. Recommended realization: logit-primary + an estimable direct-space residual

Rather than choosing logit-only by fiat OR carrying two nearly-collinear states, reparameterize the dual
pair into equivalent, well-conditioned coordinates (treatment (iii), which is exact):

    canonical state:     b_ℓ (logit — the clean-propagation home, per §2)
    discrepancy state:   δ̃ = (b_μ − L*_k · b_ℓ) / s_k     (STANDARDIZED; zero prior, unit scale;
                          s_k = conservative placeholder until the follow-up delivers state-level R_c,k)

This IS the dual-state design — (b_ℓ, δ̃) is a linear change of variables from (b_ℓ, b_μ) — but the second
state is now the small, nearly-independent part instead of a near-copy, and standardizing by sqrt(R_c,k)
keeps the precision matrix sanely scaled even when R_c is tiny (raw δ with a tight prior would just move
the ill-conditioning from correlation into scale). "Well-conditioned" still requires verification, not
assertion: the build plan includes dense-vs-QR parity and a condition-number sweep over bin widths
(consistent with DESIGN_joint_square_root_qr_conditioner.md's own warning). The "Lagrangian" is just δ̃'s
unit prior. Two properties make this the right form:

1. **It is testable at the system level, not per-state.** With ~50 states, "any |δ̃| > 2 SD" would fire
   almost surely under the null (multiplicity), so the retention gate is GLOBAL and decision-oriented —
   and guarded against pick-the-bigger-model-under-a-null selection: freeze ONE primary metric and a
   minimum practical gain BEFORE the run (S-marginal NLL on the node-disjoint ladder; floor set from the
   split-stability SD), select on inner node-disjoint validation, report once on the untouched outer
   test, and require the paired node-block bootstrap interval on the gain to exclude zero. Per-bin δ̃
   posteriors are then read DESCRIPTIVELY to localize where the gain lives (candidates: the boundary
   bins, where direct-space mass and the lattice bite). No qualifying gain ⇒ collapse to logit-only and
   cite the run.
2. **It keeps the output layer untouched.** The bias filter runs underneath in (b_ℓ, δ̃); the two-space
   MIXTURE (G_sl champion — a mixture of μ/logit experts, NOT a product; weight fit on calibration;
   change-of-variables NLL as the common error currency) remains the output layer combining posteriors.
   Mean fusion and distribution fusion stay separated, as measured.

Position statement, for the record: I lean logit-primary because second-order propagation is only clean
there, but the program's own results (mixture champion; μ-expert points; boundary mass) rule out
discarding direct space at the output — and the δ-state form lets the STATE layer earn direct-space
structure empirically instead of by argument. If δ ends up flat everywhere, collapse to logit-only and
cite the run.

## 5. Build plan

1. **Batch hierarchical bias fit (target factory — Claude's lane).** Per-(j,k,c) OFFSET states in logit
   space ON TOP of the retained global affine calibration (never replacing it — §1 calibration order);
   soft outcome-blind kernel bins, weak zero prior, 5.5-bias pinned to 0 (gauge); rank + ESS diagnostics
   printed per fit. Acceptance: beats global-affine-only on the node-disjoint ladder
   (`run_sym_channel_fusion.py` harness) with the paired node-block bootstrap; per-bin posteriors
   reproduce the measured stratum-bias table's signs.
2. **δ̃-states (dual-space residual).** Add after (1) is in: fit L*_k / R_c,k by within-bin train-only
   regression (GL-3 quadrature as analytic cross-check — note direction: a new logit→direct routine, the
   existing one transports direct→logit). Decision gate is GLOBAL: keep dual-space iff held-out
   node-disjoint score improves over logit-only; per-bin δ̃ posteriors read descriptively to localize.
3. **Recursive/drift version + correlation-budget analysis (Codex's lane).** Process-noise Q for
   cross-campaign judge drift; dense-vs-QR parity + condition-number sweep over bin widths for the
   (b_ℓ, b_μ) vs (b_ℓ, δ̃) forms; the safe-correlation bound as a function of bin width.
4. Filing v1 consumes (1) directly: the Pearltrees labeling campaign's debiasing uses the shrunk
   per-stratum biases from day one (the overlap set is the fit set; biases labeled relative-to-5.5).

Related docs: DESIGN_cheap_judge_pipeline.md §4 (batch-vs-dynamic ladder — this doc is rung 4 made
concrete), DESIGN_amortized_fusion_heads.md (metastable drift states), REPORT_product_kalman_statlin.md
(the statlin machinery reused for L*, R_c), REPORT_product_kalman_logit.md (coordinate evidence),
DESIGN_joint_square_root_qr_conditioner.md (the conditioning engine §3 stresses).
