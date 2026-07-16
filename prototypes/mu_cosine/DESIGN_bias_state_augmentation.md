# Bias-state augmentation: per-judge × per-distance bias states, and the dual-space question

Co-designed (user proposal, 2026-07-16) — the state-space form of what the record already demands:
judge bias VARIES with graph distance (REPORT_luna_campaign §1: luna D bias +0.090 transitive vs −0.05
lateral; S bias −0.15 → −0.06 by stratum), bias handling is first-order (the #3648 blocker-3 correction
flipped a headline claim), and "per-stratum/per-D-bin bias models" is the declared follow-up in the
corrected luna report. This doc gives that follow-up a proper estimator, and works through the user's
dual-space (logit ⊕ direct) state proposal with the Lagrangian coupling.

## 1. The state model

**States.** For each judge j ∈ {luna, 5.5, sonnet-5, graph, prior-as-expert…}, each distance bin
k ∈ {h1..h5, sib, cous, rand} (strata classes; d_sym bins for corpora without clean hops), and each
channel c ∈ {D, S}: a bias state b_{j,k,c}. Plus the core state x = (D, S) per row.

**Measurement equation.** A row at (soft) distance w with judge j's calibrated reading z:

    z = H_core·x + Σ_k w_k · b_{j,k,c} + v,        v ~ N(0, R_j)

- w = outcome-blind soft assignment over bins (graph features only — never labels): softmax over
  d_sym/hop features. Hard switching is the one-hot special case. The soft weights do two jobs the user
  identified: they express genuine uncertainty in graph distance (Wikipedia min-hop is ambiguous under
  multipath), and they couple neighboring bins in the information matrix — kernel smoothing that lets thin
  bins borrow strength. That coupling is exactly the "implicit measurement correlation" of the proposal.
- **Linear-combination form, not mixture form.** Treating w as mixture weights over "which bin is true"
  yields a Gaussian-mixture filter; treating Σ w_k b_k as a linear functional of the states keeps the
  update exactly linear-Gaussian. We take the linear form: same intent, one Kalman update.

**Zero-bias prior as a weak pseudo-measurement (user).** Observe 0 on each b with large R_prior:
mathematically identical to the Gaussian prior b ~ N(0, σ²_prior) and to ridge shrinkage. This is the same
object as the name-architecture's ‖r‖→0 residual pull: shrink to the family default, deviate only where
data insists. Priors and constraints are BOTH pseudo-measurements — they differ only in R (weak vs strong).

**Transition.** Identity within a campaign (rows are exchangeable — there is no time axis to track).
Consequence worth stating plainly: with identity transition and no process noise, the sequential filter's
posterior over the biases equals ONE batch hierarchical (partial-pooling) fit — per-(j,k,c) biases with
shared shrinkage. So the batch implementation (a small ridge regression in the target factory) and the
recursive implementation are the same estimator. The recursion buys something only ACROSS campaigns: add
small process noise Q and the biases become the random walk of the metastable-drift design, tracking judge
version changes. Batch now; recursive when there is a stream.

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
  propagation equations for second-order statistics. A direct-space Gaussian on a [0,1] quantity is
  misspecified near the edges (variance must shrink as μ(1−μ) but a constant-R Gaussian doesn't know
  that); pushing a logit Gaussian to μ space multiplies by σ'(ℓ) = μ(1−μ) per row (heteroscedastic
  Jacobian), and the mean acquires a Jensen correction (E[σ(ℓ)] ≠ σ(E[ℓ])). Second order in direct space
  is inherently operating-point-dependent.

Neither space wins uniformly — that is itself a measured result (the champion predictive is the μ⊕logit
MIXTURE, and the mean-vs-distribution split showed the μ expert carries points while the mixture carries
uncertainty shape). So "logit only" would contradict our own record at the OUTPUT layer. The question is
what the STATE layer needs.

## 3. The dual-state proposal (user), analyzed

Proposal: maintain bias states in BOTH spaces, b_ℓ (logit) and b_μ (direct), tied by a strong Lagrangian
constraint; expect their error correlation to be near — but not exactly — one, the gap coming from
nonlinearity.

**The constraint, linearized.** Within bin k the operating point is approximately fixed (that is what the
bins are for: transitive-h1 sits near μ≈0.9, laterals low — the bins stabilize μ̄_k). The exact map is
b_μ = E[σ(ℓ + b_ℓ) − σ(ℓ)] over the within-bin ℓ distribution. Statistical linearization — already the
program's champion tool — gives the honest affine version:

    b_μ ≈ L*_k · b_ℓ,      L*_k = E[σ'(ℓ)] over bin k   (Stein/statlin slope, NOT σ'(ℓ̄))

with a RESIDUAL whose variance is computable from the same quadrature: the linearization error from
within-bin spread of μ and the curvature of σ. That residual is precisely why the correlation is not
exactly one — the user's intuition, quantified: corr(b_μ, b_ℓ) = L*_k σ_ℓ / sqrt(L*²_k σ²_ℓ + σ²_resid),
which → 1 as the bin narrows and stays < 1 at any finite bin width.

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

    canonical state:   b_ℓ (logit — the clean-propagation home, per §2)
    discrepancy state: δ = b_μ − L*_k · b_ℓ      (small; strong zero prior R_c from statlin)

This IS the dual-state design — (b_ℓ, δ) is a linear change of variables from (b_ℓ, b_μ) — but the second
state is now the small, nearly-independent part instead of a near-copy, so the covariance is
well-conditioned by construction and the "Lagrangian" is just δ's tight prior. Two properties make this
the right form:

1. **It is falsifiable rather than aesthetic.** If the posteriors of the δ states stay at their prior
   (nothing learned), logit-only was sufficient — and we will have MEASURED that instead of assumed it.
   If δ absorbs real mass in specific bins (candidates: the boundary bins, where direct-space mass and the
   lattice bite), that is exactly where and how much the dual-space structure earns.
2. **It keeps the output layer untouched.** The bias filter runs underneath in (b_ℓ, δ); the two-space
   PoE mixture (G_sl champion: μ⊕logit experts, weight fit on calibration, change-of-variables NLL as the
   common error currency) remains the output layer combining posteriors. Mean fusion and distribution
   fusion stay separated, as measured.

Position statement, for the record: I lean logit-primary because second-order propagation is only clean
there, but the program's own results (mixture champion; μ-expert points; boundary mass) rule out
discarding direct space at the output — and the δ-state form lets the STATE layer earn direct-space
structure empirically instead of by argument. If δ ends up flat everywhere, collapse to logit-only and
cite the run.

## 5. Build plan

1. **Batch hierarchical bias fit (target factory — Claude's lane).** Per-(j,k,c) biases in logit space,
   soft outcome-blind bins, weak zero prior; replaces the global affine debiasing in target construction.
   Acceptance: beats global affine on the node-disjoint ladder (`run_sym_channel_fusion.py` harness) with
   the paired node-block bootstrap; per-bin posteriors reproduce the measured stratum-bias table's signs.
2. **δ-states (dual-space residual).** Add after (1) is in: statlin L*_k and R_c per bin from the existing
   GL-3 quadrature code; report δ posteriors per bin. Decision gate: any |δ| posterior mean > 2 SD from 0
   → keep dual-space in that bin; else logit-only.
3. **Recursive/drift version + correlation-budget analysis (Codex's lane).** Process-noise Q for
   cross-campaign judge drift; conditioning of the (b_ℓ, b_μ) vs (b_ℓ, δ) forms under the QR/whitening
   engine; the safe-correlation bound as a function of bin width.
4. Filing v1 consumes (1) directly: the Pearltrees labeling campaign's debiasing uses the shrunk
   per-stratum biases from day one (the overlap set is the fit set).

Related docs: DESIGN_cheap_judge_pipeline.md §4 (batch-vs-dynamic ladder — this doc is rung 4 made
concrete), DESIGN_amortized_fusion_heads.md (metastable drift states), REPORT_product_kalman_statlin.md
(the statlin machinery reused for L*, R_c), REPORT_product_kalman_logit.md (coordinate evidence),
DESIGN_joint_square_root_qr_conditioner.md (the conditioning engine §3 stresses).
