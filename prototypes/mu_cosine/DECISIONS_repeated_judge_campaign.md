# Repeated-judge campaign decisions

Append-only rationale for `PREREG_graph_geometry_repeated_judge.md`.  Entries record rejected alternatives and
the conditions under which they may be reconsidered.

## 2026-07-12 — freeze a no-spend campaign layer before fresh judgments

**Decision:** implement and audit the sampler, scoring schema, repeat estimand, full-procedure simulation, and
decision gates before making any judge call.  A scorer pilot needs a preregistered amendment and is excluded
from confirmation.

**Rejected:** collecting a convenient number of labels first and choosing the covariance analysis afterward.

**Reason:** candidate capacity, sample size, and nuisance refitting all affect the familywise null.  Freezing
them after outcomes would convert a confirmatory claim into post-hoc reuse.

## 2026-07-12 — substitute cumulative walk without expanding the selector

**Decision:** use one deployment-capable convex family
`K_gamma=gamma K_cumulative+(1-gamma)K_Nomic`, with the complete `gamma x rho_max` grid calibrated as one
familywise maximum.  Cumulative decay `(1,.5,.25,.125)` replaces same-hop walk.

**Rejected:** separate deployment votes for closed, same-hop, heat, resolvent, cumulative, Nomic, MiniLM, e5,
and their products.

**Reason:** closed/cumulative/heat/resolvent were nearly equivalent on the inspected topology, same-hop failed
its adjacency ordering, and Nomic/MiniLM/e5 were strongly redundant.  Extra branches add null multiplicity
without identified geometry.

**Reconsider if:** a new outcome-blind graph has pairwise kernel correlation below the frozen equivalence
threshold before any residual is inspected.  Any expansion requires a new null calibration.

## 2026-07-12 — match maximum coupling, not a common path coefficient

**Decision:** search a common `rho_max` grid and derive `alpha(K,rho)` per kernel.

**Rejected:** applying one `alpha` to kernels with different maximum/off-diagonal energy.

**Reason:** the failed v1 mechanism audit showed that common alpha is not matched covariance strength.

## 2026-07-12 — repeat waves identify noise components, not independent votes

**Decision:** retain at least three fresh calls per row and judge, with wave/request/batch identity.  Estimate
persistent, cross-wave, and repeat-specific covariance separately.

**Rejected:** averaging repeats before covariance estimation or counting three outputs as three independent
components.

**Reason:** repeats estimate call noise; endpoint components determine cross-item power.  Unrecorded call
pairing cannot identify stochastic cross-judge covariance.

**Reconsider:** four waves are already frozen as a sensitivity design; increase them if missingness or
repeat-split instability dominates, not as a substitute for more components.

## 2026-07-12 — Luna is a measurement family, not truth

**Decision:** repeat both the operating gpt-5.5-low judge and calibrated gpt-5.6-luna.  Restrict the primary
claim to operating-judge fidelity unless a blinded third-family, human, structural, or downstream target is
added under an amendment.

**Rejected:** calling cross-model agreement ground truth or treating the two judges as independent fusion
votes.

## 2026-07-12 — size by full deployment-event power

**Decision:** simulate `G={160,320,512,800}` independently per required corpus, using full nuisance refitting,
1,999 null draws, and 200 alternatives.  Four repeats are a sensitivity at `G={320,800}`.

**Rejected:** treating the favorable 320-component pointwise mechanism result or the phrase “more than 400” as
a final sample size.

**Reason:** simultaneous inference plus calibration, mean, marginal covariance, and selector fitting consumes
more information.  If 800 fails, increase components rather than lower 95% confidence.

## 2026-07-12 — use PSD-path shrinkage and a spectral envelope

**Decision:** choose `alpha_safe` by multiplicity-adjusted held benefit on a PSD path and add a separately
estimated Loewner-safe `delta_95` spectral envelope.

**Rejected:** elementwise lower confidence bounds on covariance entries.

**Reason:** decreasing individual off-diagonals is not ordered in the PSD cone and can decrease an eigenvalue.
Statistical uncertainty and numerical Cholesky loading have different meanings and must remain separate.

## 2026-07-12 — independence needs an upper bound

**Decision:** omit a cross-batch block only if its simultaneous 95% upper whitened spectral-norm bound is below
a preregistered `epsilon_batch`.

**Rejected:** using a small fitted alpha, large distance, or extra diagonal loading as proof of independence.

## 2026-07-12 — graph familiarization is cross-fitted mean learning

**Decision:** reuse anchor/adjacent/distant triples as a contrastive curriculum only inside outer-training
components.  Held components never train the graph judge/model.

**Rejected:** training on the confirmation components and then crediting smoother residuals to covariance.

**Reason:** the graph judge is also intended to learn a dataset's topology, but improved mean prediction and
random residual covariance are different estimands.

## 2026-07-12 — defer numerical specialization until statistical promotion

**Decision:** dense correlated QR remains the reference.  A passing single kernel may later use
`K=U Lambda U.T` and parallel factorizations of `B0+lambda_i Bg`, benchmarked end to end at
`N_item=128,M_channel=32,P_state=2`.

**Rejected for this PR:** CUDA performance claims, multiple noncommuting kernels, explicit covariance inverse,
and identifying JointPosterior with a matrix factorization.

**Reason:** JointPosterior is the calibrated decision comparator.  The eigenmode/QR conditioner is a numerical
implementation of an already validated covariance model.

## 2026-07-12 — distinguish campaign quotas from fold apportionment

**Decision:** interpret `G` as the count per required corpus.  Preserve exact equal quotas across each corpus's
32 joint cells, then apportion those selected components across five outer folds and three nested inner folds
as evenly as integer counts allow.  Materialize all assignments before outcomes.

**Rejected:** require every cell to divide equally across every fold.

**Reason:** all frozen `G` values divide by 32, so campaign-cell balance is exact.  At `G=512`, however, each
cell has 16 components and cannot divide equally over five outer folds.  Requiring impossible fold equality or
silently changing `G` would invalidate the power recommendation; near-balanced fold apportionment preserves it.

## 2026-07-12 — isolate every confirmatory judge request

**Decision:** send one campaign row per stateless judge request and retain request identity.  Numerical batches
used later by the conditioner are formed only after measurement collection.

**Rejected:** score ten campaign rows in one prompt while treating endpoint components as independent.

**Reason:** a shared prompt can induce request-level covariance and connects otherwise endpoint-disjoint
components.  The frozen simulator has no such cluster effect, so shared requests would invalidate its unit of
analysis.

## 2026-07-12 — freeze the PSD safety adapter completely

**Decision:** use shrinkage multipliers `s_safe in {0,.25,.50,.75,1}`, define missing spectral mass from
inner-held whitened repeat-mean residual second moments including `W_call/R_eff`, and set the cross-batch
omission tolerance to `epsilon_batch=.025`.

**Rejected:** an unspecified “lower covariance confidence bound,” entrywise lower bounds, or tuning the
batch-independence tolerance after observing residuals.

**Reason:** covariance safety is an eigenvalue/PSD-order question.  The `.025` tolerance equals the smallest
nonzero coupling the registered selector attempts to resolve; anything larger must remain in the conditioning
matrix.

## 2026-07-12 — supersede single-row calls with explicit prompt blocks

**Decision:** amortize the system prompt over at most ten same-role rows from distinct components.  Keep stable
prompt-block membership across judges/repeats/roles, contain every block within one corpus/outer/inner split
signature, fit request effects, and use whole prompt blocks as the conservative inference clusters.

**Supersedes:** the earlier same-day single-row-request decision, before any campaign selection, simulation
result, or fresh judge response existed.

**Rejected:** either repay the full system prompt for every row or batch rows while continuing to bootstrap
individual components as if the request induced no dependence.

**Reason:** single-row requests waste tokens; unmodelled shared prompts invalidate component-only uncertainty.
Stable bounded blocks preserve token amortization, prevent requests crossing held-out boundaries, and make the
remaining dependence explicit in the power calculation and inference.

## 2026-07-12 — include prompt incidence in the conditioner

**Decision:** construct the fitted prompt covariance from immutable request-incidence matrices and add it,
along with row-call and retained wave terms, to the safe measurement covariance before dense QR.

**Rejected:** use prompt blocks only for bootstrap clustering while omitting their shared random effect from
the covariance supplied to the conditioner.

**Reason:** repeated averaging reduces but does not erase a shared prompt effect.  Its schedule-dependent
incidence term can couple multiple components and may not commute with the selected graph kernel.  Dense QR
handles the general sum; the eigenmode shortcut requires a separately proved block/commutation condition.

## 2026-07-12 — do not extrapolate a three-row spectral envelope to 128 items

**Decision:** validate the numerical `N_item=128` conditioner against stipulated dense covariance only.  Require
an N-aware simulation, direct validation, or concentration/row-sum envelope before calling its statistical
covariance Loewner-safe.

**Rejected:** reuse the three-row campaign's scalar `delta_95` unchanged at 128 items.

**Reason:** coherent missing covariance can accumulate with block size, so local entry/block accuracy does not
imply a dimension-free spectral bound.
