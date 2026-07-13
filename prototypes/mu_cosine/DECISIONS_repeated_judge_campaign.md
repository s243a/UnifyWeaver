# Repeated-judge campaign decisions

Append-only rationale for `PREREG_graph_geometry_repeated_judge.md`.  Entries record rejected alternatives and
the conditions under which they may be reconsidered.

## 2026-07-12 — freeze a no-spend campaign layer before fresh judgments

**Decision:** implement and audit the sampler, scoring schema, repeat estimand, synthetic primary-event simulation, and
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

## 2026-07-13 — both zero-coupling controls must pass the false-promotion gate

**Decision:** apply the same observed primary-event rate at most 10% and one-sided exact-binomial
non-rejection gate to both the graph-local block null and the zero-coupling smooth-mean-only control.  Retain
the held residual no-harm checks as separate requirements.

**Rejected:** allowing smooth-mean-only to pass merely because its average residual gain is nonnegative.  That
would permit an arbitrarily high structured false-promotion rate under a zero-coupling control.

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
inner-held whitened repeat-mean residual second moments including the full fitted call/prompt/wave sampling
covariance, and set the cross-batch omission tolerance to `epsilon_batch=.025`.

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

## 2026-07-12 — distinguish synthetic primary sizing from deployment power

**Decision:** let the synthetic harness report only the smallest joint-two-corpus `G` passing the simultaneous
residual/posterior primary event.  Keep final campaign `G` and every deployment flag unset until decision-space,
source-component, spectral-safety, and batching gates receive their own power/feasibility bridge.

**Rejected:** call two Gaussian NLL endpoints a full deployment event or transfer its synthetic null threshold
to unknown real-data nuisance scale.

**Reason:** calibration, AURC/noninferiority, and spectral safety are meaningful real gates but are not made
representative merely by inventing a synthetic class-label process.  Real selector nulls are recalibrated from
outer-training prompt blocks with the complete refit.

## 2026-07-12 — contain and stress-test graph source components

**Decision:** keep each selected graph source component inside one outer/global-inner fold and require a
two-way prompt-block/source-component multiplier sensitivity in addition to primary prompt-block inference.

**Rejected:** assume endpoint disjointness alone removes every graph-component-level random effect.

**Reason:** prompt blocks capture shared requests, while connected graph components can carry a different
dependence path.  Fold containment prevents leakage; the two-way sensitivity exposes remaining held dependence.

## 2026-07-12 — retain stable inner labels with the minimax integer balance

**Decision:** freeze one global inner label per component/source group.  The synthetic sizing model, whose
dependency atom is one component, accepts a maximum inner-total spread of two in each leave-one-outer training
set on the registered G grid.  The real selector instead balances indivisible source groups, records the
realized global and leave-one-outer totals, and does not claim the synthetic bound for those grouped data.

**Rejected:** independently rebalance inner labels for each outer fold or claim an impossible spread of one.

**Reason:** stable labels keep shared prompts out of train/held crossings.  At `G=160`, five outer folds of 32
force one leave-one-outer inner allocation to be 44/42/42; changing labels by outer split would break the stable
prompt-block contract.

## 2026-07-12 — distinguish reproducible declarations from verified campaign inputs

**Decision:** let the no-spend selector validate and hash an externally materialized candidate pool and request
contract, but label even a frozen-shaped output `protocol-shape-compatible-no-spend-inputs-unverified` until a
repository-owned candidate builder and an approved exact model-revision/prompt/settings contract exist.

**Rejected:** interpret a caller-supplied implementation or artifact hash as independent attestation, or let a
dry-run manifest authorize live judge calls.

**Reason:** content addressing makes the supplied bytes reproducible; it does not prove how those bytes were
created or that a prompt was approved.  Conflating those properties would turn a useful dry-run tool into a
false confirmatory readiness claim.

## 2026-07-12 — counterbalance prompt position and bind exact request bytes

**Decision:** keep prompt-block membership stable, give each role-by-judge measurement an independently hashed
base order, and spread its repeat rotations across the available positions.  Hash the exact serialized
request-input bytes into the logical request ID, and require provider request/nonempty response IDs to be
unique across logical attempts.  Treat a position-effect pilot, train-only adjustment, and power sensitivity
as prerequisites for live approval rather than claims of this synthetic harness.

**Rejected:** keep each component at one list position, identify a request only by normalized endpoint IDs, or
accept one provider call as evidence for multiple nominal repeats.

**Reason:** a persistent list-position effect could otherwise follow a component and masquerade as cross-item
or cross-role covariance.  Independent bases plus spread rotations reduce that alignment but cannot identify
an arbitrary position nuisance with only three repeats.  Normalized identity deliberately ignores presentation
changes that alter actual prompt bytes, while reused provider IDs destroy the fresh-call estimand.  Membership
stability preserves the inference cluster; the remaining position question is explicitly fail-closed.

## 2026-07-12 — treat smooth mean misspecification as a false-promotion null

**Decision:** require the smooth-mean-only, zero-coupling control to pass the same at-most-10% joint primary
promotion and exact-binomial no-excess gate as the simpler block null, in addition to its no-harm diagnostic.

**Rejected:** let improved held likelihood alone validate a structured covariance under mean-only truth.

**Reason:** a covariance kernel can absorb a misspecified mean and improve prediction while still being a
false covariance discovery.  The scientific question is residual coupling after mean removal, so robustness
to the frozen nonlinear mean control is part of false-promotion control, not merely a predictive diagnostic.

## 2026-07-12 — use exact prompt-block eigenmodes with a dense fallback

**Decision:** when every component in a prompt block has the identical observed request schedule, write its
joint covariance as `I_m tensor A + J_m tensor B` and score one collective mode `A+mB` plus `m-1` contrast
modes `A`.  Validate the exact boolean schedule, use the dense overlap-aware solve otherwise, and densely
recompute candidates within `1e-12` of a strict-zero eligibility decision.  Permit factor caching only inside
one fitted nuisance object.

**Rejected:** infer exchangeability from repeat counts alone, apply the shortcut to arbitrary real incidence
matrices, reuse fitted factors across folds/replicates, or round gains at the decision boundary.

**Reason:** shared request-level missingness makes the reduction algebraically exact in the current synthetic
blocks and reduces 120-dimensional factorizations to 12-dimensional collective/contrast systems.  Equal counts
can hide different overlap schedules, while real prompt incidence need not commute with the item kernel; dense
joint QR therefore remains the general correctness reference.
