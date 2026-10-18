# Pearltrees HOP untouched-audit transaction

**Status:** prospective implementation contract, frozen before any real audit
response is computed. This stage consumes an accepted no-solve HOP plan and an
accepted calibration lock, evaluates only the 24 registered audit batches, and
installs one immutable local-only result. Synthetic fixtures are permitted
during implementation. The real plan, lock, and audit must be generated only
after this design, the protocol, and both runners have landed at one exact
repository commit.

## 1. Purpose and claim boundary

Calibration selected one scalar `alpha_top` and one audit mode using only the
32 calibration anchors. The audit applies that frozen choice to the disjoint 96
audit anchors. It does not retune alpha, change a role, replace a failed batch,
regenerate a bootstrap schedule, or inspect placement labels, embeddings,
judges, filing ranks, or downstream outcomes.

The result remains a snapshot-relative topology-only HOP fidelity result. It is
not a filing improvement, covariance estimate, graph-judge promotion, CUDA
crossover, or whole-corpus statement.

`confirmatory_claim_authorized=true` is the continuity flag for a decisive
prospective finite-contrast result. It is true for either a clean material
larger-domain efficacy result or a clean smaller-domain convergence result,
and false for an inconclusive conflict. The narrower
`convergence_claim_authorized=true` is possible only when the frozen smaller
endpoint is adequate, passes the complete noninferiority intersection, the
larger endpoint does not meet the efficacy rule, and every completeness,
numerical, bootstrap, reference, and resource gate below passes. Neither flag
can be set by `absolute_only` or `right_censored_diagnostics`.

## 2. Full-chain verification and binding

Before any audit solve, the runner performs full lock verification. That in
turn repeats declaration, two-attempt snapshot consensus, and no-solve-plan
verification against the actual source specification, relation policy, receipt,
and attempt directories. Receipt-only, plan-only, or lock-manifest-only
verification is prohibited.

The lock must have all of:

- `accepted=true`, `calibration_completed=true`, and
  `audit_solve_authorized=true`;
- `audit_solves_executed=0` and
  `confirmatory_claim_authorized=false`;
- a non-`blocked` mode, finite nonnegative `alpha_top`, and a nonempty
  path-free BLAS identity with one observed thread; and
- implementation records matching repository `HEAD`, including this design
  and `prepare_pearltrees_hop_audit.py`.

The audit scientific core binds the complete lock-manifest content record,
`lock_fingerprint`, complete plan-manifest content record, `plan_fingerprint`,
the frozen selection record, the audit work order, every scientific result,
the bootstrap decision, the numeric backend, and the audit implementation
records. A copied fingerprint without its verified content is insufficient.

The upstream chain and captured bytes are verified again after numerical work
and before installation. The installed result is then verified once more.

## 3. Audit-only work order

The numerical worker receives a derived work order containing exactly the rows
`audit-00` through `audit-23`. Each row has four unique anchors, in the frozen
quartile order `q1` through `q4`; all 96 anchors are globally unique. A
calibration-tagged row, unknown split, missing or duplicate anchor, unbalanced
quartile, unknown role, changed protected set, non-nested domain, or boundary
ledger mismatch fails before a solve.

The work order contains:

- the frozen mode, `alpha_top`, effective-resistance disposition, and exact
  `frozen_audit_roles` from `selection.json`, recorded as the decision roles;
- the deduplicated required bounded-model roles in frozen HOP order: every
  decision role plus `S_1024`, which is a support role when it is not already a
  decision endpoint, so audit reference adequacy is always re-evaluated against
  `R_top`;
- the exact protected nodes, candidate/reference node lists, complete incident
  adjacency, and exact Dirichlet boundary ledgers from canonical attempt A;
- the 96 rows from `audit_shells.jsonl`;
- the numeric, statistical, and resource contracts; and
- the exact 9,999 records from `bootstrap_multiplicities.jsonl`.

It contains no calibration anchor response, calibration fidelity metric,
minimality bracket, label, embedding, judge output, filing result, or prior
audit response. Full lock verification necessarily reads calibration evidence
to authenticate the lock, but that evidence is not passed into the numerical
worker or the audit decision functions.

The worker consumes the frozen node order directly. It may not rerun HOP
selection, change a comparator, discover a replacement shell, or use a cache
from another plan or lock.

The runner separately derives the complete calibration-anchor ID set from the
verified plan and rejects any audit anchor ID in that set. Graph domains may
overlap; anchor identity may not. Calibration IDs are not passed into the
numerical worker after this disjointness check.

## 4. Mode-specific execution

`R_top` is the shared exact-Dirichlet reference in every authorized mode. A
reference solve needed to score an authorized candidate is implicit and is not
an extra candidate role.

- `finite_contrast`: evaluate `K_low` and the next frozen finite `K_high`
  against `R_top`, plus `S_1024` as a reference-adequacy support solve when it
  is not already one of those endpoints. The support result participates only
  in the frozen reference-adequacy gate, never in the efficacy or
  noninferiority contrast.
- `absolute_only`: evaluate `S_1024` against `R_top`; efficacy and resource-
  contrast decisions are absent.
- `right_censored_diagnostics`: evaluate `S_256`, `S_512`, and `S_1024`
  against `R_top`; every result is diagnostic and claim authorization remains
  false.
- `blocked`: execute zero audit solves and do not manufacture an audit result.

Each batch uses `evaluate_nested_bounded_domain_fidelity` with the exact
protected set, one scalar `alpha_top`, `rank_top_k=8`, the frozen minimum
reciprocal condition, and the preregistered effective-resistance disposition.
Embeddings, closure, node-varying alpha, and candidate-specific recalibration
are unavailable. One reference model/factor is built per batch and each unique
non-reference decision or support model is built/factored once; node-identical
reuse is recorded rather than relabeled as a resource reduction.

The runner maps the evaluator's source-node order back to the frozen quartile
labels before aggregation. It persists timing-free binary64 scientific values
exactly and keeps observational timings and peak RSS outside the scientific
fingerprint.

## 5. Audit screening shells

The no-solve plan contains a separate `audit_shells.jsonl` artifact with one
radius-3 shell for each audit anchor. The planner derives these shells from the
same complete single-source HOP traversals used before the split was exposed to
any response. Each shell is nonempty, contained in its batch's `R_top`, and
strictly interior (`beta=0`); otherwise the plan is blocked.

The audit uses these shells only with the cached `R_top` response to report
realized screening radius, shell attenuation, and censoring at the already
frozen `alpha_top`. It does not report candidate-specific screening; one
reference-only record supplies batch provenance for every role. Audit
screening cannot change alpha, a domain, completeness, or a decision endpoint.

## 6. Complete batches and failures

A complete batch contains all four quartile anchors and every result, safety
diagnostic, and active endpoint required by its lock mode. A failed batch is
never split into surviving anchors. The complete-batch mask is fixed before
any aggregate or bootstrap endpoint is computed.

At least 18 of the 24 balanced audit batches are required for a decision-bearing
result. With 18 through 23 complete batches, estimation uses only the fixed
complete-batch mask under the exact rule in Section 7. Fewer than 18 is
descriptive only and cannot authorize a claim.

The closed allowlist for `c_b=0` is a structured whole-batch candidate- or
reference-protected-coverage failure reported before any metric from that batch
is committed. No other exception or failed check is ordinary missingness.
Malformed provenance, changed adjacency, nonfinite output, M-matrix failure,
loss of grounding, insufficient reciprocal condition, Cholesky/solve residual
failure, maximum-principle failure, an elapsed/RSS failure, or a partially
written batch is a deterministic global block, not an incomplete batch. An
absolute, reference-adequacy, efficacy, or noninferiority threshold miss is a
complete observed result and can never be converted to `c_b=0`. Allowed
coverage failure removes the whole four-anchor batch, is never silently
imputed, and remains subject to the 18-batch floor.

For a finite contrast, `K_low` must realize strictly fewer nodes than `K_high`
in every decision-bearing complete batch for a resource-contrast statement.
Node-identical endpoints remain valid absolute diagnostics but cannot support a
resource contrast or confirmatory convergence authorization.

## 7. Frozen paired bootstrap with incomplete batches

The runner consumes, and never regenerates, each frozen 24-entry multiplicity
vector `m_r`. Let `c_b` be one for a complete whole batch and zero otherwise.
For replicate `r`, compute the retained integer mass

    M_r = sum_b m_r,b c_b.

The frozen schedule additionally certifies at least 10 nonzero batch entries
in every multiplicity vector. Therefore any mask retaining at least 18 of 24
batches has positive retained mass. Verification nevertheless recomputes the
support and `M_r`; if either certificate fails, the bootstrap fails closed with
no redraw, seed change, replacement vector, or confirmatory authorization.
Otherwise the normalized batch weight is

    w_r,b = m_r,b c_b / M_r.

Implement the weighted statistic as a multiplicity-weighted numerator divided
by the exact positive integer `M_r`; do not independently round a stored weight
vector. The same complete mask and identical `w_r,b` are applied to both roles
and every paired endpoint. Because a retained batch always contains one anchor
from every quartile, each replicate computes one weighted mean per quartile and
then the equal average of the four quartile means. No surviving-anchor or
endpoint-specific mask is permitted.

For 18 through 23 complete batches this is explicitly conditional
complete-case inference. It estimates the frozen macro statistic conditional
on the observed complete-batch set. Masking preserves pairing and the
preregistered multiplicity schedule, but it does not correct nonrandom
missingness or recover the original all-24-batch estimand; that limitation is
reported with every decision-bearing result.

The unresampled point estimate uses equal weight over the same complete batches,
first within quartile and then across the four quartiles. The one-sided upper
endpoint is the observed `method="higher"` order statistic at 0.95 over all
9,999 values; the lower endpoint uses observed `method="lower"` at 0.05. No
interpolation is allowed.

Paired log-error contrasts use the protocol's extended-real convention without
an epsilon: equal zeros contribute zero, zero numerator against a positive
denominator contributes negative infinity, and a positive numerator against
zero contributes positive infinity. The artifact uses explicit canonical
extended-real tokens rather than non-standard JSON numbers. Any undefined
operation outside those three registered cases fails closed.

## 8. Decision states

Absolute candidate and reference gates use the frozen conservative observed
Q90/Q10 rules. A finite contrast additionally computes:

- the larger-domain efficacy upper endpoint for
  `log(E_g(high))-log(E_g(low))`, which passes only when strictly below
  `log(0.9)`; and
- every active noninferiority upper endpoint, each of which must be strictly
  below its own frozen margin.

The exhaustive result classes are:

- `low_endpoint_converged`: `K_low` passes absolute adequacy, all active
  noninferiority endpoints pass, `K_low` realizes strictly fewer candidate
  nodes than `K_high` in every complete batch, and larger-domain efficacy does
  not pass;
- `larger_endpoint_efficacious`: the efficacy rule passes, `K_low` realizes
  strictly fewer candidate nodes than `K_high` in every complete batch, and the
  smaller endpoint is not authorized as converged;
- `inconclusive_frontier`: the rules conflict or neither resolves;
- `absolute_only`: report `S_1024` and reference adequacy only;
- `right_censored_diagnostics`: report the frozen diagnostics only;
- `reference_inadequate`: the untouched `S_1024`-to-`R_top` support gate
  fails, so no finite or absolute claim is authorized;
- `descriptive_incomplete`: fewer than 18 complete batches; and
- `safety_or_resource_blocked`: a deterministic safety gate or frozen resource
  ceiling fails.

An invalid schedule, zero retained multiplicity, undefined extended-real
aggregation, or any other unusable-bootstrap condition aborts preparation and
fails verification before an installed decision. It is not converted into
`descriptive_incomplete`.

`confirmatory_claim_authorized=true` for either a clean
`low_endpoint_converged` or a clean `larger_endpoint_efficacious` result under
a `finite_contrast` lock. `convergence_claim_authorized=true` only for the
former, when at least 18 complete balanced batches remain, all 9,999 masked
bootstrap replicates have positive retained mass, audit reference adequacy and
all required endpoint/safety checks pass, `K_low` realizes fewer nodes in every
complete batch, timing provenance is complete, and process peak RSS is within
the frozen ceiling. Both flags remain false for a conflict, every other result
class, or any failed gate.

The authorization booleans written during decision derivation are not
transaction-effective by themselves. They become effective only after staged
and installed audit verification validates the complete, status-consistent
24-batch timing ledger and every required per-role timing field, then rederives
the decision and confirms the authorization invariants. Missing or malformed
timing aborts preparation or verification; it does not produce a different
decision class.

## 9. Resource provenance

The dense projection attached to each frozen domain is an analytic pre-workspace
quantity. Endpoint timing provenance is exactly the evaluator's
`candidate_selection_seconds`, `reference_selection_seconds`,
`candidate_build_seconds`, `reference_build_seconds`, and `solve_seconds`, plus
the runner's whole-batch elapsed time. Factorization is included in evaluator
build time; no separate factorization or metric timer is claimed. These fields
are observational and order/cache sensitive; they are reported, not
bootstrapped and not treated as an efficacy endpoint.

Peak RSS is the process-global Linux `ru_maxrss` high-water observation from
audit-derivation entry through all numerical solves, all 9,999 bootstrap
records, the decision, and scientific-payload serialization, immediately
before staging. If post-solve statistics or serialization first cross the
ceiling, the decision and scientific payloads are rebuilt as blocked. The
later filesystem staging and replay-verification pass are deliberately outside
this scientific resource gate and are not attributed to an endpoint. The
measured scope is stored verbatim. The runner does not claim endpoint-specific
RSS attribution. A resource-
contrast statement requires strict realized-node reduction and complete timing
provenance; it does not require or imply an endpoint-specific RSS reduction.
The per-batch elapsed ceiling remains a post-hoc gate, not an in-flight hard
interrupt.

## 10. Artifacts, verification, and replay

The fresh mode-0700 audit directory contains only mode-0600 unique-link regular
files from a fixed inventory:

- `audit_work_order.json`;
- `audit_fidelity.jsonl`;
- `batch_status.jsonl`;
- `bootstrap_statistics.jsonl`;
- `decision.json`;
- `execution.json`;
- `manifest.json`; and
- `LOCAL_ONLY_DO_NOT_PUBLISH`.

Node-level material never enters stdout. The CLI reports only aggregate status,
mode, reason, complete-batch count, and authorization. Scientific artifacts and
the complete decision enter the deterministic audit fingerprint; timings and
peak RSS enter only the complete manifest integrity seal.

Preparation uses a fresh output path, descriptor-relative no-follow writes,
exclusive creation, exact modes and link counts, per-file and directory
`fsync`, verified staging, no-replace rename, and conservative rollback. It has
no resume, append, overwrite, cache, or in-place repair mode. Output/input
overlap with any upstream bundle, declared raw source, plan, or lock is
prohibited.

One successful installation is one primary audit transaction. A later explicit
reproduction must use another fresh directory and is not a second confirmatory
reveal. Its scientific artifacts must be byte-identical to the primary result;
observational execution provenance may differ. A scientific mismatch fails
closed and neither run may be selected post hoc. As elsewhere in this chain,
unkeyed hashes and fresh-path checks prevent accidental drift but cannot stop a
malicious same-user from replacing a whole self-consistent chain; hostile
authenticity requires an external signature or immutable trusted store.

Audit verification repeats the full upstream and lock chain, rederives the
audit-only work order, validates every metric/safety/domain/alpha relationship,
recomputes all 9,999 masked bootstrap statistics and the final decision, and
checks every artifact byte and authorization invariant. It deliberately does
not rerun expensive numerical solves. Numerical reproduction is a separate
fresh audit transaction and can never retroactively replace the primary result.
