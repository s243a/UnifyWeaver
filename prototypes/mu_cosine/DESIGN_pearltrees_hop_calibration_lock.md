# Pearltrees HOP leakage calibration and immutable lock

**Status:** prospective implementation contract. This stage consumes an
accepted no-solve HOP plan, uses only its eight calibration batches, and
freezes the scalar topology leakage and the later audit work order. It is not
an audit and contains no confirmatory result.

## 1. Why calibration is its own transaction

The no-solve plan freezes topology, anchors, shells, budgets, and statistics,
but deliberately leaves two quantities unknown: the physical uniform leakage
`alpha_top` and the HOP contrast that the untouched audit will evaluate. Those
quantities may be learned from the 32 registered calibration anchors only.

The transaction is one-way:

1. reverify the complete declaration -> two snapshots -> receipt -> HOP plan
   chain;
2. derive a calibration-only work order;
3. prove every calibration reference is numerically admissible at `alpha=0`;
4. calibrate 32 separate anchor requirements and take their maximum;
5. apply that one maximum unchanged to every calibration candidate and
   reference;
6. evaluate the preregistered adequacy rules and freeze the audit mode; and
7. atomically install and verify an immutable local-only lock.

No audit solve is callable before step 7 succeeds. Calibration results are
tuning results; they are never reused as confirmatory evidence.

## 2. Executable audit isolation

Full plan verification necessarily rederives the outcome-blind audit topology.
That integrity operation is separate from numerical calibration. The numerical
worker receives a derived work order containing exactly eight rows named
`calibration-00` through `calibration-07`, four unique anchors per row (one per
frozen degree quartile), the four frozen roles, the protected set, exact
Dirichlet boundary ledger, and each anchor's registered radius-3 shell.

The work-order builder rejects an audit-tagged row, an unknown role, a missing
or duplicate anchor, an unbalanced quartile, a non-nested domain, or anything
other than 8 batches and 32 anchors. It reads no labels, placements, judge
outputs, embeddings, filing metrics, prior response cache, or audit response.
The worker consumes the frozen node order directly; it never reruns HOP
selection with another comparator.

For each frozen domain, complete incident adjacency is recaptured from
canonical attempt A. Recomputed internal edges, real retained-to-omitted cut
edges, and integer `beta` must reproduce the plan's exact boundary row. A
traversal collision remains path convergence or a cycle, never a shunt.

## 3. Zero-alpha admissibility is a gate

Uniform leakage is a physical screening parameter, not numerical jitter. For
each calibration `R_top`, first construct

    J_0 = L_ind + diag(beta)

with no intrinsic leakage. `J_0` must already be grounded, positive definite,
an M-matrix to tolerance, and at least as well conditioned as the plan's
minimum reciprocal condition. Its Cholesky reconstruction, multi-anchor solve
residual, nonnegative Green response, maximum principle, boundary harmonic,
and Kirchhoff current balance must pass.

If this gate fails, calibration is installed as `blocked`; the implementation
may not repair the system by calling a conditioning-derived value physical
leakage. After the gate passes, the spectral calibrator must report
`numerical_minimum_added_leakage == 0`. A nonzero value is also a block.

This precedence also makes an ordinary fully exhausted, otherwise ungrounded
connected component a `blocked` phase-one calibration: its zero-alpha
Laplacian has the constant gauge mode. Calling the exhausted component an
exact structural reference does not silently turn physical alpha into a
numerical gauge fix. The general exhausted-endpoint `absolute_only` rule can
apply only when the frozen zero-alpha operator is independently grounded; a
future gauge-aware whole-component calibration requires a prospective
amendment and separate parity tests.

This rule is deliberately stricter than merely obtaining an SPD matrix after
adding alpha. It prevents an ill-conditioned reference from making a larger,
more localizing alpha appear scientifically measured and thereby making small
domains look artificially converged.

## 4. Two-pass leakage calibration

For anchor `s` in its own frozen `R_top`, the registered shell is the nonempty,
strictly interior set at graph-hop radius three. Calibration uses the raw
killed-walk ratio

    a_s(alpha) = max_{i in shell_s} G_alpha(i,s) / G_alpha(s,s)

and chooses the smallest alpha, to the frozen relative bracket tolerance, for
which `a_s(alpha) <= exp(-1)`. Correlation-normalized Green entries are not
used because their dependence on leakage need not be monotone.

Every anchor records a minimality certificate: the final lower and upper
alpha bracket, attenuation at both endpoints, relative width, and evaluation
count. Unless zero already meets the target, the lower endpoint must miss and
the upper endpoint must meet. The upper endpoint is the anchor requirement.
The radius-three value controls the deterministic chain bracket seed as well
as naming the scientific shell; a domain cutoff is not substituted silently.

Each four-anchor batch shares one `numpy.linalg.eigh` decomposition, but its
anchors retain separate shells and requirements.

"Separate" is an algorithmic statement, not a statistical-independence claim:
the registered anchors are unique, but their bounded graph domains may overlap.

After all 32 succeed,

    alpha_top = max_s alpha_s.

No mean, quantile, lower confidence bound, survivor-only maximum, or per-batch
final value is licensed. A failed anchor blocks the transaction. A second pass
rebuilds all calibration roles with exactly

    J_D = L_D + diag(beta_D + alpha_top).

There is no candidate-specific recalibration and neither `beta` nor alpha is
added twice. Per-anchor attenuation and the radial tail-envelope crossing are
recomputed at global `alpha_top`, so over-grounding relative to less demanding
anchors remains visible.

## 5. Calibration fidelity and selection

Within each batch the shared `R_top` factorization is built once and reused for
`S_256`, `S_512`, and `S_1024`. Each candidate is factored once. This is a
performance requirement, not a statistical relaxation: every result is
algebraically identical to an independent candidate/reference evaluation.

Reference adequacy is the all-32-anchor conservative tail comparison of
`S_1024` with `R_top`. Candidate raw, normalized, ranking, source-diagonal,
and (when preregistered) selected effective-resistance endpoints aggregate over
all 32 calibration anchors. The protected boundary-harmonic endpoint
aggregates the eight batch maxima. Upper tails use observed higher order
statistics; lower tails use observed lower order statistics. Interpolated
quantiles are prohibited.

The smallest absolutely adequate member of `{256,512,1024}` is `K_low`.
Ordinarily `K_high` is the next nominal candidate. If `K_low=1024`, the high
endpoint is `R_top` and the later audit is absolute-only. If no candidate is
adequate, the exact diagnostic role list is frozen before audit. If a connected
component is exhausted and nominal endpoints are node-identical, the phase-one
zero-alpha gate blocks before selection. The lock records no nominal-label
efficacy or resource benefit; gauge-aware handling is future work.

An effective-resistance endpoint is present exactly when the plan froze that
arm as enabled. A prospectively omitted arm is absent from both adequacy and
the later intersection test; it is never dropped after seeing a failure.

## 6. Lock modes and authorization

Exactly one mode is installed:

- `finite_contrast`: `K_low` is 256 or 512 and `K_high` is the next distinct
  registered endpoint;
- `absolute_only`: `K_low` is 1024 with `R_top` as the endpoint; no larger-domain
  efficacy claim is licensed;
- `right_censored_diagnostics`: no candidate is adequate and the audit may run
  only the frozen diagnostic roles without a convergence claim; or
- `blocked`: provenance, completeness, zero-alpha, numerical, reference, or
  resource failure; no audit solve is authorized.

Reference inadequacy always produces `blocked`. For the first three modes,
`audit_solve_authorized=true` authorizes only the bound audit transaction; it
does not authorize a result claim. Confirmatory claim authorization remains
false until the untouched audit satisfies its own complete-batch, efficacy,
noninferiority, safety, and resource rules.

## 7. Numerical and resource provenance

The plan distinguishes these LAPACK roles:

- alpha calibration spectrum: `numpy.linalg.eigh`;
- condition diagnostics: `numpy.linalg.eigvalsh`;
- decision factor: `numpy.linalg.cholesky`; and
- response solve: `numpy.linalg.solve` on the stored triangular root.

The lock records NumPy/Python versions and a nonempty path-free BLAS identity
(`user_api`, `internal_api`, `prefix`, `version`, `threading_layer`, and
`architecture`). Absolute shared-library paths are excluded. `threadpoolctl`
must observe one BLAS thread while both calibration passes execute; an empty
identity or a different thread count blocks before numerical work.

Float64, zero bath temperature, zero base intrinsic alpha, no embeddings, unit
conductance, no closure, no hidden jitter, the bisection tolerance/evaluation
cap, and every residual/root/symmetry tolerance come from the bound plan.
Peak RSS and phase timings are measured and checked against the frozen resource
ceiling. They are observational provenance outside the deterministic
scientific fingerprint but inside the complete manifest integrity seal.
The frozen per-batch elapsed ceiling is checked after each calibration and
fidelity batch. It is not advertised as a hard interrupt of an in-flight LAPACK
call; a future hard deadline requires process isolation. Exceeding the observed
elapsed ceiling still blocks audit authorization.

## 8. Artifacts and transaction

The fresh mode-0700 lock directory contains only mode-0600, unique-link,
regular files:

- `calibration_work_order.json`;
- `anchor_calibrations.jsonl`;
- `batch_calibration.jsonl`;
- `calibration_fidelity.jsonl`;
- `selection.json`;
- `execution.json`;
- `manifest.json`; and
- `LOCAL_ONLY_DO_NOT_PUBLISH`.

The deterministic lock fingerprint binds the complete plan-manifest byte
record, calibration work order, scientific output records, implementation and
protocol records, backend identity, alpha, and audit mode. The full manifest
seal additionally binds observational execution provenance. Merely copying
`plan_fingerprint` is insufficient.

Preparation reverifies the full upstream chain before work and again before
installation. Staging uses descriptor-relative no-follow exclusive writes,
per-file and directory `fsync`, no-replace rename, and conservative rollback.
The output may not overlap the plan, receipt, either attempt, the declaration
bundle, policy, or any declared raw source. This phase has no cache input or
cache-root CLI surface; a future cache-backed adapter must add and test its own
overlap guard prospectively. Node-level material never enters stdout.

Lock verification repeats full upstream and plan verification and checks every
bound byte and authorization invariant. It rederives the 32-anchor, eight-batch,
and global leakage maxima; minimality brackets; factor/reuse ledger; physical
metric ranges and order statistics; frozen candidate/reference/alpha
fingerprints; and the RSS-dependent authorization decision. Per-artifact read
ceilings are fixed at 128 MiB for the work order, 16 MiB for fidelity, 8 MiB
each for anchor and batch evidence, and 4 MiB each for selection and execution;
the manifest and marker have separate smaller bounds. An advertised size above
its ceiling is rejected before the payload is read.

Verification deliberately does not repeat the expensive numerical calibration.
Thus the unkeyed chain detects accidental or unsynchronized replacement but
does not authenticate a malicious same-user who replaces the entire
self-consistent chain. Hostile authenticity still requires an external
signature or immutable trusted digest.

## 9. Sequencing, scale, and non-claims

The HOP plan binds its repository commit and scientific-file hashes. Therefore
the real private plan and lock must be generated from the final landed
implementation commit (or its exact detached checkout), after this runner is
reviewed. Generating a real plan from an earlier commit and attempting to
reinterpret it with later code is prohibited.

The dense phase is for bounded references on the current host. It is not a
million-node algorithm, CUDA crossover result, covariance estimate, filing
improvement, graph-judge deployment, or whole-corpus truth claim. A larger
graph must still be reduced by the already frozen topology-only planning step;
scaling the planner or numerical backend requires a prospective adapter rather
than a silent resource downgrade.
