# Prospective protocol: synthetic exact multi-port Schur reference

## Status and decision boundary

**PROSPECTIVE, SYNTHETIC ONLY.** Freeze this protocol before evaluating the
implementation on its required fixtures. It licenses a dense CPU-float64
correctness reference for exact elimination of a represented graph component.
It does not license a Pearltrees/private-data load, a sparse approximation, a
filing experiment, training, deployment, or publication of a private-derived
operator.

The decision is whether one implementation can:

1. eliminate a grounded multi-port interior without forming an inverse;
2. reproduce the full system's boundary response;
3. expose the induced transfer-conductance and residual-shunt ledger; and
4. fail closed when the result is not a numerically valid grounded
   resistor-network operator.

Passing every required fixture authorizes the routine as an outcome-blind
exact reference for a later, separately preregistered approximation study. It
does not authorize a real private-data run.

## Frozen component snapshot

Each fixture first uses the existing deterministic exterior traversal to
produce one immutable `ExteriorComponent` snapshot. The snapshot contains:

- one or more canonical retained ports `B` and at least one canonical exterior
  node `H`;
- canonical, unique boundary-to-exterior cut edges;
- canonical, unique undirected edges internal to `H`;
- canonical edges from `H` to the clamped outside bath;
- a verified SHA-256 fingerprint over those exact topology fields; and
- a separate finite nonnegative intrinsic-leakage scalar, sparse node mapping,
  or `H`-aligned vector.

One positive finite uniform topological conductance `c0` applies to every
listed graph edge. Per-edge conductances and semantic modulation are deferred.
The component contains no embeddings, titles, URLs, placements, judge
outcomes, corpus IDs, private IDs, or machine-local paths. The reducer must use
the frozen component as its sole topology authority; rereading the source graph
after discovery is forbidden.

## Frozen exact identities

Assemble the exterior precision `J_HH` from intrinsic leakage, internal-edge
Laplacian terms, cut-edge diagonal terms, and outside-bath diagonal terms.
Let `C >= 0` be the port-to-exterior coupling matrix and

\[
\beta=C\mathbf 1,\qquad D_B=\operatorname{diag}(\beta).
\]

Solve, without materializing `J_HH^-1`,

\[
X=J_{HH}^{-1}C^T.
\]

The exact exterior return and reduced exterior boundary contribution are

\[
B_H=CX=CJ_{HH}^{-1}C^T,\qquad
Q_H=D_B-B_H.
\]

If a caller has an independently assembled public-side boundary precision
`J_P`, the full reduced precision is `S=J_P+Q_H`. `J_P` is not part of this
primitive. The direct full-graph parity fixture assembles it independently so
the test cannot pass merely by repeating the reducer's construction.

For a valid grounded resistor network, recover the joint multi-terminal
ledger

\[
\kappa_{ij}=-(Q_H)_{ij}\quad(i\ne j),\qquad
q=Q_H\mathbf 1,
\]

and verify

\[
Q_H=L(\kappa)+\operatorname{diag}(q).
\]

`kappa` is one joint operator, not independently estimated pairwise
`1/R_eff` edges. `q` must be retained: interior leakage and represented bath
shunts cannot generally be expressed using boundary-to-boundary edges alone.

## Numerical contract and fail-closed gates

Use CPU `float64`, the existing dense reference backend, and no jitter,
pseudoinverse, or explicit inverse. Roundoff-only symmetry cleanup is allowed.
`B_H` and `Q_H` remain unaltered and fingerprinted. After the material-negative
gates, the derived public nonnegative bridge, self-return, and residual-shunt
views may canonicalize a tolerated negative roundoff value to zero; every raw
value remains recoverable from fingerprinted `B_H` or `Q_H`.

Let

\[
s_B=\max(1,\max_{ij}|(B_H)_{ij}|,\max_{ij}|(Q_H)_{ij}|,
           \max_i|\beta_i|),
\]
\[
\tau_{\rm sign}=\max(10^{-12}s_B,64\epsilon_{64}s_B),\qquad
\tau_{\rm psd}=10^{-10}s_B.
\]

Every expected-valid fixture must pass all of these gates:

- `J_HH`, `B_H`, and `Q_H` are finite and symmetric within the recorded
  tolerances;
- `J_HH` is a Cholesky-factorable SPD M-matrix;
- the reciprocal condition number of `J_HH` is at least
  `sqrt(float64_epsilon)` unless the caller supplies a stricter floor;
- the Cholesky reconstruction uses `rtol=1e-11`, `atol=1e-12`;
- the relative infinity-norm residual of `J_HH X = C^T` is at most `1e-10`;
- `B_H` has no entry below `-tau_sign` and no eigenvalue below `-tau_psd`;
- `Q_H` has no eigenvalue below `-tau_psd`, no positive off-diagonal above
  `tau_sign`, and no row-sum shunt below `-tau_sign`;
- reconstruction from `kappa` and `q` agrees with `Q_H` within
  `rtol=1e-12` and the cut-mass ledger closes within tolerance; and
- every column of `J_HH^-1 C^T` is nonnegative and its all-one boundary
  harmonic extension remains in `[0,1]`, each within `1e-10`.

`J_HH` must be SPD. `Q_H` need only be PSD and may be singular when the
represented exterior has no intrinsic leakage or outside-bath route. The
routine must not invent a gauge, jitter, or grounding term for that valid
case.

Values inside the tolerance remain in the recorded operator; they are not a
license to repair an invalid input. Any failed gate makes that expected-valid
fixture a failure. Every expected-invalid fixture must be rejected before
emitting a usable operator.

## Required synthetic fixtures

The frozen additions to the existing bounded-diffusion suite contain:

1. a grounded three-port star with analytic `B_H`, `Q_H`, transfer, self-return,
   and residual-shunt values, plus a check that `B_H[i,j]` is not generally
   `1/R_eff(i,j)`;
2. a four-port branched/cyclic exterior whose `J_P+Q_H` and multi-right-hand-
   side responses match an independently assembled full-graph Schur solve;
3. connected parallel multi-port paths whose exact induced transfer exceeds
   one ordinary branch conductance and is not capped;
4. an equivalent graph with reversed insertion order and heterogeneous node
   identifiers, yielding the same canonical component and reduction
   fingerprints;
5. a guard that rejects any explicit inverse and verifies immutable outputs
   plus the recorded numerical diagnostics;
6. an allowed exterior with an outgoing bath edge, verifying the analytic
   residual shunt; and
7. a one-port component plus negative, nonfinite, unknown-node, and
   insufficient-condition input rejections.

The pre-existing two-port analytic, parallel-path, retained-edge, topology-only,
and numerical-failure tests remain regression coverage for the delegated
benchmark path. All fixtures are deterministic synthetic code; no random graph
generator enters the decision-bearing suite.

## Output and provenance

Each successful result exposes canonical ports and exterior nodes, immutable
`J_HH`, `C`, harmonic extension, `beta`, `B_H`, and `Q_H`, along with
`kappa`, self-return, transfer degree, and residual shunt views. It records
exterior spectral extrema, reciprocal condition and required floor, Cholesky
reconstruction error, solve residual, maximum-principle violation, minimum
eigenvalues of `B_H` and `Q_H`, and ledger error.

The reduction SHA-256 binds the verified component fingerprint, canonical node
tokens, `c0`, leakage vector, required condition floor, and scientific outputs
using hexadecimal float64 serialization. Equivalent discovery/input order
must reproduce that fingerprint. Synthetic records may be committed. This
does not authorize persistence or publication of a real private component or
operator; those remain private under `DESIGN_private_boundary_expert.md`.

## Acceptance and explicit nonclaims

The reference passes only when every expected-valid fixture passes every gate,
every expected-invalid case is rejected, the existing delegated two-port
regressions remain green, and equivalent input order reproduces the scientific
fingerprints. There is no favorable-subset report or tolerance retuning after
results.

This protocol deliberately defers:

- sparse support selection, top-`K` truncation, and sparse-DtN optimization;
- semantic candidate filtering or learned conductance;
- effective-resistance candidate search beyond analytic fixture diagnostics;
- private/public loaders and an owner-authorization manifest;
- harvester authentication repair and any Pearltrees snapshot;
- real-component structural-shadow or fidelity studies;
- private-expert representation alignment, training, routing, or evaluation;
- covariance promotion, filing-performance, CUDA, and scale claims; and
- differential privacy, release testing, export, or publication.

A later sparse-approximation protocol may consume this exact reference only
after its support, resource budget, loss, component-level split, stopping
rule, and untouched evaluation are frozen prospectively.
