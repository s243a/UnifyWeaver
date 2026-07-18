# Semantically weighted leaky graph diffusion

## Status and scope

This document specifies a reusable graph geometry and numerical primitive. It
is not specific to mu-cosine training, a particular judge, or Pearltrees.
Potential consumers include routing, smoothing, covariance construction,
root-anchored metrics, and square-root Gaussian conditioning.

The implementation in src/unifyweaver/graph/leaky_diffusion.py is a dense
float64 correctness reference. It establishes the algebra and fail-closed API;
it does not claim a CUDA crossover or authorize a learned cross-item
covariance. The completed Stage-A repeated-judge source-power experiment
failed closed and unlocked no candidate enumeration, Nomic use, judge calls,
covariance deployment, independent batching, QR specialization, or CUDA
claim. This primitive therefore remains outcome-blind infrastructure.

## Separate topology, semantics, and grounding

Let the fixed node universe be V and the undirected graph support be E.
Topology decides whether a direct path exists. For an existing edge (i,j),
semantic embeddings z_i and z_j modulate its conductance:

    c_ij = c0_ij [
        epsilon + (1-epsilon)
        exp(-||z_i-z_j||^2 / (2 ell^2))
    ].

Here c0_ij is the base topological conductance, ell is the semantic length
scale, and epsilon is an optional minimum conductance fraction. The reference
implementation currently fixes c0_ij=1. A semantic model never creates an edge
between graph non-neighbours. With epsilon=0, an RBF value below float64 range
is represented as zero and the realized numerical graph loses that edge.
Applications that must preserve every retained topological path therefore use
a strictly positive, recorded epsilon.

This division is intentional:

- topology supplies domain-specific reachability and lineage;
- semantics controls resistance along paths that the graph already permits;
- grounding controls how far influence persists before leaking to a common
  bath.

For the current mu-cosine stack, a revision-pinned Nomic clustering embedding
is the preferred semantic candidate because e5 is already an input to the
deployed ranker. MiniLM is a sensitivity comparator. The general API accepts
arbitrary finite embeddings and performs no model loading.

## Electrical and thermal construction

Let W be the symmetric conductance matrix and define the combinatorial
Laplacian

    L = diag(W 1) - W.

Each node i may have a shunt conductance alpha_i to a common ground. Its
leakage current is voltage dependent:

    I_leak,i = alpha_i v_i,
    R_leak,i = 1 / alpha_i.

When alpha_i=0, the corresponding shunt resistance is infinite.

The grounded precision is

    J = L + diag(alpha).

The API allows:

- uniform weak leakage at every node;
- heterogeneous leakage;
- sparse boundary grounding, provided every connected component has a path to
  at least one positive shunt.

The last condition is checked on realized positive-conductance components
before factorization. The spectrum and Cholesky factor then enforce numerical
positive definiteness. An ungrounded component fails explicitly.

The transient assumes unit heat capacity. Its electrical RC analogue assumes
unit node capacitance C=I. For source q switched on at time zero,

    du/dt = -J u + q,
    u(0) = 0.

The transient and equilibrium responses are

    u(t) = J^-1 (I - exp(-tJ)) q,
    u(infinity) = J^-1 q.

For an initial impulse h rather than a maintained source,

    u(t) = exp(-tJ) h.

Without grounding, an ordinary connected Laplacian retains a constant mode
and an isolated impulse converges to a uniform temperature. Grounding removes
that mode: heat leaks to the bath and all impulse responses eventually decay.

## Green kernel and distance

The equilibrium Green kernel is

    G = J^-1.

For a unit current injected at source s, column s of G gives the equilibrium
voltage field. This is a source-conditioned closeness score, not by itself a
symmetric metric.

A symmetric grounded effective-resistance distance is

    d_R(i,j)^2 = G_ii + G_jj - 2 G_ij.

It is the quadratic form of e_i-e_j under G and therefore a Euclidean distance
whenever J is positive definite. If J=U^T U and X=U^-T, the same quantity is

    d_R(i,j)^2 = ||X[:,i] - X[:,j]||^2.

The reference computes distances in these factor coordinates, avoiding the
subtraction of nearly equal Green-kernel entries. It materializes G only for
small graphs. Large implementations should use selected linear solves or
sparse factorization rather than form a dense inverse.

With uniform leakage alpha,

    J^-1 = (L + alpha I)^-1
         = alpha^-1 (I + alpha^-1 L)^-1.

Thus the equilibrium kernel is exactly a regularized-Laplacian/resolvent
kernel up to a scalar. This explains the connection to the existing
small-graph resolvent reference in prototypes/mu_cosine/graph_geometry.py.
The new construction adds circuit units, semantic edge resistance,
heterogeneous grounding, source response, effective resistance, and a direct
precision root.

## Square-root conditioning

Because J is positive definite, Cholesky factorization gives

    J = U^T U.

Since G^-1=J, U is already an inverse-covariance square root for the physical
Green kernel. Equilibrium solves use two triangular solves:

    U^T y = q,
    U u = y.

No covariance inverse is needed.

Some statistical consumers require a unit-diagonal correlation rather than
the physical Green kernel. Let

    D = diag(G),
    C = D^-1/2 G D^-1/2.

Then

    C^-1 = D^1/2 J D^1/2
         = (U D^1/2)^T (U D^1/2).

Therefore scaling the columns of U by sqrt(diag(G)) produces the correlation
precision root used by a square-root/QR conditioner. Correlation
normalization is explicit because it discards physical voltage/temperature
units.

## Float64 numerical contract

This module is a correctness reference, not a best-effort solver. Before
Cholesky it requires:

- every realized positive-conductance component to reach a positive shunt;
- an equilibrium scale whose reciprocal is representable in float64; and
- reciprocal spectral condition at least sqrt(machine epsilon), approximately
  1.49e-8, corresponding to condition number at most about 6.71e7.

The minimum reciprocal condition is an explicit argument and is recorded on
the result. A caller may knowingly choose a weaker threshold for an experiment,
but cannot obtain it through hidden jitter or an accidental near-singular
factorization.

Transient calculations also avoid unnecessary loss of precision. The maintained
source response evaluates (1-exp(-t lambda))/lambda with expm1 rather than
subtracting equilibrium-sized vectors. A normalized heat kernel shifts the
spectrum separately within each disconnected component; the removed scalar
decay cancels exactly during diagonal normalization. Matrix symmetry cleanup
uses half-scaled operands, and every public result is checked for finiteness.
Direct construction validates shapes, symmetry, the graph identities, and the
stored precision factor.

## Leakage is not numerical jitter

Both leakage and diagonal jitter can improve a condition number, but they are
different objects:

- alpha is part of the graph model. It controls physical correlation range,
  equilibrium amplitude, and relaxation time.
- jitter is an implementation safeguard against floating-point failure after
  the model is fixed.

The reference applies no hidden jitter. A singular or insufficiently grounded
model fails. A future sparse/GPU backend may expose bounded scale-relative
jitter, but it must report it separately from alpha and may not use it to
silently change the intended diffusion.

For uniform alpha, increasing alpha raises the smallest precision eigenvalue
and shortens the diffusion range. Taking alpha toward zero recovers global
graph coupling and the Laplacian singular limit. Alpha and ell must be selected
on training-only data for any learned application.

## Directed graphs and principal lineage

The resistor/heat operator requires a symmetric conductance network. The
reference takes the undirected union of neighbor declarations on a fixed node
universe. A directed lineage application must document whether that
symmetrization is scientifically appropriate. Directional diffusion requires
a separate reversible construction or directed-Laplacian theory; silently
feeding an asymmetric transition matrix into this API is not allowed.

For candidate-lineage work, principal path identity must come from the
account-tagged materialized path, not the first parent in a sorted multi-parent
union. This primitive does not choose a lineage source.

## Rejected and deferred alternatives

- Raw shortest-path Gaussian kernel: rejected as a general covariance because
  an arbitrary graph shortest-path RBF is not guaranteed PSD.
- Semantic completed graph: rejected for the first implementation because
  embedding similarity would create shortcuts and erase the topology/semantic
  separation. Weak semantic k-nearest-neighbor edges are a separate future
  model.
- Raw embedding distance as resistance: rejected because its units and zero
  behavior are unsuitable. A nonnegative monotone conductance transform is
  explicit and bounded; a positive epsilon is required when every retained
  topological edge must remain numerically present.
- Arbitrarily choosing distant nodes as ground: supported only when the
  boundary is externally defined. Uniform weak leakage is the default because
  defining distant by the metric being estimated would be circular.
- Symmetric normalized Laplacian for the circuit: retained for spectral
  references, but not used for physical KCL. The combinatorial Laplacian
  preserves edge conductance units.
- Explicit matrix inverse: rejected. The precision root and triangular solves
  are primary; dense G is a reference output.
- Immediate sparse/CUDA specialization: deferred until graph sizes and a
  matched end-to-end benchmark justify it. Likely paths are sparse Cholesky or
  preconditioned conjugate gradients for equilibrium, and
  Lanczos/Chebyshev methods for heat action.
- Treating G as learned measurement covariance without evidence: rejected.
  Statistical promotion still requires train-only fitting and held,
  dependence-aware predictive validation.

## Reference verification

tests/test_leaky_diffusion.py verifies:

- semantic modulation only on existing edges, including extreme float64 scales;
- the analytic two-node circuit and its Green kernel;
- Kirchhoff residuals and superposition;
- uniform-leakage resolvent equivalence;
- failure of an ungrounded component and unknown sparse-mapping keys;
- default and explicitly overridden reciprocal-condition gates;
- failure of unrepresentable equilibrium scales;
- heat semigroup and relaxation to equilibrium;
- component-wise normalization after large-time heat underflow;
- cancellation-stable tiny-time step response;
- monotonic resistance under semantic separation;
- factor-coordinate resistance parity with unit-current energy;
- correlation precision-root parity;
- direct-constructor invariants, immutable arrays, and fail-closed inputs;
- absence of any explicit matrix inverse.

The next engineering PR, if warranted, should add sparse operators and a
matched CPU/GPU crossover benchmark without changing this statistical object.
