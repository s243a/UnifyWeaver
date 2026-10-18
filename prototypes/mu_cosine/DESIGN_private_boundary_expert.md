# Private boundary expert with sparse resistive interface

## Status and scope

**DESIGN ONLY — PARKED.** This document records a possible local
personalization architecture for Pearltrees data that is genuinely private.
It does not authorize private-data ingestion, training, inference service,
external processing, export, deployment, publication, or a claim that a
private-data-derived model is safe to release.

The first possible consumer is a local filing expert. The public model stays
frozen. A separate private expert learns from private nodes and agrees with the
public representation at a small, explicit public/private interface. When a
large private interior must be removed from a bounded numerical solve, its
exact grounded Schur/Dirichlet-to-Neumann response is the reference. A sparse
network of nonnegative resistive bridges and shunts may approximate that
response locally.

Work remains parked until the harvester's masked-authentication path is
fail-closed, a fresh snapshot has authenticated revision-bound visibility
evidence, and the owner has supplied a snapshot-bound authorization manifest
for the private roots and permitted local use. A private/restricted flag is a
taint requiring protection; it is not by itself consent to train. The
privacy-aware acquisition gate is specified in
[`DESIGN_pearltrees_stem_harvest_plan.md`](DESIGN_pearltrees_stem_harvest_plan.md).

## Decision

Treat public and private data as separate trust domains, not ordinary training
and test splits of one publishable corpus:

1. keep the public encoder and public graph artifacts frozen;
2. train a private expert only inside the local private environment;
3. use shared public boundary anchors to align the private representation;
4. use private graph physics, not filing labels, to define and approximate the
   private interior's boundary response; and
5. keep every private-derived artifact private, including an apparently
   content-free boundary operator.

Graph topology is the primary geometry. Frozen local embeddings may reduce the
search cost for plausible boundary pairs, but they do not create physical
connections or set bridge conductance. A candidate bridge must represent an
actual path through the private interior, and its strength must come from that
graph response.

## Trust domains and threat model

Define five trust/data classes:

- **Public base:** public graph evidence, frozen public embeddings, and the
  frozen public model. This domain must remain unchanged by private training.
- **Boundary `B`:** explicitly public nodes incident to, or deliberately shared
  with, the private domain. Boundary membership is authenticated evidence, not
  a title-based guess. The public node content may be public, but the selected
  boundary set, pairings, order, targets, residuals, and routes are private
  relational metadata.
- **Private interior `H`:** owner-authorized private nodes, edges, titles, URLs,
  placements, and any local labels or prompts derived from them.
- **Quarantine `U`:** unknown, masked-authentication, conflicting, malformed,
  or unauthenticated visibility. It is excluded from both training and boundary
  construction until authenticated revalidation.
- **Private artifacts:** private embeddings, optimizer state, adapters,
  checkpoints, caches, logs, exact Schur operators, sparse bridge support and
  weights, shunts, effective resistances, spectra, harmonic measures,
  diagnostics, manifests, fingerprints, and evaluation reports.

The last category remains private even when it contains no raw text. A boundary
operator or sparse bridge graph can reveal connectivity, branching, shared
parents, relative path strength, and the presence of hidden topics. A private
checkpoint can memorize titles or placements. Boundary agreement is not
anonymization, differential privacy, or evidence that the artifact is safe to
publish.

Consequently:

- no hosted embedding, LLM, judge, telemetry, or training API receives private
  nodes or private-derived representations;
- local embedding models are the preferred first representation tool;
- a local language model, including a Phi-family model, is optional and not on
  the critical path;
- no gradient, adapter, distilled target, bridge ledger, or diagnostic produced
  from private data is merged back into the public base; and
- all private inputs and outputs live outside Git worktrees in access-controlled
  local storage, with local-only provenance and generic console errors.

The external-processing prohibition includes otherwise-public payloads whose
selection, ordering, pairing, batching, or requested label was derived from
`B`, `H`, or any private artifact. It also includes hosted coding/analysis
assistants and crash-reporting services. A hosted public-only embedding or
label catalog may enter the private workspace only when it was generated
independently from a frozen public-only manifest; its private join happens
locally.

The existing public scrub-everywhere pipeline in
[`privacy.py`](privacy.py) and the snapshot contract in
[`DESIGN_pearltrees_diffusion_snapshot.md`](DESIGN_pearltrees_diffusion_snapshot.md)
remain unchanged. There is no `include-private` escape hatch. A future private
implementation requires a separate loader and one-way dependency: frozen
public artifacts may enter the private workspace, but no private association,
gradient, parameter, or metric flows back. Hash the public base before and
after every private run and require byte identity.

Any future private run root must be outside Git and synchronized folders,
reject symlink/hardlink escapes, install artifacts atomically, and disable
network access and model telemetry during private processing. Verify effective
access control on the underlying filesystem: mode-0700 directories and
mode-0600 single-link files with POSIX ownership on a Linux filesystem, or
restrictive Windows ACLs on Windows/DrvFS. `chmod` metadata alone is not proof
of protection on DrvFS. WSL and Windows on this machine are one host trust
boundary, not isolation from each other. If visibility or owner authorization
changes, invalidate and destroy every transitive private artifact and rebuild;
do not claim surgical unlearning.

Provider promises about retention do not relax this boundary. Public visibility
also does not by itself authorize third-party processing, as the harvest plan
already records.

## Representation alignment

Let `z_P(b)` be the frozen public embedding of boundary node `b`, and let
`z_H(b)` be the private expert's representation of the same anchor. A first
training objective is

\[
\mathcal L = \mathcal L_{\mathrm{private}}
  + \lambda_B \sum_{b\in B_{\mathrm{train}}}
      \lVert T z_H(b)-z_P(b)\rVert_2^2
  + \lambda_R \mathcal R(T).
\]

`L_private` is a local objective over private graph structure, private filing
examples, or self-supervised private text. Boundary agreement alone cannot
identify useful behavior in the private interior. It supplies a coordinate
interface, not the private expert's training signal.

Start `T` with a rotation/orthogonal Procrustes family. Rotation-first alignment
preserves distances and avoids inventing scale or anisotropy before the data
support it. If a richer map is later licensed, record its gauge, singular
values, conditioning, and held-out boundary error. Pin the gauge explicitly;
otherwise changes can move arbitrarily between the private representation and
`T` while leaving the alignment loss unchanged.

Boundary anchors used to fit `T` must be separated from boundary audit anchors
by the exposure blocks defined below. Alignment quality is descriptive unless
an independent downstream private filing target also improves.

## Exact electrical reference

This section inherits the grounded-operator and bounded-domain semantics in
[`LEAKY_GRAPH_DIFFUSION.md`](../../docs/design/LEAKY_GRAPH_DIFFUSION.md) and
[`LOCAL_GROUNDED_DIFFUSION.md`](../../docs/design/LOCAL_GROUNDED_DIFFUSION.md).
It changes the trust domain and future consumer, not the underlying circuit
identities.

### Grounded private system

The electrical graph contains only prospectively allowed, frozen
folder-to-folder topology: containment and any explicitly admitted structural
cross-link relations. Bookmark-to-folder placements, placement-derived counts,
judge outcomes, and filing labels are supervised targets and are excluded from
`J`, component discovery, support selection, `alpha`, `beta`, and every exact or
sparse conductance fit. The relation policy is part of the private manifest.

Use a symmetric nonnegative-conductance graph with explicit leakage to a bath.
Its precision is

\[
J=L+\operatorname{diag}(\alpha+\beta),
\]

where `alpha` is model leakage and `beta` represents exact Dirichlet grounding
of omitted edges. Numerical jitter, if ever required, is separate provenance
and is never relabeled as physical leakage.

Partition the retained interface and private interior as

\[
J=\begin{bmatrix}
J_{BB} & J_{BH}\\
J_{HB} & J_{HH}
\end{bmatrix}.
\]

When `J_HH` is grounded SPD, eliminating `H` gives the exact boundary precision

\[
S=J_{BB}-J_{BH}J_{HH}^{-1}J_{HB}.
\]

This is the joint multi-terminal Schur/Dirichlet-to-Neumann reference. It is not
formed by solving boundary pairs independently.

For an interface-only accounting, write `C=-J_BH >= 0`, let
`D_B=diag(C 1)` be the original boundary-to-private cut degree, and separate
the public-side boundary precision `J_P`. Define the full-Dirichlet boundary
block and the private return term as

\[
J_D=J_P+D_B,
\qquad B_H=CJ_{HH}^{-1}C^T.
\]

`B_H` is positive semidefinite and entrywise nonnegative. The exact private
contribution and total reduced precision are

\[
Q_H=D_B-B_H,
\qquad S=J_P+Q_H=J_D-B_H.
\]

For a valid grounded resistor network, `Q_H` is a symmetric positive
semidefinite M-matrix. Its off-diagonal transfer conductances and residual bath
shunts are

\[
\kappa_{ij}=-(Q_H)_{ij}\ge 0\;(i\ne j),
\qquad q_i=(Q_H\mathbf 1)_i\ge 0,
\]

so `Q_H=L(kappa)+diag(q)`. The shunt term matters: private leakage or omitted
private exterior cannot generally be represented by boundary-to-boundary edges
alone.

V1 is grounded only. An ungrounded private component requires a gauge or
pseudoinverse and is deferred rather than silently regularized.

### Equivalent resistance is a diagnostic, not an edge recipe

For a grounded boundary operator `Q` whose inverse exists,

\[
R_{ij}=(e_i-e_j)^TQ^{-1}(e_i-e_j)
\]

is a useful boundary distance. The operator may be `Q_H` when the private
contribution is itself SPD or the full grounded `S` when the public-side
operator supplies the grounding. Record which operator was used.

In the special two-terminal ungrounded series case, `1/R_ij` is the exact
replacement conductance. That identity does **not** license setting every edge
of a multi-terminal boundary graph to `1/R_ij`. Pairwise effective resistances
do not uniquely specify a grounded multi-terminal response, and independently
adding them can double-count shared private paths. Equivalent resistance can
rank candidates and diagnose an approximation; the joint Schur operator is the
reference.

## Sparse private-boundary approximation

The exact private return `B_H`, and therefore `Q_H`, is generally dense. If the
interface is too large, approximate it with a single jointly fitted return
operator

\[
\widehat B_H=\operatorname{diag}(\widehat\sigma)
  +A(\widehat\kappa),
\qquad \widehat\kappa_{ij}\ge0,\;\widehat\sigma_i\ge0,
\]

where `A(kappa)` has the symmetric off-diagonal bridge weights and zero
diagonal. Preserve the existing exact cut-mass ledger at every boundary port:

\[
(D_B)_{ii}=\beta_{\mathrm{res},i}+\widehat\sigma_i
  +\sum_j\widehat\kappa_{ij},
\qquad \beta_{\mathrm{res},i}\ge0.
\]

The sparse reduced model is then

\[
\widehat S=J_P+L(\widehat\kappa)
  +\operatorname{diag}(\beta_{\mathrm{res}})
  =J_D-\widehat B_H.
\]

Bridges replace a funded share of the original private cut shunt; they are
never added on top of an unchanged `D_B`. Require the fitted return operator to
be positive semidefinite and recheck the fitted reduced precision as an SPD
M-matrix. This is the ledger already
specified for future graph-derived closure in section 7 of the
[`bounded-diffusion fidelity protocol`](PROTOCOL_bounded_diffusion_fidelity.md).
The fit is one coherent sparse approximation to the joint operator, not a
splice of per-anchor solves or independently truncated pairs whose diagonal
ledger was not recomputed.

### First measure whether bridges are needed

Before fitting an approximation:

1. retain common public parents and other graph junctions when the fixed domain
   budget permits;
2. discover private exterior components with one deterministic traversal and a
   memoized visited set;
3. collect all boundary ports incident to each actual private component; and
4. report exact induced off-diagonal mass, shunt mass, component port counts,
   and response change against the plain grounded boundary.

A traversal collision with an already visited node denotes path convergence or
a loop, not a cut point. It neither grounds the path nor creates a bridge.
Including common parents may remove most apparent cross-cut connections. An
empty or numerically negligible private bridge operator is a valid result and
ends this line without manufacturing an experimental arm.

### Candidate support

Generate sparse support graph-first:

- reuse deterministic component traversal, cycle memorization, common-parent
  stopping, and distance/path precedents already present in the repository;
- consider a pair only when both ports reach the same represented private
  component through actual graph edges;
- use exact two-port reduction only for a genuinely isolated two-port component;
  handle a multi-port component jointly; and
- freeze top-`K` support and resource limits before inspecting any filing
  outcome.

Frozen local semantic embeddings may prescreen or order graph-supported
candidates because semantically close ports can make bridge search cheaper.
They may not create a pair lacking a represented private path, set `kappa`, or
rescue a graph-derived approximation that fails its structural gates. A local
LLM is unnecessary for this geometry.

Expected-branching, topic-generality, or entropy-derived resistors are plausible
future proposals, analogous to earlier mind-map heuristics. They require a
separate prospective comparison to the exact private Schur response. They are
not part of the primary graph-first approximation.

### Fit and validation

At fixed support, fit nonnegative symmetric conductances and nonnegative shunts
against exact outcome-blind private folder-graph quantities only. Candidate
objectives include a weighted combination of:

- relative Frobenius or energy-norm error in `Q_H`;
- raw Green-response error between the full grounded `S` and `S_hat` on
  frozen multi-anchor probes;
- grounded effective-resistance error from that same full-operator pair; and
- preservation of candidate ranks or screening tails.

All hyperparameters, including support size and regularization, are selected
against an exact small/private Schur reference or a larger graph-only bounded
domain. Filing placements, judge scores, and downstream ranking labels never
calibrate bridge strength. Use one union-of-anchors domain and one factorization
whenever cross-anchor responses are consumed jointly.

For every exact and approximate solve, record and verify:

- symmetry, nonnegative conductance and shunt ledgers, M-matrix signs, and
  positive definiteness/semidefiniteness as applicable;
- Cholesky or solve residuals and reciprocal condition estimates;
- maximum-principle bounds and current conservation;
- exact-versus-sparse operator, raw response, resistance, and rank errors;
- total induced transfer and shunt mass relative to the original cut mass;
- node/edge counts, factorization reuse, runtime, and peak memory; and
- snapshot, visibility, public-model, local-embedding, policy, and software
  fingerprints.

Every manifest and diagnostic above is a private artifact.

Do not impose the heuristic rule that every exact bridge must be weaker than
one ordinary graph branch: parallel private paths can have greater equivalent
conductance. A branch-strength cap may be preregistered only for an explicitly
approximate family. Likewise, any claim that componentwise raw responses for
nonnegative source vectors lie between plain Dirichlet and exact Schur
responses requires the additional entrywise order
`0 <= fitted return <= B_H` and nonsingular M-matrix inverses. That order does
not by itself make normalized responses, resistances, or rankings conservative.

## Interaction between public and private experts

The public base remains a frozen reference. The private expert may produce a
private residual, private candidate score, or private routing signal during
local inference. A hard privacy-domain route would be the first policy route,
not a learned confidence decision: public-only queries use the public base;
queries explicitly authorized for local private processing may use the private
expert.

If a future system combines several outputs, do not assign manual confidence
or inverse-variance weights. The sources share the public representation and
boundary, so their errors are correlated. Follow
[`DESIGN_uncertainty_estimation_playbook.md`](DESIGN_uncertainty_estimation_playbook.md):

1. print source separability and the correlation matrix;
2. fit a calibrated `JointPosterior` and its probability/margin threshold on a
   dedicated private calibration split, never the untouched audit;
3. use top-one minus top-two margin as the selective-routing gate, not an
   absolute score level or a per-item multiplier; and
4. compare against public-only and factored-combiner controls exactly once on
   the audit split.

A new source earns admission only if the paired exposure-block bootstrap 95%
interval for with-source margin-AURC lies below the without-source point
estimate. This is a selective-routing criterion, not permission to expose the
route, margin, or response delta to a public caller.

The combiner, calibration rows, thresholds, and reports remain private. A
publicly releasable combiner would require a distinct privacy review and new
evidence.

This is a new, parked geometry. It is not authorized as a post-hoc feature in
the already frozen filing-ranker evaluation protocol
[`PROTOCOL_filing_ranker_eval.md`](PROTOCOL_filing_ranker_eval.md).

## Evaluation contract

Boundary alignment, Schur fidelity, and private filing utility answer different
questions and must be reported separately.

### Structural and numerical evaluation

- Use synthetic series, star, branching, cyclic, parallel-path, grounded, and
  multi-terminal graphs with exact identities.
- Compare exact elimination, plain Dirichlet grounding, and sparse closure on
  the identical retained domain, leakage, anchors, and probes.
- Report operator, raw Green-response, effective-resistance, rank, current,
  maximum-principle, and resource diagnostics.
- Include failures for asymmetric, negative, oversubscribed, double-counted,
  ungrounded, incomplete, and traversal-capped inputs.

### Representation and filing evaluation

- Build one exposure graph joining rows that share private-component
  membership, public boundary-anchor incidence, canonical aliases, lineage, or
  parent family. Take its connected components as indivisible blocks and assign
  each block wholly to train, combiner-calibration, or untouched audit. If this
  leaves too few independent blocks, fail closed or report descriptively; do
  not fall back to node-disjoint rows.
- Keep boundary anchors used for Procrustes/alignment out of the boundary audit.
- Evaluate against an independent local target such as held private placements
  or human review. Do not share a judge between training and evaluation.
- Report private filing/ranking performance, boundary continuity, and an exact
  check that frozen public outputs did not change.
- When routing is evaluated, report accuracy or filing utility, log loss, ECE
  with fixed bins, risk/coverage, and margin-gated AURC with paired
  exposure-block bootstrap intervals.
- Freeze one primary comparison and handle decision-bearing ablations with a
  gatekeeping hierarchy or explicit multiplicity correction.

A study on one already assembled private hierarchy is transductive when that
hierarchy or its candidate catalog exposes held destination folders. Label it
accordingly. An inductive deployment claim requires topology and a candidate
catalog frozen before a future bookmark cohort arrives. Before any labels or
results are inspected, a future protocol must freeze the exact train,
combiner-calibration, and untouched-audit manifest plus the primary metric and
minimum practical improvement threshold.

Required ablations are public-only; private expert without electrical closure;
exact private Schur response where feasible; sparse graph-first closure; and
optional semantic candidate prescreening. A semantic arm cannot replace the
graph-only primary. Report losses as well as wins.

## Staged acceptance

1. **Synthetic identities:** prove exact elimination and sparse-ledger safety on
   small graphs under
   [`PROTOCOL_synthetic_multiport_schur.md`](PROTOCOL_synthetic_multiport_schur.md).
   This stage cannot authorize private training.
2. **Private structural shadow study:** on a freshly authenticated local
   snapshot, measure whether the private interior induces material
   multi-terminal transfer after common-parent retention. No filing labels tune
   this stage.
3. **Fixed-budget approximation:** freeze support/resource budgets, fit on
   graph-only calibration components, and test exact-versus-sparse fidelity on
   untouched components. Proceed only if the approximation is numerically safe
   and materially better than plain grounding at the same budget.
4. **Private expert study:** train locally only after stages 1–3 and the
   harvester/privacy gates pass. Evaluate once under the split and uncertainty
   contract above. No publication or deployment follows automatically.

Each stage requires a prospective implementation protocol before real data are
inspected. Failure or an empty bridge signal is terminal for that arm unless a
new protocol is written before new evidence is generated.

Before stage 1 grows beyond synthetic fixtures, require the owner-authorization
manifest, separate public/private loaders, a private storage and deletion plan,
an external-network/telemetry audit, and tests that reject masked/unknown data,
propagate private taint to boundary artifacts, keep the public-base hash
unchanged, refuse worktree/symlink outputs, emit generic errors, and rebuild all
derivatives after privacy-state invalidation.

## Explicit non-goals and deferred work

- publishing or pushing any private-derived artifact;
- sending private data or representations to an external model or judge;
- jointly fine-tuning the public base on private data;
- treating boundary agreement as proof of privacy or downstream quality;
- building an all-pairs `1/R` boundary graph;
- allowing semantic similarity alone to create or weight bridges;
- splicing independently calibrated per-anchor systems into one kernel;
- tuning physical geometry against filing labels;
- hand-set confidence blending;
- claiming differential privacy, membership protection, or safe model export;
  and
- implementing Phi inference, sparse solvers, training, or deployment in this
  design PR.

Differential privacy, formal release testing, ungrounded gauges, learned
generality conductance, and public export are separate future research
programs. They must not be inferred from success of a local private expert.
Any future export protocol needs explicit human approval plus prospective
membership-inference, canary, and verbatim-recovery tests; passing non-DP tests
still cannot prove absence of leakage.
