# Prospective protocol: bounded grounded-diffusion fidelity

**Status:** prospective and frozen before any real-corpus fidelity result is
computed. Synthetic unit tests and implementation smoke tests are permitted in
the protocol PR. If the real graph, embeddings, budgets, selectors, protected
set, leakage rule, metric, or decision threshold must change, amend this file
and its machine-readable manifest before running the affected comparison.
Disposable synthetic test plans may be made during development. The actual
corpus plan must be generated only after the calibration-lock implementation,
this protocol, and the planner are all at their final committed versions; a
later change invalidates that plan rather than being absorbed by its old
repository SHA.

The first confirmatory phase is prospective snapshot-relative HOP convergence
only, and it is BLOCKED on a raw-source snapshot preparer. The current
`assembled_dag` is legacy parity input, not an authoritative scientific asset:
it loses relation type, privacy-propagation semantics, and visibility. The
14,246-node scrubbed largest-component figure is an audit estimate, not the
frozen study universe. Unknown visibility means `privacy_certified=false`. No real
solve may start until the preparer contract in Section 2 is satisfied. The
same-code fresh-process reproduction and immutable receipt that close this gate
are specified in
[`DESIGN_pearltrees_diffusion_consensus.md`](DESIGN_pearltrees_diffusion_consensus.md).
No node IDs, titles, paths, embeddings, or node-level manifests may be published; only
aggregate summaries and content hashes approved for the local provenance store
may leave the machine. Results must be labeled relative to this scrubbed
snapshot, never as coverage of the user’s complete Pearltrees corpus.

This is an outcome-blind numerical study. Placement labels, judge outputs,
filing ranks, and downstream MRR are prohibited inputs. A selector that wins
here is only a better approximation to a larger grounded-diffusion system at a
matched resource budget; it is not thereby a better filing ranker or a learned
covariance model.

## 1. Decision and estimand

The first confirmatory question is whether a deterministic HOP domain reaches
prespecified fidelity as `K` increases toward one larger topology-only
reference. Calibration chooses one `K` contrast; the untouched audit evaluates
that contrast once. There is no selector-promotion claim in this phase.
SKELETON, semantic RESISTANCE, and graph-derived closure remain prospective
secondary phases gated on the asset and provenance requirements below.

Topology-only and semantic conductance define DIFFERENT physical operators.
Each regime therefore has its own candidate union, outer reference, calibration
shell solves, and frozen scalar leakage. No candidate is compared across
regimes, and neither a reference nor an alpha calibrated for one regime may be
reused implicitly in the other.

For anchor `s`, candidate domain `D`, common protected nodes `P`, and the
matching regime-specific larger reference `R`, let

    g_D^s = J_D^-1 e_s,
    h_D^s(i) = g_D^s(i) / g_D^s(s).

The primary per-anchor error is the relative raw-response error

    E_g(s,D) = ||g_D^s[P] - g_R^s[P]||_2
               / max(||g_R^s[P]||_2, tiny).

The population estimand is the EQUAL-STRATUM MACRO mean: first average the
anchor-level contrast within each frozen degree stratum, then average
those stratum means with equal weight over decision-bearing strata. It is not
the pooled anchor mean, so a populous or easy stratum cannot dominate. Raw
response is primary because it preserves circuit units and the Dirichlet
monotonicity interpretation. Normalized screening, selected grounded effective
resistance, and rank fidelity are required noninferiority checks; none may be
used post hoc to replace a failed primary endpoint.

An honest outcome may be: a smaller adequate domain; better fidelity at the
same budget; no material difference, retaining the simpler hop default; no
adequate bounded domain; or an inadequate outer reference. All are complete
results.

## 2. Frozen graph and conductance regimes

Before ANY real solve, a content-hashed raw-source snapshot preparer must:

- freeze every raw input hash, parser version, and deterministic rerun hash;
- preserve source relation types and freeze an explicit physical-edge policy,
  including named include/exclude treatment for containment, aliases, shortcuts,
  and cross-links before reciprocal conductance adjacency is built;
- propagate privacy from known-private roots through the frozen containment
  relations, record how non-containment links are treated, and exclude affected
  nodes before anchor selection;
- preserve source visibility where known and set `privacy_certified=false` when
  any required visibility is unknown;
- emit local-only typed-node, exclusion, conflict, physical-edge, reciprocal
  adjacency, component, and scrub manifests plus aggregate publishable counts;
  and
- freeze the resulting study-universe and largest-component hashes, software
  commit, float64 backend, thread settings, and resource ceiling.

The preparer gate closes only through the consensus contract, not by possessing
a receipt file alone. The source specification must be the complete private
declaration bundle: a mode-0700 directory outside Git containing exactly the
fixed mode-0600 local-only marker and canonical mode-0600 specification, with
explicit RDF accounts and resolved absolute, non-symlink source paths of the
declared types. Before HOP planning, every leaf in that declaration, receipt,
and snapshot-attempt bundle is rechecked through a bound directory descriptor
with a nonblocking open as a mode-0600, unique-link regular file; a FIFO or
device substitution fails closed. Consensus binds the
declaration-validator implementation and runs exactly two fresh compiler
attempts, with no third retry and no pooling. Its verifier must be supplied the
receipt, source specification, relation policy, and both attempt directories;
it reruns fixed snapshot verification on both, reloads the actual manifests,
and re-derives all comparisons, common fields, warnings, and the decision.
Legacy artifact records/statuses are retained per attempt, but legacy-only
disagreement is warning-only. The receipt's unkeyed self-hash is an integrity
aid, not an authenticity credential.

The legacy assembled DAG may be compared only as a parity diagnostic; it cannot
source the scientific adjacency, privacy decision, or study universe. Detailed
preparer artifacts remain local-only. Any change to raw inputs, edge policy,
privacy propagation, or visibility invalidates downstream anchor and domain
manifests.

Only after that preparer gate passes, the first phase uses only the
**topology-only HOP operator**, with unit base conductance on every canonical
physical edge admitted by the frozen raw-source relation policy. It makes a snapshot-relative
convergence claim, not a claim that aliases or shortcuts are physically correct.
Changing their inclusion requires a prospective amendment and rebuilt hashed
adjacency; the study runner must not guess directions or relation types.

SKELETON is unavailable for decision-bearing use. The clean Collection parent
source covers only 1,324 of 14,520 nodes (9.1%), path records cover 1,283, and
their union covers 2,542 (17.5%) with one directed cycle. SCC condensation can
represent that cycle but cannot manufacture missing parents or roots. Before a
future SKELETON phase, freeze a typed parent/root/missing/conflict manifest with
positive root evidence and a prospective coverage threshold; failure of that
threshold leaves SKELETON descriptive or unavailable.

Semantic conductance and RESISTANCE are also later secondary phases. When
licensed by a revision-pinned embedding/cache manifest, set `epsilon=0.05`, set
`ell` to the median finite positive within-edge embedding distance, and freeze
`c_ref=1.0` in `c_ref/c_ij`. Nomic requires its exact revision and `clustering: `
text contract; MiniLM is descriptive, and e5 is never a silent substitute. A
semantic phase must construct its own candidate union, reference, shell
calibration, and scalar alpha rather than reusing topology artifacts.

## 3. Anchor batches and protected nodes

Eligible anchors are ALL non-isolated folder nodes in the largest frozen
positive-conductance component. They are explicitly not restricted to the
sparse 1,504-row lineage/path export or to nodes for which one convenience path
was materialized. Exclusions are determined from the frozen graph manifest
before any metric is computed.

The frozen eligibility ledger may also contain excluded-private rows whose
nodes are intentionally absent from retained adjacency. Such a row is valid
only with `eligible=false` and reason `direct_private` or
`private_descendant`; it is evidence of exclusion, not a malformed missing
adjacency row and never enters a degree quartile. Retained eligibility rows must
cover every adjacency node and satisfy the retained-row degree/component
contract.

HOP consumes only the frozen reciprocal conductance adjacency. A memoized
revisit during bounded traversal means path convergence or a cycle; it is not a
cut point and must not create a ground shunt. Cuts are actual retained-to-omitted
edges. No parent direction, root, or depth is inferred in phase one.

Construct exactly four deterministic rank-based degree quartiles. Sort all
eligible anchors by `(undirected_conductance_degree, stable_typed_node_id)` and
split that ordered population into four contiguous groups whose sizes differ by
at most one. If `N=4q+r`, the first `r` quartiles receive `q+1` members. The
typed-ID key is `(namespace, positive decimal integer)`, so `pt:2` precedes
`pt:10`. Within each quartile order nodes by
`SHA-256(3882001, "select", typed_node_id)` and select exactly 32 anchors. Each
key is SHA-256 of newline-terminated canonical UTF-8 JSON
`[3882001,purpose,typed_node_id]`, with the typed-ID key as a collision tie. If
any quartile cannot supply 32, phase one is coverage-inadequate before solves.

Within each quartile reorder the 32 selected anchors by the independent key
`SHA-256(3882001, "split", typed_node_id)`: the first 8 are calibration and the
remaining 24 are untouched audit. Reorder each split within quartile by
`SHA-256(3882001, "batch", typed_node_id)`. Calibration batch `j` and audit
batch `j` contain exactly the `j`th anchor from each of the four quartiles, for
8 calibration and 24 audit batches. Persist the quartile boundaries, ordered
memberships, split, and batch IDs. Every batch is therefore exactly balanced;
calibration metrics cannot enter the final contrast.

For each anchor independently, freeze its source-relative ordinary-hop order
with the canonical tie key. Its protected set is the anchor plus its first 16
reachable non-anchor nodes. The batch protected set is the union across its four
anchors. This set is fixed before any selector is scored. A phase-one HOP budget that
does not retain the complete protected set fails that batch; do not form a
post-hoc intersection and do not rank missing nodes last.

Report source-to-source hop distances within each batch. Primary summaries use
the equal-stratum macro estimand from Section 1. If reference failure or method
coverage leaves fewer than 18 of the 24 complete
balanced audit batches, demote the whole contrast to descriptive. A failed batch
is never decomposed into surviving anchors; never redistribute quartile weight
or fall back to an unbalanced pooled mean. Degree and distance-stratified
summaries beyond the frozen macro estimand are descriptive.

## 4. Frozen HOP budgets and gated future families

Freeze one maximum-budget HOP ordering and take deterministic prefixes, so

    S_256 subset S_512 subset S_1024.

Every domain is one union-of-anchors system with one precision matrix and one
factorization. Its budget includes anchors and every retained node. HOP uses the
deterministic multi-source ordering with the existing stable final-shell tie
rule. Use nominal retained budgets `K={256,512,1024}`. A required protected set
that exceeds `K` is a recorded coverage failure. Complete final-shell runs may
be reported descriptively, but the decision-bearing comparison uses strict
prefix budgets.

### Gated future families (not phase-one decisions)

SKELETON requires the prospective parent/root/missing/conflict coverage gate in
Section 2. If licensed later, freeze `ancestor_depth=3`, SCC-condense the full
typed `child -> principal_parent` relation, and treat traversal collisions as
cycle convergence rather than cuts. RESISTANCE requires the semantic cache gate
and uses only existing graph edges with frozen length `c_ref/c_ij`,
`c_ref=1.0`; embeddings never create shortcut edges. Neither family enters the
phase-one candidate union, calibration, reference, or audit contrast.

Exact one-port Kron reductions and multi-port sparse closure remain future
families until their separate gates are satisfied. Multi-port elimination may
create fill and must never be mislabeled a scalar tree shunt.

## 5. Regime-specific larger-domain references and leakage

For phase one, `U_top` is exactly the frozen HOP `S_1024` domain for that
batch. Freeze one continued anchor-source multi-source HOP order through at
most 4,096 nodes; `S_256`, `S_512`, `S_1024`, and `R_top` are its prefixes.
This preserves all of `U_top` and makes reference expansion unambiguous. Build
the exact-Dirichlet models only after the no-solve plan is accepted. If `U_top`
exceeds the resource contract or `R_top` cannot
be built, that batch is reference-inadequate and cannot support a convergence
claim. If the connected component is exhausted, the whole component is an
exact structural reference. This does not override the zero-alpha numerical
gate below: an otherwise ungrounded whole-component Laplacian has a constant
gauge mode and therefore blocks this phase rather than licensing a hidden
ground or conditioning-derived alpha. A gauge-aware whole-component extension
requires a prospective amendment.

A later licensed semantic phase must independently construct `U_sem` from only
the families compared under that semantic operator, then build `R_sem`; it may
not reuse `U_top`, `R_top`, or their calibration merely because node hashes
happen to match.

Each `R` is a bounded regime-specific reference, not whole-graph truth. It is
adequate only if its `U` versus `R`, on the frozen protected set, has:

- conservative 90th-percentile `E_g <= 0.01`;
- conservative 90th-percentile maximum absolute `h` error `<=0.005`; and
- conservative 10th-percentile top-8 overlap `>=0.98`.

Compute every upper-tail percentile as the observed higher order statistic and
every lower-tail percentile as the observed lower order statistic; do not use
linear interpolation that moves a tail estimate toward the center. Failure is
reported as right-censored reference convergence and forces lock mode
`blocked`, so no audit solve is authorized. Do not choose a different outer
size after inspecting which K looks best.

On the hash-balanced calibration batches only, calibrate anchor-specific uniform
leakage on `R_top` for the topology operator. A later semantic phase repeats
this independently on `R_sem` rather than importing `alpha_top`, using the
graph-hop radius-3 shell and target `exp(-1)` from
`LOCAL_GROUNDED_DIFFUSION.md`. Every shell node must be reachable and strictly
interior (`beta=0`) in its own reference.

The calibration call is frozen with base intrinsic uniform `alpha=0`, bath
temperature 0, radius-3 bracket seed, relative bisection tolerance `1e-8`, and
at most 80 attenuation evaluations per anchor. It has no maximum-alpha cap and
may return only a finite nonnegative result. Evaluate `alpha=0` first. The
zero-alpha precision must itself pass grounding, SPD, reciprocal-condition,
and residual checks, and the reported numerical minimum added leakage must be
exactly 0. A positive conditioning-derived numerical minimum, a nonfinite
result, inability to bracket, or evaluation-budget exhaustion blocks the lock;
none licenses a hidden floor, cap, jitter, or substituted alpha.

Use all 32 calibration anchors. Persist each anchor's required added leakage,
take the maximum of the four requirements in each of the eight balanced
calibration batches, then freeze the maximum of those eight batch maxima as
phase-one `alpha_top`. This is algebraically the maximum over all 32 anchors,
but both levels are retained as provenance. Apply `alpha_top` unchanged to
every calibration and audit reference, union, and candidate. A later
`alpha_sem` is a distinct scalar parameter even if its numeric value happens
to match. Node-varying alpha and per-domain recalibration are descriptive
alternatives only. Recalibrating each candidate would change the physical
model and is prohibited in the primary comparison.

Numerical routines have separate frozen roles: `numpy.linalg.eigh` performs
the shared alpha-calibration decomposition, `numpy.linalg.eigvalsh` estimates
the decision-model spectrum/condition, `numpy.linalg.cholesky` creates the
decision factor, and `numpy.linalg.solve` performs the lower/upper triangular
factor solves. All are CPU float64 operations. Exactly one BLAS thread is
requested, and the lock must record both a nonempty path-free actual BLAS
identity and an observed thread count of exactly one. A configured thread
count without observation is insufficient.

## 6. Fidelity and boundary metrics

Compute per anchor on the complete frozen protected set:

- relative L2 raw-response error `E_g`;
- maximum absolute error in `h_s`;
- relative error of selected grounded effective-resistance distances from the
  anchor to protected nodes, using factor coordinates or selected solves;
- deterministic pairwise rank-inversion fraction and top-8 overlap for `h_s`;
- source diagonal Green-response error; and
- solve residual and sign/maximum-principle violations.

Effective resistance may be omitted only when the run manifest declared the
selected-solve resource arm unavailable before any family result. Record an
explicit omitted flag. That preregistered omission removes the
effective-resistance absolute gate and noninferiority endpoint from the active
endpoint sets; it is neither imputed as a pass nor treated as a post-result
missing-value failure. The reduced run cannot resolve the resistance part of
`OPENQ-004`, and the endpoint cannot be restored after any family result is
seen.

For every batch/family/budget also report:

- retained/boundary node counts, cut-edge count and cut mass;
- per-anchor cut-current fraction;
- the batch/domain protected-set maximum of `p=J_D^-1 beta`;
- regime-specific frozen scalar `alpha`, realized e-fold radius and censor flag
  for every anchor;
- reciprocal condition number, M-matrix sign check, Cholesky/solve residuals;
- adjacency calls or touched weighted degree where observable;
- selection, assembly, factorization, metric, and total wall times; and
- projected dense bytes, peak RSS, backend identity, and thread settings.

An individual phase-one HOP budget is **adequate** only with 100% protected coverage,
using higher-order-statistic Q90 and lower-order-statistic Q10 summaries,
all numerical checks passing, and across anchors:

- 90th-percentile `E_g <= 0.05`;
- 90th-percentile maximum absolute `h` error `<=0.025`;
- 90th-percentile rank-inversion fraction `<=0.05`;
- 10th-percentile top-8 overlap `>=0.90`.

Across decision-bearing batches, the conservative higher-order-statistic Q90
of the batch/domain protected boundary-harmonic maximum must be `<=0.10`.
When resistance is evaluated, its 90th-percentile relative error must also be
`<=0.05`. These are deliberately conservative engineering tolerances on a
unit-normalized screening field, not claims about downstream filing loss.

## 7. Gated future graph-derived boundary closure

This section is implemented and synthetic-tested but is NOT part of the first
confirmatory corpus phase. After HOP convergence, a separate prospective
amendment may license this topology-only closure rule. Plain exact Dirichlet
remains the baseline: every omitted edge contributes to
the boundary shunt `beta`. For an omitted exterior `E`, the exact multi-port
target is its Dirichlet-to-Neumann/Schur response, not a collection of semantic
similarities. If `B` denotes the nonnegative Schur term subtracted from the
Dirichlet principal block, write its retained sparse approximation as transfer
conductances `kappa_ij` and per-port self-return `sigma_i`. Require

    beta_i = beta_res_i + sigma_i + sum_j kappa_ij,
    beta_res_i >= 0,
    sigma_i >= 0,
    kappa_ij = kappa_ji >= 0.

The closed model is

    J_closed = L_ind + L(kappa) + diag(alpha + beta_res)
             = J_D - B_hat,

where `B_hat` has off-diagonal transfer `kappa_ij` and diagonal self-return
`sigma_i`. Thus bridge degree and self-return are both removed from the full
Dirichlet shunt; bridges may never be added on top of `beta`. Verify the ledger
to float64 tolerance and recheck component grounding, M-matrix signs, SPD, and
the maximum principle.

For a genuinely two-terminal ungrounded exterior, `kappa=1/R_eff` is the exact
replacement conductance. More generally, compute the full two-port response:
for boundary branches `c_i`, `c_j` meeting at one exterior node with leakage
`a` and `d=c_i+c_j+a`,

    kappa_ij = c_i c_j / d,
    sigma_i = c_i^2 / d,
    sigma_j = c_j^2 / d,
    beta_res_i = c_i a / d,
    beta_res_j = c_j a / d.

Pairwise effective resistances do not uniquely determine a multi-terminal or
grounded exterior. Independently adding many `1/R_eff` bridges can double-count
shared paths. A component touching more than two retained ports therefore
requires an audited sparse approximation to its joint Schur map, or remains
grounded under the plain Dirichlet baseline.

Only cut branches represented inside the frozen reference exterior `E=R\D`
may fund a graph-derived bridge. Candidate cut edges from `D` directly beyond
`R` remain in `beta_res`; they cannot be credited to an unobserved path.
Discovery traverses only the induced `E`. Edges from `E` beyond `R` become
zero-bath shunts in the exterior precision before `E` is eliminated. Thus an
exact component is exact relative to the frozen bounded reference `R`, not a
claim about the unbounded full graph.

Exterior discovery uses one deterministic component traversal with a memoized
visited set. Memorization terminates loops and deduplicates work only. A search
frontier colliding with an already visited exterior node means path convergence
or a cycle in the same exterior component; it is not a cut point and must not
create grounding. Dirichlet cuts are defined only by actual edges crossing from
the retained set to the omitted set. Articulation status, if ever needed for a
selector, requires a separate explicit topology calculation. Collect every
retained boundary port incident to each exterior component once; a two-port
component may be reduced exactly, while a multi-port component is handled
jointly or left grounded rather than decomposed at traversal collisions.

Repository precedents constrain the implementation. Reuse the deterministic
BFS and actual cut-edge accounting in
[`local_diffusion.py`](../../src/unifyweaver/graph/local_diffusion.py). The
cycle-correct minimum-distance and caret mixing-boundary routines in the
[Rust WAM boundary cache](../../templates/targets/rust_wam/boundary_cache.rs.mustache)
are examples for bounded search and common-parent stopping, not physical Schur
oracles. [Tarjan SCC
condensation](../../src/unifyweaver/core/advanced/scc_detection.pl) is
the repository precedent for preprocessing the typed parent relation before
principal depth or SKELETON ancestry is computed. [Node-only
memoization](../../scripts/parent_histogram_recurrence.py) is not licensed for
simple-path counts or other path-state-dependent transfer quantities; this PR
uses global component memoization only for connectivity. Semantic
[k-nearest-neighbour code](../../scripts/mindmap/mst_folder_grouping.py) may
propose searches, but an embedding MST or kNN cut is never a cut in the
original graph.

A future closure amendment must freeze this primary rule before its real run:

- deterministically discover every exterior component inside the matching
  topology reference `R_top \ D` and collect all retained ports from actual cut
  edges;
- include EVERY fully traversed component with exactly two retained ports,
  aggregating parallel components between the same pair in one audited ledger;
- compute its complete two-port `kappa` and self-return from topology-only `c0`
  Schur solves, without embeddings, semantic prescreening, a pair budget, `rho`
  truncation, or an approximate bridge cap;
- leave every one-port, multi-port, traversal-incomplete, or numerically failed
  component fully grounded under its exact Dirichlet beta; and
- reject shared-path pairwise composition unless one future method represents it
  by a single audited joint sparse-DtN calculation.

The implementation distinguishes failure scopes conservatively. If the frozen
component-size cap interrupts exterior discovery, the component partition is
not known to be complete, so the entire closure arm becomes a recorded no-op
and the unchanged full-Dirichlet baseline remains the result. After discovery
has completed, a numerical failure in one two-port Schur solve grounds only
that complete component; other successfully reduced complete two-port
components may remain in the ledger. Record stable reason counts and a
content-based failure fingerprint in either case. Malformed, incomplete, or
nonreciprocal graph input is an integrity failure and still aborts rather than
being mislabeled a conservative numerical fallback.

A simple exterior series path is naturally weaker than each constituent branch,
but parallel exterior paths can be stronger and must not be clipped if the
reduction is called exact. Approximate multi-port closure, semantic proposal
filters, and `q/rho/cap` sweeps are separate exploratory future families. They
are not primary fallbacks and cannot rescue a failed or empty exact two-port
closure result.

In a future licensed selector phase, SKELETON may avoid boundary error by retaining
common parents and other high-value junctions before spending budget on local
fill. Closure is considered only for residual boundary ports joined by an
actual path through one represented omitted component. It must never force a
bridge merely because two ports are semantically close. Consequently, an
empty closure is a valid and scientifically useful result. In particular, the
primary hypothesis is that preserving common parents may leave no eligible
cross-boundary bridge, or may leave so little Schur transfer mass that closure
cannot materially improve the selected-domain solution.

Report, before any response comparison, the number of exterior components with
two or more retained ports, the number of graph-connected port pairs proposed
and realized, total transfer degree relative to original cut mass, and
`||B_hat||_F / ||J_D||_F`. If the realized closure is empty, record a
no-op and do not manufacture an
experimental arm. If it is nonempty but small, still report the ledger and
paired response comparison; retain plain Dirichlet unless a future frozen
closure amendment meets the
Section 8 efficacy and noninferiority form on untouched data.

Before the corpus run, exact small-graph tests must include equal and unequal
exterior series paths, exterior leakage, and a multi-terminal shared-path case,
plus oversubscribed, asymmetric, self-edge, already-adjacent, ungrounded, and
double-counted-beta failures. When a future corpus closure phase is licensed, compare closure with
plain Dirichlet on the identical selected domain, protected set, `alpha`, and
reference.

## 8. Fixed HOP convergence contrast and paired uncertainty

Calibration may inspect all three HOP budgets, but the lock has exactly four
modes and no audit result may change one:

- `finite_contrast`: the reference is adequate, `K_low` is the smallest
  calibration-adequate value in `{256,512}`, and the next larger finite HOP
  endpoint has a distinct node-content hash. This mode licenses the frozen
  paired efficacy/noninferiority contrast.
- `absolute_only`: the reference is adequate and `K_low=1024`, with `R_top` as
  the endpoint. Report absolute/reference adequacy only. Under the phase-one
  zero-alpha gate, component exhaustion produces an ungrounded gauge mode and
  therefore blocks; a gauge-aware extension needs a prospective amendment.
- `right_censored_diagnostics`: the reference is adequate but no HOP candidate
  is adequate. The untouched audit may report the frozen diagnostics, but it
  cannot make convergence, efficacy, noninferiority, or resource claims.
- `blocked`: the reference is inadequate or calibration/numerical requirements
  fail. No audit solve is authorized. In particular, reference inadequacy is
  not converted into a right-censored audit run.

The lock records its mode, reason, endpoint node-content hashes, selected
budgets where applicable, and the resulting audit/decision authorization
booleans. Nominally different endpoints whose hashes are equal because the
component was exhausted never enter a log-ratio or resource contrast and block
phase one.

Use the frozen `bootstrap_multiplicities.jsonl` artifact containing exactly
9,999 deterministic paired resamples of the 24 audit four-anchor batches with
seed `3882002`. For replicate `r`, draw `d`, and nonce `n`, hash
newline-terminated canonical UTF-8 JSON
`[3882002,"paired_audit_batch_bootstrap",r,d,n]`; interpret SHA-256 as an
unsigned 256-bit integer, reject values at least
`2^256 - (2^256 mod 24)`, and take the first accepted value modulo 24. Store
the resulting 24-entry count vector, whose entries sum to 24. Later runners
must consume these vectors and must not regenerate them with a library PRNG.
Apply each vector's identical
batch multiplicities to both budgets and every paired endpoint. Every
resampled batch contains one anchor from each quartile, so
recompute the four quartile means and average them equally; no replicate can omit a quartile
and no redraw rule is needed. The one-sided 95% upper endpoint is the higher
observed bootstrap order statistic at 0.95, and the lower endpoint is the lower
observed order statistic
at 0.05. These intervals describe the frozen structural sampling design, not
solver randomness.

For a finite `K_high`, define the primary paired log-error contrast

    Delta_hi_lo = macro_s mean[log(E_g(high)) - log(E_g(low))].

Use extended-real zero handling without an epsilon: equal zeros contribute
zero; zero numerator with positive denominator contributes negative infinity;
positive numerator with zero denominator contributes positive infinity. A
material larger-domain efficacy finding requires the one-sided upper endpoint
of `Delta_hi_lo` to be strictly below `log(0.9)`, not merely below zero. The
smaller domain is declared converged only when it independently passes the
absolute Section 6 adequacy gates and the full noninferiority intersection below,
while `K_high` does not meet that efficacy rule. If larger-domain efficacy and
smaller-domain noninferiority conflict or neither resolves, report an
inconclusive frontier rather than choosing post hoc. When `K_high=R_top`, report
only the absolute adequacy of `S_1024` and reference convergence; do not call the
reference trivially lower error an efficacy result.

Noninferiority is a one-sided INTERSECTION-UNION TEST: EVERY named endpoint
must pass its own 95% one-sided upper bound with these frozen margins:

- primary `log(E_g(low))-log(E_g(high)) < log(1.10)`, with the same
  extended-real zero convention;
- maximum absolute `h` error and rank-inversion fraction: paired
  low-minus-high harm `<0.01` absolute;
- batch-level protected boundary-harmonic maximum: paired low-minus-high harm
  `<0.01` absolute;
- top-8 overlap: paired high-minus-low loss `<0.01` absolute; and
- source-diagonal relative error: paired low-minus-high harm `<0.01`
  absolute; and
- effective-resistance relative error: the same `<0.01` harm margin only when
  the effective-resistance arm was predeclared enabled. A predeclared omitted
  arm removes this endpoint as specified in Section 6.

Use the corresponding bootstrap upper endpoint for each strict inequality.
M-matrix signs, grounding, reciprocal condition, Cholesky/solve residuals, and
maximum-principle checks are deterministic safety gates, not noisy
noninferiority endpoints; any failure fails immediately. Resource use is
reported as a frontier rather than hidden inside statistical fidelity: the
smaller K must actually use fewer retained nodes, and projected bytes, peak RSS,
and wall time are reported for both endpoints. No endpoint may rescue another.

SKELETON, RESISTANCE, semantic conductance, closure, alternate strata, alternate
alpha, and complete-shell domains are not phase-one confirmatory contrasts. A
future licensed method comparison uses the same paired equal-stratum machinery,
the `log(0.9)` efficacy threshold, and the noninferiority IUT, but requires its
own prospective amendment and untouched audit data.

## 9. Reproducibility and fail-closed rules

The scientific `plan_fingerprint` covers the plan's `fingerprint_core` rather
than machine-local paths. A separate full-manifest integrity seal covers the
complete no-solve manifest, including accepted/blocked and authorization
fields. These unkeyed records detect accidental or unsynchronized drift; they
do not authenticate a wholly replaced self-consistent chain against a
malicious same-user without an external signature or immutable trusted store.
Before the
no-solve plan is frozen, the full consensus verifier above must pass and the
plan must bind both the receipt content record and canonical attempt-A manifest
record; receipt-only verification is prohibited. Taken together, the
deterministic scientific fingerprints and complete manifest provenance bind:

- raw-source and parser hashes, deterministic preparer hash, physical-edge
  policy, privacy-propagation and visibility-limitation manifests, frozen
  reciprocal adjacency, and all approved byte hashes;
- anchor, balanced-batch, protected-node, HOP-ordering, domain, boundary, and
  reference manifests;
- parent or embedding manifests only in a future phase that has passed their
  prospective coverage gate;
- `K`, shell, `alpha_top`, stable tie keys, and any future licensed selector or
  closure ledger;
- metric definitions, candidate and reference adequacy tolerances, the minimum
  18-of-24 complete-audit-batch rule, prospective `K_low`/`K_high` selection
  cases, noninferiority rules, the exact bootstrap multiplicity artifact, and
  software SHA;
- float dtype, numerical thresholds, backend/thread identity without absolute
  library paths; and
- cache keys and deterministic-rerun hashes; plus per-phase timings and
  measured peak RSS in the complete manifest seal only.

The phase-one resource contract uses a post-hoc per-batch elapsed ceiling for
both calibration and fidelity. It does not claim to interrupt a hung in-flight
LAPACK call; hard deadlines require a future process-isolated adapter.

Deterministic plan content and observational execution provenance are separate:
elapsed times and measured peak RSS do not enter the no-solve plan fingerprint.
The plan freezes their later measurement contract. The implementation and
transaction details are specified in
[`DESIGN_pearltrees_hop_plan.md`](DESIGN_pearltrees_hop_plan.md).

The planner preflights bounded receipt and attempt-manifest reads, then enforces
its own aggregate ceiling over the receipt, both manifests, adjacency,
eligibility, source specification, and relation policy that it captures. This
does not claim to cap memory inside the separately invoked upstream snapshot
verifier; that verifier retains its own resource contract. The plan output may
not overlap any receipt/attempt/policy input, the declaration bundle, any
declared raw source, or the optional legacy parity source. Staging writes,
verification, no-replace rename, directory synchronization, and conservative
rollback are descriptor-relative; an unprovable replacement is leaked for
manual inspection rather than recursively deleted.

The no-solve manifest must state `structural_metrics_computed=true`,
`diffusion_or_fidelity_metrics_computed=false`, and
`audit_solve_authorized=false`. The calibration lock must bind the complete
plan-manifest content record—not merely `plan_fingerprint`—before it records
the nonempty actual path-free BLAS identity, confirms one observed BLAS thread,
freezes `alpha_top` and one of the four lock modes, and authorizes any audit
solve allowed by that mode.

Lock verification is full-chain and content verification: rerun declaration,
two-attempt consensus, and plan verification; check the complete plan record,
calibration artifacts, hashes, lock fields, and authorization consistency. It
does not numerically recompute eigendecompositions, bisection, factors, or
responses. Such recomputation is a separate explicit rerun. Verification of
unkeyed content hashes also does not authenticate a wholly replaced
self-consistent chain against a hostile same-user; that requires an external
signature or immutable trusted store.

Fail closed on a missing or mismatched raw-source preparer manifest, unfrozen
physical-edge policy, incomplete privacy propagation, incomplete or asymmetric
adjacency, duplicate/unknown IDs,
non-nested HOP prefixes, missing protected nodes, stale caches, changed edge
weights between candidate and
reference, nonfinite values, closure-ledger error, an ungrounded component,
M-matrix sign failure, insufficient reciprocal condition, solve residual above
the recorded tolerance, or reference/calibration inadequacy. Numerical jitter,
zero cut conductance, insulating truncation, and hidden embedding fallback are
not repairs.

The phase-one report must separate synthetic correctness, snapshot-relative
topology-only HOP convergence, reference failures, coverage failures, and
resource failures. Any later semantic, SKELETON, or closure report is a
separate prospectively licensed artifact. No geometry result may be described
as a filing improvement,
covariance estimate, CUDA crossover, or Stage-A dependence promotion.
