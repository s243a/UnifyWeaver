# Pearltrees HOP fidelity no-solve plan

**Status:** prospective implementation of the planning link between the
accepted two-process snapshot-consensus receipt and the calibration lock in
[`PROTOCOL_bounded_diffusion_fidelity.md`](PROTOCOL_bounded_diffusion_fidelity.md).
This artifact performs topology bookkeeping only. It computes no diffusion
response, leakage estimate, fidelity metric, filing metric, or judge result.
Synthetic tests may generate disposable plans while implementation is changing,
but the actual corpus plan must be generated only from the final committed
calibration-lock implementation. Any later implementation or contract change
invalidates that plan and requires regeneration under a new version.

## 1. Why planning is a separate transaction

The snapshot receipt proves repeatable construction of one privacy-certified
graph. It does not choose anchors or authorize an audit. The HOP plan freezes
every outcome-blind choice that could otherwise move after calibration:

- the degree-stratified anchors and calibration/audit split;
- the four-anchor balanced batches and protected nodes;
- one nested HOP ordering per batch, its strict candidate prefixes, and its
  larger reference;
- every real retained-to-omitted cut edge under exact Dirichlet grounding;
- the radius-3 calibration shells;
- the complete 9,999-replicate paired-bootstrap multiplicity schedule; and
- the numerical, memory, and statistical contracts that later stages must use,
  including reference adequacy, the 18-of-24 complete-audit-batch floor, and
  the prospective `K_low`/`K_high` selection cases.

An accepted plan authorizes only the registered calibration batches. It never
authorizes audit solves. The later calibration-lock manifest must bind the
complete plan-manifest content record, not only `plan_fingerprint`; freeze
`alpha_top`, the selected lock mode and any applicable contrast, and its actual
path-free BLAS identity; and only then authorize the untouched audit when that
mode permits it.

The lock has exactly four modes. `finite_contrast` licenses the frozen paired
finite-budget comparison. `absolute_only` licenses only absolute/reference
adequacy for the `K_low=1024` case. Phase-one component exhaustion leaves an
ungrounded zero-alpha gauge mode and therefore blocks rather than manufacturing
an executable comparison; a gauge-aware extension requires an amendment.
`right_censored_diagnostics` licenses an
untouched audit diagnostic when the reference is adequate but no candidate is
adequate; it licenses no convergence, efficacy, or resource claim. `blocked`
licenses no audit solve. Reference inadequacy, nonfinite calibration, or a
numerical-contract failure necessarily selects `blocked`.

## 2. Full-chain verification and read boundary

The planner requires the receipt, both snapshot attempts, source declaration,
and relation policy. It reruns the full consensus verifier and accepts only
`exact_consensus_ready`. Receipt-only verification is prohibited. The plan
binds the receipt bytes and canonical attempt-A manifest bytes, plus the exact
records for adjacency, eligibility, components, physical edges, source
declaration, and relation policy. Attempt B remains repeatability evidence and
is never pooled.

Before invoking the upstream consensus verifier, the planner preflights fixed
upper bounds for the receipt and both attempt manifests. It then rechecks every
leaf in the three already-verified private bundles through bound directory
descriptors: every leaf must remain a unique-link, mode-0600 regular file. This
planner preflight does **not** bound the memory used inside the separate
upstream snapshot-verifier subprocesses; those retain their own snapshot
resource contract. Leaf descriptors are opened nonblocking and then checked as
regular files, so a substituted FIFO or device fails rather than hanging or
being consumed.

After verification, attempt A is not reopened casually by pathname. The
planner holds an `O_DIRECTORY|O_NOFOLLOW` descriptor, reads only
`manifest.json`, `adjacency.jsonl`, and `anchor_eligibility.jsonl` through that
descriptor, checks leaf identity/mode/link count, enforces the manifest-record
size before reading, and operates on the captured bytes. The planner-input
ceiling covers the receipt, both manifests, adjacency, eligibility, source
specification, and relation policy that this planning transaction reads; it is
not advertised as a bound on the upstream verifier. The planner reruns
consensus and recaptures those bytes before atomic installation. The complete
plan verifier repeats the full chain and derives every output byte again.

The planning derivation never uses node titles, accounts, URLs, source IDs,
embeddings, lineage exports, labels, judge outputs, filing results,
calibration results, or caches. Account, source, and path declarations are
inspected only to verify provenance and reject output/input overlap; they never
enter anchor selection or graph construction. Reads performed by the required
upstream verifier remain within that verifier's separately declared scope. The
legacy DAG is not a graph input.

## 3. Exact deterministic choices

These details resolve ambiguities that prose alone left open:

1. A typed node ID is ordered as `(namespace, positive decimal integer)`, so
   `pt:2` precedes `pt:10`.
2. For purpose `p` in `{select, split, batch}`, the random-looking key is
   SHA-256 of newline-terminated canonical UTF-8 JSON
   `[3882001,p,node_id]`. A hash collision is broken by the typed-ID order.
3. If `N = 4q+r`, the first `r` degree quartiles receive `q+1` members and the
   others receive `q`.
4. Each batch has one continued multi-source ordinary-HOP order. Its first
   256, 512, and 1,024 nodes are `S_256`, `S_512`, and `S_1024`; its first at
   most 4,096 nodes are `R_top`. This guarantees a single nested chain rather
   than restarting expansion from `S_1024`.
5. A memoized revisit is path convergence or a cycle. Only a real edge with
   one retained and one omitted endpoint enters the cut ledger.
6. Bootstrap replicate `r`, draw `d`, and nonce `n` use SHA-256 of
   newline-terminated canonical JSON
   `[3882002,"paired_audit_batch_bootstrap",r,d,n]`. Interpret the digest as
   an unsigned 256-bit integer and reject it when it is at least
   `2^256 - (2^256 mod 24)`; map the first accepted value modulo 24. The
   resulting 24-count multiplicity
   vector is persisted for each of all 9,999 replicates; later runners consume
   this artifact rather than calling a library PRNG.

Later runners must consume the frozen node lists. They may not recompute a
domain with a library comparator whose string ordering differs.

## 4. Frozen design

Eligible anchors are the snapshot verifier's public, non-isolated nodes in the
largest component. The eligibility ledger legitimately also contains excluded
private nodes that are absent from retained adjacency; those rows must be
exactly `eligible=false` with reason `direct_private` or
`private_descendant`, and they never enter the anchor population. Retained rows
must cover adjacency and satisfy their complete retained-row schema. Eligible
anchors are sorted by `(degree, typed-ID key)` and divided into four rank
quartiles. Each quartile contributes 32 selected anchors: eight to calibration
and 24 to untouched audit. Independent selection, split, and batch hash domains
prevent one ordering from doing multiple jobs. Batch `j` contains the `j`th
anchor from each quartile.

For every anchor, a complete single-source HOP traversal freezes a traversal
hash and reachable count. Its protected prefix is the anchor plus the first 16
reachable non-anchor nodes. A batch protected set is their union. Missing
protected coverage blocks the plan; there is no post-hoc intersection.

Every domain is topology-only, uses unit conductance on policy-admitted graph
edges, and retains exact Dirichlet grounding with no closure. The boundary
artifact records every cut edge and integer `beta`. `R_top` is a bounded
reference unless its boundary is empty, in which case it is the exact connected
component.

For each calibration anchor, the plan freezes all nodes at graph-hop radius 3.
The shell must be nonempty, contained in that batch's `R_top`, and strictly
interior (`beta=0`). The plan records the later target `exp(-1)` but leaves
`alpha_status=unfrozen`. Calibration starts from base intrinsic uniform
leakage `alpha=0` and bath temperature 0. It must evaluate `alpha=0`, prove it
numerically admissible, and record a numerical minimum of exactly 0; a positive
conditioning-derived minimum is a block, not permission to inject leakage.
The bracket seed uses radius 3, bisection relative tolerance is `1e-8`, and at
most 80 attenuation evaluations are allowed per anchor. The result must be
finite. There is no hidden alpha cap, floor, or jitter; inability to bracket a
finite result within the evaluation budget blocks the lock.

All 32 calibration anchors participate. Record each anchor requirement, take
the maximum within each of the eight four-anchor calibration batches, then
take the maximum of those eight batch maxima as `alpha_top`. This equals the
maximum across all 32 anchors while preserving batch provenance. No failed or
inconvenient anchor may be dropped.

## 5. Resource and numeric contracts

Three resource quantities are deliberately distinct:

- the inherited snapshot compiler's raw/artifact byte ceiling;
- a planner input-byte ceiling and deterministic edge-touch ceiling; and
- the future study peak-RSS ceiling.

The dense pre-workspace projection is `4 * 8 * n^2` bytes: four square
float64 arrays, excluding factorization workspace. If the largest realized
reference projection exceeds the registered study ceiling, the plan is an
immutable scientific block. The later runner must measure peak RSS; the
projection is not a claim that the workspace fits.

The first phase freezes separate numerical roles rather than one ambiguous
"backend": CPU `numpy.linalg.eigh` for the shared alpha-calibration spectral
decomposition, `numpy.linalg.eigvalsh` for decision-model condition estimates,
`numpy.linalg.cholesky` for decision factorization, and
`numpy.linalg.solve` for the two triangular factor solves. All use float64,
with no hidden jitter, reciprocal condition at least `sqrt(eps)`, and explicit
symmetry/root/solve tolerances. Exactly one BLAS thread is requested. The
calibration lock must record a nonempty path-free BLAS identity and an observed
thread count of exactly one; a request or environment variable alone is not
evidence. The no-solve planner intentionally never loads BLAS. There is no
automatic CPU/CUDA switch. A different routine, device, dtype, or thread count
requires a prospective amendment.

The planner loads a bounded in-memory adjacency. This is suitable for the
current bounded Pearltrees study, not a million-node scaling claim. A large
snapshot that exceeds the explicit planner byte or edge-touch ceiling fails
closed pending a disk-backed adjacency adapter; it does not silently consume
unbounded memory.

The structured statistical contract is complete rather than illustrative. It
contains the absolute candidate gates, the stricter `S_1024`-versus-`R_top`
reference gates, the minimum 18 complete balanced audit batches, conservative
observed-order-statistic tails, all noninferiority margins, extended-real zero
handling, and the calibration rule selecting the smallest adequate `K_low` and
the next larger `K_high` (or the absolute-only `R_top` endpoint when
`K_low=1024`). The paired-bootstrap contract binds the actual multiplicity
artifact, identical multiplicities across endpoints, and the lower/upper
order-statistic endpoint rules.

The effective-resistance arm is fixed before results. If it is predeclared
`omitted`, its absolute-adequacy and noninferiority endpoints are removed
explicitly from the active endpoint sets; omission is neither a passing value
nor a missing-result failure. It cannot be restored after any family result is
seen.

## 6. Artifacts, privacy, and blocked outcomes

The mode-0700 local-only directory contains mode-0600 canonical artifacts for
quartiles, selected anchors, batches, anchor traversals, domains, boundaries,
calibration shells, all 9,999 bootstrap multiplicity vectors, and the manifest.
Every leaf is a unique-link regular file. Node-level material never appears in
stdout. The CLI emits only acceptance, reason, batch counts, and `no_solve`.

A structurally valid graph may produce an installed `accepted=false` plan for
quartile coverage, protected coverage, calibration-shell adequacy, or study
resource inadequacy. This is a complete no-solve scientific result, not an
operational crash. Invalid provenance, privacy, artifact structure, path
identity, or consensus aborts without installing a plan.

The manifest states `solves_executed=0`,
`structural_metrics_computed=true`,
`diffusion_or_fidelity_metrics_computed=false`, and
`audit_solve_authorized=false`; it contains no realized `K_low`, `K_high`, or
measured `alpha_top`. `plan_fingerprint` hashes the scientific
`fingerprint_core`. A separate `manifest_integrity_sha256` seals the complete
manifest outside that field, including acceptance and authorization state.
Neither unkeyed hash authenticates a malicious same-user replacement: the
full upstream verification plus byte-for-byte re-derivation detects accidental
or unsynchronized replacement, but a malicious same-user could replace an
entire self-consistent declaration/receipt/plan/lock chain. Hostile
authenticity requires an external signature or immutable trusted store. The
calibration lock binds the complete manifest content record to prevent ordinary
downstream drift; it is not authentication unless that lock is itself trusted
externally.

Lock verification replays the full declaration, two-attempt consensus, plan
inventory, hashes, content records, and lock-content relationships. It does
not rerun eigendecompositions, bisection, or decision solves, and it does not
turn unkeyed hashes into authentication. Numerical recomputation is a separate
explicit rerun; hostile same-user authenticity still requires an external
signature or immutable trusted store.

## 7. Output transaction and overlap boundary

The plan target must be fresh, below the approved local root, and outside Git.
It may not contain, be contained by, or equal the consensus receipt, either
attempt, the source specification, the relation policy, the complete source
declaration directory, any declared authoritative raw source, or the optional
legacy parity input. This check is repeated during verification so a copied
plan cannot be paired with an overlapping private input layout.

Staging is a descriptor-bound transaction. Files are created relative to a
bound mode-0700 staging-directory descriptor with no-follow, exclusive-create,
mode-0600, unique-link, exact-size, per-file `fsync` checks. The staging
directory is itself synchronized and fully verified, then renamed without
replacement through the still-bound parent descriptor; the parent is
synchronized before success is returned. On failure, cleanup removes only the
known regular unique-link artifact inventory through the bound staging
descriptor. If identity or inventory is no longer provable, cleanup fails
closed and leaves the directory for manual inspection rather than recursively
deleting a possibly substituted path.
