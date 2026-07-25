# Filing path decoder handoff

**Status: prospective implementation contract; not an authorization to train,
score an untouched set, create a folder, or mutate a graph.**

This document replaces
[`DESIGN_lineage_decoder.md`](DESIGN_lineage_decoder.md) as the implementation
handoff for selective filing and new-folder proposals. The older document is
retained as research history. This contract follows the standing decisions in
[`ARCHITECTURE_filing_engine.md`](../../docs/design/ARCHITECTURE_filing_engine.md):

- frozen, revision-pinned e5 is the filing-ranking backbone;
- learned μ heads currently serve calibration, label fusion, and conflict
  routing, not primary filing ranking;
- an existing folder is identified by a typed stable ID, never by its title;
  and
- the recorded canonical/principal path is authoritative for that folder.

“Decoder” here means a **selective search and proposal policy over a frozen
filing catalog**. It is not the reconstruction decoder in
[`DESIGN_expression_encoder_future.md`](DESIGN_expression_encoder_future.md).
That other decoder maps a canonical process-expression AST to its canonical
token sequence to test whether an expression latent preserved syntax. It does
not select folders, infer novelty, name folders, or edit a filing graph. The
two components may eventually exchange typed process metadata, but they do not
share targets or acceptance tests.

## 0. Reuse the landed filing-task authority

This handoff extends the existing content-bound filing chain; it does not define a parallel
baseline:

- `routed_queries.py` and `routed_policy.py` produce and rederive
  `unifyweaver.routed-task.v2`;
- that parent task remains authoritative for source bytes, privacy certification, catalog,
  population, ordered queries and candidate menus, frozen e5 ranker, policy tier, prompt, and
  judge contract; and
- `DESIGN_routed_execution_bundle.md` remains the only hosted-judge execution extension.

The path-decision layer consumes a verified parent task and one exact row by QID. It may add graph
and principal-path receipts, search limits, population-specific proposal calibration, and an
advisory decision. It must not restate or override inherited source, privacy, candidate, ranking,
or policy fields.
Verification calls the existing v2 re-derivation path and byte-compares the parent before reading
any supplement.

Version 1 therefore covers the certified-public parent population. A future local-private parent
contract may reuse the same additive decision layer, but it must bind private source inventory and
privacy classification with equivalent rigor; a caller-provided privacy label is insufficient.
This document does not weaken the public-only boundary of `routed-task.v2`.

## 1. Decision surface

Every successful inference emits exactly one of three decisions:

1. **`SELECT_EXISTING`** — recommend one folder already present in the frozen
   catalog.
2. **`PROPOSE_NEW`** — recommend one or more provisional new folder segments
   below a named existing parent.
3. **`ABSTAIN`** — decline to select or propose because the evidence is
   ambiguous, unsupported, resource-censored, or outside the calibrated
   operating envelope.

All three are advisory and require user confirmation. No decision authorizes
an API call, folder creation, bookmark move, graph edit, or breadcrumb rewrite.
An invalid input or provenance failure produces a blocked error receipt and no
valid decision; it must not be disguised as an ordinary `ABSTAIN`.

### 1.1 `SELECT_EXISTING`

The selected object is a stable typed folder ID. Its display title and
breadcrumb are looked up after selection from the same frozen catalog used for
ranking. The engine must return the catalog's exact recorded
canonical/principal path:

```text
selected folder_id
        │
        └── catalog lookup ──> authoritative path_ids and display titles
```

The model may search with title, semantic, lineage, or graph features, but it
may not generate, repair, shorten, reorder, or substitute that breadcrumb.
Duplicate titles are distinct candidates. A multi-parent folder keeps the
principal-path policy and record chosen by the catalog builder; search-time
graph traversal is not authority to replace it.

### 1.2 `PROPOSE_NEW`

A proposal anchors to an existing parent ID and appends a dynamically sized
ordered list of provisional segments. The initial implementation should
support one new leaf. The schema permits a longer proposed suffix so a later
held-subtree study does not require a fixed-width redesign.

A provisional segment has no remote folder ID. It receives only a local
proposal ID, suggested display name, naming provenance, and privacy class.
Pearltrees or another filing system assigns a durable ID only after explicit
user approval and successful creation. Until then, the proposal is not part of
the catalog or graph.

### 1.3 `ABSTAIN`

At minimum, distinguish:

- `ambiguous_existing` — several existing folders remain plausible;
- `proposal_need_uncertain` — applicable calibrated evidence does not separate
  the proposal-needed regime from difficult-existing cases;
- `outside_calibration_support`;
- `resource_censored`;
- `privacy_restricted_naming`; and
- `no_eligible_candidate`.

These reason codes are operational outputs, not scientific conclusions.

## 2. Ambiguity is not novelty

A small e5 top-one minus top-two margin says that the **observed candidates are
hard to distinguish**. It does not say why. Common causes include duplicated
titles, a broad query, two legitimate homes, incomplete text, or a genuinely
missing folder. Therefore:

- a low margin may route to `ABSTAIN` or optional user/LLM review;
- it must not by itself trigger `PROPOSE_NEW`;
- it must not be reported as a novelty probability; and
- any probability of “no suitable folder exists” requires a separately fit
  and evaluated model on prospectively adjudicated naturally absent cases.

The simulated-absence protocol in §7 may test recovery and authorize only a
clearly labeled, owner-reviewed research pilot. It does not create a live
novelty probability. Ordinary `PROPOSE_NEW` remains disabled until a
prospective naturally absent cohort has produced an applicable calibration
artifact. Before that gate passes, the ordinary legal behavior is e5-based
`SELECT_EXISTING` or `ABSTAIN`.

## 3. Typed request contract

The additive envelope is versioned as `unifyweaver.filing-path-request.v1`. It references one
verified `routed-task.v2` parent and one exact QID; it does not copy the parent row into a new
source of truth:

```text
FilingPathRequest {
  schema_version
  request_id
  parent_task {
    task_id
    task_file_content_record
    qid
    row_sha256
  }
  path_supplement {
    graph_snapshot_sha256
    edge_table_sha256
    principal_path_records_sha256
    principal_path_policy_id
    allowed_roots: [TypedId]
  }
  decision_policy_sha256
  calibration_receipt_sha256 | null
  search_limits {
    maximum_search_nodes
    maximum_search_steps
  }
  external_naming_authorization_receipt_sha256 | null
}

TypedId {
  node_type
  corpus_or_account
  stable_value
}
```

The parent v2 row supplies the typed item, its certified-public privacy receipt, ordered candidates,
e5 scores, catalog and ranker receipts, policy, and content hashes. The new verifier must call the
existing parent re-derivation path and require an exact QID join. A copied task ID or retyped
candidate list is insufficient. It derives `row_sha256` from the canonical parent row after
re-derivation; callers do not get to assert it.

Titles and breadcrumbs are attributes resolved through `folder_id`; they are not join keys. The
request must not duplicate a second, independently derived “best path.” If a display view is
included for convenience, its ID sequence and content hash must match the authoritative path
records.

Graph provenance binds the topology snapshot separately from the principal
path records. This permits outcome-blind structural search over a DAG while
keeping the recorded principal breadcrumb authoritative for display and
action.

External naming authorization is a separate content-bound receipt, never a Boolean. It binds the
owner authorization event, exact request and item-content hashes, inherited privacy receipt,
provider/model/revision, fields permitted to leave the machine, purpose, scope, and expiry. A
missing, expired, or mismatched receipt means local-only naming. The receipt is reverified
immediately before any external call.

## 4. Typed decision contract

The output should be a discriminated union versioned as
`unifyweaver.filing-path-decision.v1`:

```text
FilingPathDecision {
  schema_version
  request_id
  request_sha256
  parent_task_id
  parent_task_file_sha256
  parent_task_row_sha256
  inherited_privacy_receipt_sha256
  decision: SELECT_EXISTING | PROPOSE_NEW | ABSTAIN
  catalog_snapshot_sha256
  ranking_receipt_sha256
  calibration_receipt_sha256 | null
  search_receipt_sha256
  external_naming_authorization_receipt_sha256 | null
  requires_user_confirmation: true
  evidence_summary
  payload: SelectExisting | ProposeNew | Abstain
}

SelectExisting {
  folder_id: TypedId
  authoritative_path_ids: [TypedId]
  path_source: frozen_principal_path_records
  catalog_rank
  e5_score
  top_two_margin
}

ProposeNew {
  anchor_parent_id: TypedId
  authoritative_parent_path_ids: [TypedId]
  proposed_segments: [
    {
      local_proposal_id
      suggested_title
      naming_method
      inherited_privacy_class
      privacy_derivation_receipt_sha256
    }
  ]
  open_set_calibration_id
  calibrated_open_set_score
  operating_threshold_id
}

Abstain {
  reason_code
  candidate_ids_considered: [TypedId]
}
```

`calibrated_open_set_score` may appear only when its calibration artifact
matches the corpus, catalog construction, candidate generator, e5 revision,
privacy policy, and evaluation regime of the request. It is not automatically
a universal probability. The output may include display titles, but consumers
must resolve actions by typed ID and recheck the current catalog immediately
before presenting or applying them.

## 5. Staged implementation

Each stage is useful on its own. Later stages must demonstrate incremental
value against the best earlier stage; they do not inherit authorization merely
because they are more elaborate.

### Stage A — frozen-e5 existing-folder baseline

1. Reverify the `routed-task.v2` parent with the landed code path.
2. Consume the parent row's frozen e5 candidate order; do not rebuild a parallel catalog or
   ranker receipt.
3. Return its top existing folder by typed ID and look up the stored principal path under the
   path supplement.
4. Optionally use a train-only selected margin threshold to `ABSTAIN`.
5. Reproduce the current filing metrics and provenance before adding any new path-search or
   novelty machinery.

This is both the deployment-safe starting point and the rollback baseline.

### Stage B — acceptable-set simulated-absence calibration

First freeze an acceptable-folder set `A_q` for every eligible query using owner/human suitability
labels or another independent, preregistered source. The recorded destination and title aliases
alone are not an acceptable-set oracle: filing can have multiple legitimate homes, duplicate
titles, or a suitable retained parent.

A simulated positive removes **every** acceptable existing folder in `A_q` from the fold-wide
training, calibration, candidate, graph-feature, alias, embedding, and cache inputs. If any
retained folder is independently labeled acceptable, the case is not a positive. Queries whose
acceptable set is unknown or disputed are excluded from decision-bearing calibration and reported
separately.

Build an explicit classifier or calibrated score for `acceptable_existing_folder_visible` versus
`all_known_acceptable_folders_withheld`. Fit its features and threshold only on the
training/calibration partition. e5 score shape, lexical coverage, catalog density, and
outcome-blind structure are possible features; none is a probability without calibration.

This is still **simulated catalog absence**, not proof of organic novelty. It
may validate hidden-folder recovery and support a clearly labeled, bounded,
owner-reviewed research pilot, but it cannot authorize a live claim that “no
suitable folder exists.” That claim requires a later prospective cohort in
which the owner or an independent adjudicator records suitability/new-folder
need without constructing the label by deleting a known folder.

Always run an **always-existing baseline** on the same examples: it selects the
best available existing folder and never proposes. This control measures
whether proposal improves the intended action rather than merely identifying
hard queries. A margin-only proposal rule is another required control, not the
default system.

### Stage C — optional outcome-blind structural candidate search

If candidate recall remains the binding ceiling, add catalog candidates using
only frozen topology, principal/alternative-parent records, revision-pinned
embeddings, and resource limits. Candidate eligibility and search budgets may
not use placements or judge outcomes.

Examples worth testing are ancestors, children, siblings through common
parents, and a bounded graph/resistance search. These extend the set searched;
they do not rewrite principal breadcrumbs. Compare at matched candidate and
resource budgets before allowing downstream scoring to claim a benefit.

### Stage D — hand-designed dynamic path search

Treat a search state as a dynamically sized sequence of existing typed IDs
plus, only after the proposal gate, an optional provisional suffix. Initialize
from the e5 top candidates. Every existing-node transition must be a legal
edge or an explicitly registered catalog transition.

There is no fixed 26-slot representation. Maximum depth, node expansions, and
steps are resource ceilings recorded in the receipt, not the semantics of a
path. The search may revise an earlier branch, but a final existing-folder
answer resolves to the catalog's stored principal path.

Every iterative algorithm must retain a `best_so_far` state under its frozen
objective and deterministic tie rule. On plateau, invalid transition,
nonfinite score, timeout, or resource exhaustion, return that state if it is
otherwise eligible, or `ABSTAIN`. Never return merely the last iterate.

### Stage E — learned optimizer, only after a measured search gap

A learned update policy is justified only if Stage D leaves a prespecified
held gap that cannot be closed by more candidates or a stronger deterministic
search at matched cost. Freeze the e5 backbone while selecting the policy,
compare with the Stage-D best-so-far baseline, and preserve projection onto
legal catalog states. The learned policy may suggest search moves; it never
becomes the authority for existing breadcrumbs.

## 6. Naming proposed folders

Placement and naming are separate decisions. First choose
`PROPOSE_NEW(anchor_parent_id, proposed_suffix)`; then generate one or more
candidate display names. A poor or blocked namer must not force a different
existing-folder selection.

Use the least expansive naming mechanism that works:

1. deterministic templates or extraction from the item and local sibling
   vocabulary;
2. a revision-pinned local embedding/model operating inside the same privacy
   boundary; then
3. an external model only for data certified public, or with a valid
   content-bound owner-authorization receipt for the exact private request,
   permitted fields, and provider/model revision.

Private and unknown inputs default to local-only processing. Proposed titles,
parent IDs, embeddings, prompts, model outputs, receipts, and calibration rows
inherit the strictest privacy state of their sources. Private-derived artifacts
remain outside Git and synchronized/public directories with restrictive local
permissions. No external LLM may receive private data merely because its
output is “only a name.”

Privacy class is derived from the reverified parent and every additional
source receipt; it is not accepted from a request field. For private data, the
authorization receipt must itself bind that content-bound privacy derivation.
Unknown data remains local-only: owner authorization cannot substitute for a
missing privacy classification.

The naming component emits suggestions, not a durable node. The user sees the
existing parent breadcrumb, proposed suffix, alternatives, and privacy status
before confirming or editing it.

## 7. Evaluation contract

Freeze a separate protocol and manifests before any decision-bearing outer
score. Exploratory development results remain exploratory.

### 7.1 Populations and splits

Evaluate at least two strata:

- **acceptable-existing stratum:** at least one folder in the independently frozen acceptable set
  `A_q` remains eligible in the frozen catalog; and
- **simulated-absence stratum:** every folder in `A_q` is removed fold-wide from training,
  calibration, candidate generation, graph features, aliases, embeddings, and caches.

Single-destination leaf removal without an acceptable-set audit is only a
**hidden-node recovery diagnostic**. It is not open-set ground truth and
cannot train or authorize `PROPOSE_NEW`. For an eligible simulated leaf case,
retain the recorded principal parent only when it is not itself in `A_q`, and
remove the acceptable leaves' IDs, titles, contents, placement-derived
statistics, aliases, and embeddings from the visible fold. For proposed
intermediate paths, hold out every acceptable subtree and all descendants'
destination evidence. The latter is a separate, harder experiment and must
not be inferred from leaf recovery.

Use an exposure graph and node-disjoint partition so query IDs, destination
folder IDs, held subtrees, and any shared placement units do not cross the
relevant train/calibration/audit boundary. Threshold selection uses inner
training/calibration only. The untouched outer simulated-absence set is read
once.
Freeze the candidate catalog before scoring.

The simulated-absence manipulation changes only availability. It must not leave behind any
withheld acceptable title embedding, recorded placement count, path feature containing a withheld
node, cached candidate order, or alias keyed to a withheld ID. The acceptable-set ledger and its
source remain sealed from the feature worker.

A later prospective naturally absent/user-adjudicated cohort is a separate stratum and the only
one capable of supporting a direct live-novelty claim.

### 7.2 Required controls

Run on identical manifests:

1. frozen e5 top-1;
2. the always-existing baseline;
3. e5 plus margin-based abstention;
4. a margin-only snap-or-propose rule;
5. the calibrated simulated-absence decision and, only when available, a
   separately calibrated prospective naturally absent decision;
6. the calibrated decision without each added feature family; and
7. when applicable, deterministic search versus learned optimizer at matched
   candidate and resource budgets.

### 7.3 Metrics

Report the three actions rather than collapsing them:

- exact folder-ID Recall@1 and MRR for `SELECT_EXISTING`;
- coverage, selective risk, and AURC when abstention is enabled;
- false-proposal rate on existing-folder cases;
- proposal precision, recall, and precision-recall curve, reported separately
  for simulated absence and prospectively adjudicated natural absence;
- exact anchor-parent accuracy, longest-correct-prefix, and path edit distance
  for `PROPOSE_NEW`;
- calibration diagnostics with the exact simulated or prospective population
  named for every proposal-need score;
- action-confusion counts, including `ABSTAIN`;
- node expansions, steps, runtime, peak memory, and resource censoring; and
- paired typed-node-block uncertainty for the frozen primary contrast.

Choose the primary action utility, costs, minimum practical effect, intervals,
and multiplicity handling before the outer set is opened. AUROC alone is
insufficient when missing folders are rare.

### 7.4 Anti-circular grading

Do not grade a proposed folder embedding solely by cosine under the same e5
model that constructed or optimized that embedding. That establishes
self-consistency, not recovery. e5 similarity may be printed as a diagnostic
only when labeled circular.

Decision-bearing proposal evaluation uses catalog facts independent of the
proposal geometry: hidden parent ID, path prefix, exact held ID when
applicable, and user action utility. Naming quality requires a separately
pinned evaluator not used to generate or rank the name, deterministic lexical
metrics with known limits, or blinded human review. An independent embedding
model may be a sensitivity analysis if its revision and text contract were
frozen before results.

## 8. Provenance and fail-closed behavior

Every run writes content-bound, no-replace artifacts:

- request and ordered candidate manifests;
- the reverified parent `routed-task.v2` content record and exact parent-row
  digest;
- graph, catalog, principal-path, privacy, and e5 receipts;
- acceptable-set, split, and fold-wide removal manifests for simulated
  evaluation, plus separate prospective-label provenance when applicable;
- any external-naming authorization receipt and the privacy derivation it
  binds;
- feature, model, calibration, and threshold receipts;
- per-step search trace including `best_so_far`;
- complete action record and user-confirmation state; and
- software commit, dependency versions, seeds, deterministic settings,
  resource ceilings, runtime, and peak memory.

Hashes establish byte identity, not privacy, provider authenticity, or
scientific independence. Do not include private titles or paths in a public
receipt merely to make it reproducible.

Fail closed before inference or action on:

- missing, duplicate, untyped, or title-only identities;
- a stale or mismatched catalog, graph, path, privacy, ranker, or calibration
  receipt;
- a selected folder without an authoritative catalog path;
- an invalid path transition, nonfinite score, or nondeterministic tie;
- a proposal outside the exact simulated or prospective calibration support
  claimed for it;
- an external naming request for private data without a valid, exact
  authorization receipt, or for unknown data under any circumstance;
- an attempt to apply a decision without user confirmation; or
- a catalog change between recommendation and confirmation.

Ordinary ambiguity may produce `ABSTAIN`; provenance failures block the run.

## 9. Implementation acceptance tests

At minimum, tests must prove:

1. duplicate titles remain separate and actions use typed IDs;
2. an existing multi-parent folder returns the frozen principal breadcrumb,
   even when search reached it through another edge;
3. paths of depth 1, the current maximum, and greater than 26 round-trip
   without a fixed-slot assumption;
4. `SELECT_EXISTING`, `PROPOSE_NEW`, and `ABSTAIN` are mutually exclusive and
   schema-valid;
5. absence of an eligible candidate leaves no stale selected/proposed target;
6. a low margin cannot authorize `PROPOSE_NEW` without a matching
   population-specific calibration artifact;
7. threshold fitting and feature selection cannot read outer
   simulated-absence labels;
8. simulated-absence construction removes every acceptable folder's IDs,
   titles, paths, embeddings, aliases, contents, and placement-derived traces
   fold-wide;
9. the always-existing and margin-only controls run on byte-identical
   manifests;
10. a fixed-resource timeout returns the deterministically scored
    `best_so_far`, not the last iterate;
11. the request rederives its exact `routed-task.v2` parent and row rather than
    accepting a duplicated baseline schema;
12. private inputs cannot reach an external namer without an exact
    content-bound receipt, and unknown inputs cannot reach one at all;
13. stale hashes, malformed paths, illegal transitions, and nonfinite values
    block;
14. recommendation code cannot mutate the source graph or call a filing API;
15. confirmation revalidates the catalog snapshot before any separately
    authorized action; and
16. the decision-bearing evaluator rejects same-e5-only grading of an
    e5-constructed proposal.

Property-based tests should generate dynamic-depth trees/DAGs, duplicate
titles, alternative parents, and adversarial catalog reorderings. Transaction
tests should cover prepare → verify → recommend, tamper rejection, and
reproduction at identical hashes.

## 10. Activation and stop gates

1. **Contract gate:** land the schemas and a no-mutation interface before
   implementing a proposal model.
2. **Baseline gate:** reproduce the frozen-e5 existing-folder baseline with
   exact IDs, authoritative breadcrumbs, and content-bound receipts. Stop and
   fix provenance if it does not match.
3. **Proposal gate:** preregister the acceptable-set simulated-absence study
   before constructing its outer set. Passing it may authorize only the
   labeled research pilot; ordinary `PROPOSE_NEW` and a live novelty
   probability remain disabled until a separate prospective naturally absent
   cohort meets its frozen calibration and false-proposal rule.
4. **Candidate gate:** add structural search only if it improves outcome-blind
   candidate coverage/stability at a matched budget. An empty increment is a
   valid stopping result.
5. **Optimizer gate:** add a learned optimizer only after deterministic search
   leaves a prespecified held gap. Roll back unless it beats the best earlier
   stage at matched resources under the frozen node-disjoint evaluation.
6. **Deployment gate:** successful evaluation authorizes at most a
   user-confirmed advisory pilot. Automatic graph mutation, private external
   inference, new-corpus transfer, and public release each require a separate
   authorization.

## 11. Engineering handoff checklist

**Landed:** [`filing_path_decision.py`](filing_path_decision.py) implements the
Stage A slice below, with acceptance tests in
[`test_filing_path_decision.py`](test_filing_path_decision.py). It emits only
`SELECT_EXISTING` or `ABSTAIN`; `PROPOSE_NEW` has no reachable code path.

The first engineering PR should implement only Stage A and the common request /
decision schemas. It should:

- rederive and consume one exact `routed-task.v2` parent row, preserving its
  source, privacy, catalog, ordered-candidate, ranker, and policy authority;
- add only the graph/principal-path supplement and decision receipt rather
  than rebuilding a parallel baseline;
- resolve top-1 by typed ID;
- copy the authoritative principal path from the catalog;
- emit `SELECT_EXISTING` or `ABSTAIN`;
- produce a no-replace search/decision receipt;
- expose no graph-mutation capability; and
- include the identity, duplicate-title, principal-path, dynamic-depth,
  provenance, and privacy tests above.

The next PR should freeze the acceptable-set simulated-absence protocol and
fixture builder. A later PR may implement only the labeled research-pilot
proposal path. Ordinary `PROPOSE_NEW` waits for the separately frozen
prospective naturally absent protocol and cohort. This ordering keeps each
target from being defined after its results are visible.

## 12. Explicitly deferred or rejected

- Treating low margin as novelty probability — rejected.
- Generating IDs or existing-folder breadcrumbs — rejected.
- Replacing the stored parent walk with a learned/generated path — rejected.
- A fixed 26-slot output — rejected.
- Training a μ head as the default primary ranker — rejected absent new held
  evidence.
- Same-e5 proposal generation and decision-bearing grading — rejected as
  circular.
- Automatic folder creation or filing — outside this architecture.
- External naming on private data by default — rejected.
- Full text-autoregressive path generation — deferred until selective
  search, population-specific proposal calibration, and compact local naming
  all show a prespecified unmet need.
