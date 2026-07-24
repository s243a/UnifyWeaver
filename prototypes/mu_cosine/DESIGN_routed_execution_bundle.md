# Routed filing execution bundle

## Status and scope

This document freezes the no-spend execution layer between a certified-public
`unifyweaver.routed-task.v2` parent task and any future hosted judge calls. It
does not call a provider, select a judge, tune a routing threshold, inspect a
placement label, or authorize a confirmatory filing result.

The execution layer solves four narrower problems:

1. split one parent task into provider-sized calls without losing provenance;
2. repeat the same judge procedure three times without confusing retries with
   independent draws;
3. counterbalance candidate presentation positions deterministically; and
4. derive one parent-compatible pick artifact with a frozen aggregation rule.

The parent task remains the authority for the public-only population, candidate
menus, ranker, policy tier, prompt, and judge contract.

## Local interface

`routed_execution.py` is deliberately a no-spend tool. Its subcommands are:

- `plan`: re-run the current certified-public parent-task build, then install a
  self-contained plan directory with copied parent/policy/prompt bytes and all
  child task/request bytes;
- `verify-plan`: re-derive every child and request from those copied inputs;
- `seal-attempt`: copy one externally obtained raw response or failure body into
  the contiguous attempt history;
- `verify-attempt`: re-parse and re-derive one sealed attempt;
- `build-bundle`: require terminal success for every logical call, aggregate,
  and install `derived/execution-bundle.jsonl` plus
  `derived/aggregate-picks.jsonl`; and
- `verify-bundle`: re-derive the complete attempt set, votes, parent picks, and
  both derived artifacts.

There is no flag that supplies or bypasses the current-source certification.
Tests replace the certifier in-process; the command-line planner always performs
the rebuild. Plan, attempt, and derived directories use atomic Linux
no-replace installation. Re-running a creator against an existing target is an
error; verification is the only accepted resume/read path. Attempt staging is
kept outside the scanned attempt tree, so a host failure can leave inert local
staging debris but cannot turn a valid retry prefix into an unresumable
noncanonical attempt sequence. A no-follow plan-wide writer lock serializes
attempt sealing and final bundle construction, preventing concurrent sealers
from both accepting the same provider ID or retry index.

Version 1 favors re-derivation over throughput: each seal verifies the complete
plan and every earlier attempt. That is appropriate for the expected tens to
low hundreds of logical calls, but it is quadratic in attempt count. A
content-bound append index or batch sealer is required before using thousands
of chunks; support for four-digit path indices is not a performance claim.

## Artifact chain

```text
certified-public parent task
        |
        v
execution plan
  |  exact child tasks + rendered request bytes
  |  three draws × stable chunks
  v
attempt artifacts
  |  retry history + provider-call IDs + raw hashes
  v
verified execution bundle
  |  canonical folder-ID votes
  v
parent-bound routed-picks.v2 aggregate
```

Every arrow is content-bound. Paths are only locators; identifiers derive from
canonical content.

## Parent certification and privacy

Planning must first reproduce the parent task from the current certified-public
source, privacy index, catalog, population, ranker, and frozen policy. Merely
rehashing a previously written task is not a fresh privacy check.

After that check, planning is a pure projection:

- it may slice and reorder existing public parent rows;
- it may rotate existing menu items and rewrite their displayed `pos`;
- it may not add a bookmark, folder, lineage node, title, URL, or source field;
- it may not read `true_folder_id`, a placement label, judge output, or result;
  and
- inverse presentation maps must preserve each menu item, folder ID, title, and
  lineage path exactly.

Task, request, response, and attempt artifacts can still reveal browsing
interests despite containing only certified-public nodes. They are local-only
run artifacts and must not be committed.

## Frozen schedule

Version 1 fixes:

- draw count: **3**;
- chunk membership: contiguous slices of parent task row order;
- chunk membership is identical in every draw;
- child order: draw-major, then chunk-major;
- menu rotation: a stable per-QID base derived from the schedule identity,
  followed by one cyclic step per draw;
- aggregation: strict majority over canonical folder IDs, with `null` as a vote;
  and
- missing/failed calls: fail closed, with no imputation.

Each draw is an exact duplicate-free partition of the parent QIDs. Stable chunk
membership avoids adding a changing prompt-neighborhood confound across draws.

For every QID, cyclic presentation makes each candidate occupy three distinct
positions when the menu has at least three entries. More generally, the
per-position counts for a candidate differ by at most one over the configured
draws. The plan binds the presented-to-canonical inverse map; aggregation never
votes directly on a displayed integer position.

## Child task and request identity

Each logical provider call is a child `routed-task.v2`. Its core inherits the
parent's source, privacy, catalog, population, ranker, selection, policy, and
judge contract, and adds:

- the schedule identity and parent task identity;
- draw and chunk indices;
- total draws and chunks;
- the exact ordered QID subset; and
- the presentation-map digest.

Consequently, otherwise identical rows in different draws or chunks still have
different child task IDs. The frozen judge prompt already requires the raw
response to echo that exact child task ID, which transitively binds execution,
draw, chunk, QIDs, and presentation.

The rendered request is deterministic prompt bytes plus the exact child task
JSONL bytes. Its size and SHA-256 digest are part of the plan. A provider adapter
must submit those logical request bytes unchanged or create a new plan version
that binds its transport representation.

## Attempts, retries, and provider provenance

A draw/chunk is one logical call. A retry is another attempt at that same call,
not another draw and never another vote.

Attempts are zero-based and contiguous. Their allowed states are:

- `retryable_failure`;
- `success`; or
- `terminal_failure`.

There is exactly one terminal outcome per logical call and no attempt may follow
it. A complete aggregatable bundle requires terminal `success` for every
draw/chunk. Every attempt binds:

- the plan, child task, rendered request, draw, and chunk;
- attempt index and status;
- the exact required judge/model/revision/prompt/settings contract;
- provider run, request, and response IDs;
- start and completion timestamps; and
- the raw response byte count and SHA-256 digest.

Provider request IDs and nonempty response IDs are globally unique across the
bundle. Exact response-byte reuse also fails because each successful raw header
must bind a distinct child task ID. These are declared, content-bound provenance
checks; they are not cryptographic provider attestation.

## Aggregation

Successful displayed positions are first mapped back to canonical folder IDs
using the verified child menu. `null` remains `null`.

For each parent QID, all three scheduled votes are required:

- a folder with at least two votes becomes that folder;
- at least two explicit `null` votes produce `null_majority`; and
- three votes without any strict majority produce `no_consensus`.

Both latter cases map to `pick: null` in the parent-compatible
`routed-picks.v2` artifact, but the bundle retains their distinct reasons.
Missing draws and terminal failures are neither reason; they block aggregation.

The aggregate pick provenance binds the plan identity, all successful attempt
identities plus failed retry identities, the vote-record digest, and the
aggregation contract. Retries affect the attempt-set digest but never add a
vote. The bundle then binds the aggregate file, avoiding a circular
bundle/aggregate identifier.

## Statistical boundary

Three repeated calls are correlated views, not three independent confidence
weights. Agreement is descriptive and is not a calibrated correctness
probability. Draw index also fixes the cyclic presentation rotation, so
disagreement conflates repeat-call variation with menu-position sensitivity;
these three views do not identify a pure judge-variance component.

Chunking also introduces shared prompt/call dependence among all QIDs in one
provider request. The existing bookmark/folder connected-component bootstrap
does not represent that source of variation. Dependence can also be shared by
draw, provider run/session, time, or model instance. Before a decision-bearing
result, a new preregistration must name the estimand, model at least stable
chunk incidence plus every applicable shared execution unit, report effective
independent block counts, and fail closed when those counts are inadequate.
Unioning chunk IDs with the existing bookmark/folder blocks is one conservative
candidate, not automatic authorization. Until then:

- execution completeness and integrity may pass;
- per-draw deltas, agreement, null rate, and no-consensus rate may be reported
  descriptively; but
- confirmatory inference remains unauthorized.

## Explicit deferrals

- provider signatures or cryptographic attestation;
- adaptive chunk sizing or early stopping;
- plurality, Bayesian, weighted, or confidence-based aggregation;
- cross-provider or mixed-judge ensembles;
- missing-response imputation;
- changing chunk membership between draws; and
- treating retries as independent evidence.
