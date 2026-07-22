# Privacy-aware Pearltrees STEM harvest planning

## Status and scope

This document specifies the first acquisition step after the Pearltrees
diffusion snapshot work: a deterministic, graph-first plan for filling public
STEM coverage gaps.  The implementation is
`prepare_pearltrees_stem_harvest_plan.py`.

The planner is **local-only** and outcome-blind.  It reads existing artifacts
and writes work orders; it does not call Pearltrees, an embedding model, an LLM,
or a filing judge.  It does not authorize training, publication, or a HOP
fidelity rerun.  Those remain downstream of a newly compiled and verified
snapshot.

## Decision

Harvest outward from an explicit allowlist of core-STEM roots using only the
containment relations `collection`, `ref`, and `path`.  Stop a branch at the
first node that lacks a policy-fresh public API response.  Fetching or locally
revalidating that frontier and recompiling the snapshot is one iteration; the
planner is then run again from scratch.

This makes the acquisition sequence conservative in two senses:

1. An unknown, masked, malformed, or restricted node is never used as a bridge
   into its descendants.
2. A stale work order is never appended to after new visibility evidence
   arrives.  A new private ancestor can therefore invalidate and scrub the
   branch before later work is scheduled.

## Inputs

### Verified diffusion snapshot

The authoritative graph and privacy state come from a successfully verified
`pearltrees-diffusion-snapshot-v1` directory.  An exit-2 snapshot with retained
unknown visibility is allowed as an acquisition input: those unknown nodes are
exactly part of the coverage problem.  It is not a training or publication
asset.

`assembled_dag.tsv` is not authoritative.  It loses relation and visibility
provenance, mixes historical inputs, and uses `child<TAB>parent` orientation.
It remains a parity diagnostic only.

### Seed manifest

The human-curated seed file has this exact shape:

```json
{
  "exclude_roots": [],
  "public_cache_not_before": "2026-07-21T00:00:00",
  "roots": ["pt:123"],
  "schema": "pearltrees-stem-harvest-seeds-v1",
  "snapshot_fingerprint": "<64 lowercase hex characters>"
}
```

Each root must be retained and explicitly public in that exact snapshot.
`exclude_roots` are optional topical scope cuts, not privacy overrides.  Their
containment descendants are omitted even if another path could reach them.
Changing the snapshot requires an explicit seed-manifest revision.
`public_cache_not_before` is an offset-free timestamp in the harvester's local
clock domain.  It is a privacy policy chosen prospectively, before queue counts
are inspected.  A cached public response older than that cutoff is unresolved;
it cannot open traversal merely because it was public when first fetched.

Core STEM means roots deliberately chosen for mathematics, physical sciences,
computer science/AI, engineering, chemistry, biology, and related technical
disciplines.  Broad environment, development, or political collections are not
implicitly promoted by title similarity.

### Raw API response cache

Completeness is derived from valid embedded ID claims in `api_responses.db`,
never filenames or `batch_state.json`.  The exact database SHA-256 must occur in
the verified snapshot as an `api_sqlite` source; matching an unrelated source
hash is insufficient.  The SQLite file must be checkpointed (no WAL, SHM, or
rollback-journal sidecars and no WAL-mode header), pass `quick_check`, and
remain byte-stable through planning.  The streaming hash has a separate 2-GiB
ceiling so corpus growth does not inherit the small JSON-artifact limit.

A row counts as fetched-public only when every embedded `info`/`tree` ID claim
agrees with the cache key, no title or visibility claim is private, at least one
visibility claim is canonical zero, and `fetched_at` meets the frozen cutoff.
The snapshot and cache are therefore two views of the same raw evidence rather
than asynchronously mixed states.

The current stores disagree in coverage, and the per-tree JSON directory also
contains a batch-array file that the strict snapshot adapter rejects.  V1 uses
the explicit raw-response SQLite cache.  A later prospective source adapter may
take the validated union; it must not silently skip incompatible files.

## Privacy states and lanes

The snapshot remains the snapshot-relative visibility authority.  “Public” in
this plan is not a timeless claim about visibility, licensing, consent, or
suitability for third-party model training.  Cache inspection decides whether a
node is a fresh enough acquisition frontier.

- `public`: explicit snapshot-public, not excluded, and backed by a public cache
  row at or after the prospective cutoff.
- `private_or_restricted`: snapshot-private/excluded, or a nonzero API
  visibility.  It is omitted from every work order; only aggregate counts are
  reported.
- `stale_public`: snapshot-public with a matching cache row older than the
  cutoff.  It enters the local revalidation lane and terminates traversal.
- `unknown` / `missing`: never inferred public.  A first reachable frontier
  enters the appropriate local lane and terminates traversal.
- `masked_auth` / malformed / contradictory evidence: consumed conservatively
  by the upstream snapshot compiler.  Masked or private nodes are excluded and
  malformed inputs fail snapshot preparation; the planner does not resurrect
  them as harvest candidates.

The output lanes are deliberately separate:

1. `public_harvest_queue.json` contains only snapshot-public IDs whose exact
   API response is absent.
2. `visibility_revalidation_queue.json` is local-private quarantine work for
   stale-public or missing/unknown frontiers.

No title, URL, account, raw API body, local path, embedding, prompt, or judge
output appears in these work orders.

## Graph search and priority

For each root independently, breadth-first search follows directed
parent-to-child containment.  A node is traversable only when both the snapshot
and the policy-fresh cache resolve it public.  Multi-parent nodes are retained
as one node; visited sets break cycles without selecting a preferred parent.

Priority is a frozen lexicographic order, not a confidence-weighted score:

1. unresolved seed (`tier 0`);
2. shared containment frontier touching at least two public in-scope parents or
   at least two roots (`tier 1`);
3. other first unresolved containment frontier (`tier 2`).

Within a tier, order by shortest root distance, descending public frontier-link
count, descending distinct-root count, then numeric tree ID.  Frontier and root
counts measure structural leverage; they are not probabilities and never modify
visibility.  The public and revalidation lanes each apply the frozen batch
limit after sorting and report both emitted and total candidate counts.

Non-containment `alias`, `shortcut`, and `cross_link` evidence is deliberately
deferred.  It can later suggest where to inspect for bridges, but must not widen
the first public STEM acquisition frontier.

## Local installation and provenance

The planner writes an exact mode-0700 directory of mode-0600, single-link files
under an explicit local root, outside every Git worktree.  Installation uses an
atomic Linux no-replace rename.  The manifest binds:

- snapshot fingerprint;
- semantic seed/root-set hash;
- raw API cache hash;
- prospective cache cutoff, containment relation policy, hop limit, batch
  limit, frozen policy, and an exact aggregate-count schema;
- byte count and SHA-256 of every work-order artifact; and
- a plan fingerprint over the complete frozen core.

The `verify` command requires the snapshot, cache, and seed manifest again.  It
re-runs the graph search and compares both work-order artifacts byte for
byte.  Artifact hashes alone provide integrity, not authenticity; they are not
accepted as a substitute for replay against the bound inputs.

The only console output is aggregate counts and the plan fingerprint.  Errors
are generic so private titles, IDs, and paths are not copied into logs.

Example:

```bash
python3 prototypes/mu_cosine/prepare_pearltrees_stem_harvest_plan.py prepare \
  --snapshot-dir /private/snapshots/attempt-a \
  --api-cache-db /private/pearltrees/api_responses.db \
  --seed-manifest /private/config/stem-seeds.json \
  --local-root /private/harvest-plans \
  --output-dir /private/harvest-plans/plan-001 \
  --max-hops 8 \
  --batch-limit 128 \
  --local-only
```

```bash
python3 prototypes/mu_cosine/prepare_pearltrees_stem_harvest_plan.py verify \
  --run-dir /private/harvest-plans/plan-001 \
  --snapshot-dir /private/snapshots/attempt-a \
  --api-cache-db /private/pearltrees/api_responses.db \
  --seed-manifest /private/config/stem-seeds.json
```

## Current harvester execution gate

Inspection of the separate local harvester found a fail-open operational path:
a masked authentication response can be persisted and marked successful when
credential refresh is unavailable or fails.  Therefore this PR authorizes work
order **generation only**.  Neither queue should be executed until the harvester
is changed to:

- quarantine masked/auth-failure responses rather than recording success;
- stop the run when reauthentication cannot be established;
- never mark those IDs processed;
- use a fresh state file bound to the work-order fingerprint; and
- recheck the plan fingerprint, cutoff, and current visibility immediately
  before every network batch (a generated plan otherwise ages); and
- treat every raw response, log, state file, and cache as local-private.

After each bounded batch: checkpoint the cache, compile raw evidence again, run
the independent snapshot-consensus step, regenerate the plan, and only then
resume acquisition.

## Optional public-data model refinement

Phi-family inference is not in the critical path.  A hosted or local model may
later classify overlap that has been freshly revalidated public to improve STEM
relevance or train a small local router.  It may not decide privacy, seed
membership, or whether an unknown node is safe to send externally.  Requests
must minimize payloads and enforce the chosen provider's zero-data-retention
route even for public input.  A separate `external_processing_allowed` gate
must check the source terms and disclosure policy: public visibility alone is
not consent or authorization for third-party processing.

For a reproducible API experiment:

- discover and freeze the live model plus provider endpoint before labeling;
- use an explicit model/provider, deterministic parameters, and structured
  output rather than a random free-model router;
- disable training/logging and require a zero-data-retention endpoint even
  though the payload is public;
- record prompt/version/model/provider hashes and costs locally; and
- split public nodes into train/calibration holdouts blocked by lineage or
  parent family before labels arrive; merely node-disjoint rows can still leak
  closely related graph families.

When API labels are fused with existing mu/judge sources, they are another
correlated signal and should enter the calibrated joint-posterior and
margin-gate workflow, not a hand-set weighted blend.  A standalone STEM router
does not require that exact combiner, but it still requires calibration and a
lineage-block-disjoint held-out evaluation.  Preserve exact request/response
provenance locally; deterministic parameters alone do not make a changing
hosted endpoint reproducible.  Value is established by held-out filing or
acquisition decisions, not agreement with the same model.

## Future private harvester repository

The browser-automation harvester is already a nested local Git repository.  A
separate private GitHub repository is reasonable, but repository creation is
not part of this PR.  Before its first push:

1. confirm the remote repository is private;
2. start from a fresh/orphan allowlisted history containing only code, public
   documentation, and synthetic fixtures; copying an allowlist into the current
   branch does not cleanse reachable old commits;
3. exclude cookies, API configuration secrets, raw JSON/SQLite, RDFs, logs,
   state/PID files, work orders, embeddings, models, and generated outputs;
4. scan the entire history and pending objects for secrets and private data;
5. rotate any credential that ever entered Git; and
6. test a clean clone using synthetic data before adding the real local paths.

Private visibility data can support a local personalization overlay, but every
transitively derived embedding, adapter, checkpoint, prompt cache, and report
then remains private.  The public corpus and the private overlay are separate
trust domains, not two splits of one publishable dataset.

## Alternatives rejected for V1

- **Title or embedding similarity as the primary frontier:** risks semantic
  scope drift and can depend on private text.  Graph containment is the primary
  geometry; semantic models may later rank only freshly revalidated public ties.
- **Traverse through unknown nodes:** can expose a private subtree before its
  ancestor is resolved.
- **Treat `*private*` as proof of privacy or proof of auth failure:** either
  inference can be wrong.  The response stays quarantined until authenticated
  revalidation.
- **Use legacy assembled DAG or harvester state as truth:** both collapse or mix
  provenance and have known coverage/ID defects.
- **One weighted relevance score:** disguises policy choices as statistical
  confidence.  The explicit tiers make each decision auditable.
- **Run Phi-3 first:** adds cost, reproducibility, and data-handling questions
  before the graph and privacy boundary are settled.
