# Pearltrees diffusion snapshot: a privacy-preserving compiler

**Status:** prospective implementation contract. This document implements the
raw-source preparer gate in
[`PROTOCOL_bounded_diffusion_fidelity.md` §2](PROTOCOL_bounded_diffusion_fidelity.md#2-frozen-graph-and-conductance-regimes).
It does not report a real-corpus result, authorize a solve, or authorize automatic
publication. `graph_asset_ready` means only that this compiled graph passed its
declared privacy and anchor-coverage checks. It is **not** study readiness: the
fidelity protocol still requires a second deterministic compilation with a
matching fingerprint plus its separately frozen run manifest. All detailed
artifacts are local-only.

## 1. Why this is a compiler

The preparer compiles immutable, typed source evidence into one deterministic
graph snapshot:

    declared raw sources
      -> typed nodes + typed relation evidence
      -> conservative privacy union and containment closure
      -> scrubbed nodes/relations
      -> explicit physical-edge policy
      -> reciprocal adjacency + components + anchor eligibility
      -> canonical manifest and content fingerprint

Parsing, privacy, and physical-edge policy are separate stages. Evidence is never
silently promoted into physics, and a physical edge never retroactively changes
privacy. The compiler is read-only on every input and must not harvest, repair, or
infer missing corpus data.

## 2. Command-line contract

The v1 entry point is `prepare_pearltrees_diffusion_snapshot.py` with three
subcommands:

```text
prepare --source-spec SPEC.json --relation-policy POLICY.json --run-dir DIR \
        --local-root LOCAL_ROOT --local-only --resource-ceiling-bytes N \
        [--minimum-anchors N]
verify  --run-dir DIR
status  --run-dir DIR
```

- `prepare` requires `--local-only`; there is no publish flag. `LOCAL_ROOT` must
  be an existing non-symlink directory, and `DIR` must be a new child of it,
  outside the Git repository. The preparer validates source inventories before
  and after parsing, compiles in a sibling staging directory, verifies the staged
  snapshot, and atomically installs `DIR` only on success.
- `--minimum-anchors` defaults to 128. Together with privacy certification it
  controls `graph_asset_ready`; legacy parity is diagnostic only and cannot change
  readiness or graph compilation. `--resource-ceiling-bytes` is a hard
  prepare-time byte-contract gate described in Section 6.
- `verify` is a self-contained, read-only artifact verifier. It checks the exact
  file inventory, local-only marker, canonical encodings and hashes, typed-node
  uniqueness, exclusion from adjacency, adjacency reciprocity, component
  partition, eligible-anchor count, and snapshot fingerprint. It does not reopen
  the raw sources; source stability is checked during `prepare`.
- `status` first verifies the run, then prints only the snapshot fingerprint,
  readiness/certification flags, aggregate edge/anchor counts, and the causally
  disconnected legacy-parity status. It never
  prints node IDs, titles, paths, URLs, or source rows.
- An existing `run-dir`, empty or not, is never overwritten. Rebuilding means
  preparing a new directory and comparing fingerprints.

Normal completion is either a verified snapshot or a fail-closed diagnostic. A
verified but blocked compilation is installed with `graph_asset_ready=false`; an
integrity failure never leaves a directory that `status` calls ready.

## 3. Source specification

`SPEC.json` has schema `pearltrees-diffusion-source-spec-v1` and this exact
shape (with `legacy_check` optional):

```json
{
  "schema": "pearltrees-diffusion-source-spec-v1",
  "snapshot_label": "local descriptive label",
  "sources": [
    {"kind": "rdf", "source_id": "rdf-primary", "path": "raw/export.rdf", "account": "primary"},
    {"kind": "api_sqlite", "source_id": "api-db", "path": "raw/api.db"},
    {"kind": "api_json_dir", "source_id": "api-files", "path": "raw/trees"},
    {"kind": "path_jsonl", "source_id": "paths", "path": "raw/paths.jsonl"}
  ],
  "legacy_check": {"dag_path": "raw/assembled_dag.tsv"}
}
```

Each source entry must contain a unique nonempty `source_id`, supported `kind`,
and `path`; `account` is optional declared provenance. Relative paths resolve
against the source-spec directory. Implicit globs, environment-dependent
defaults, and “newest file” discovery are forbidden. An `api_json_dir` means
every regular non-symlink member of that explicitly named directory; an empty
directory fails closed.

The four authoritative adapters are deliberately narrow:

- RDF recognizes `Tree`, `RefPearl`, `AliasPearl`, `NotePearl`, supported
  non-folder pearls, and `UserAccount` root evidence under an explicit
  Pearltrees/SIOC namespace allowlist. A `UserAccount` subject may use only the
  repository-observed exact `#sioc` identity fragment; its `rootTree`
  account prefix, and the declared source account must agree. Numeric identities
  must use a Pearltrees HTTP(S) host. A valid account-root `parentTree` URI creates
  root evidence but no synthetic graph node or edge; repository-observed
  `/t/<account>` team-space roots normalize to the declared `groups` account
  provenance. DTD and entity declarations are rejected before XML parsing.
- API JSON accepts an object rooted at `api_response`, `response`, or the object
  itself, with `tree` and/or `info`. `info.parentTree` becomes `collection`;
  `null` or `{}` is explicit root evidence. Content types 2 and 5 become
  `collection` and `shortcut`; repository-observed but semantically unresolved
  type 6 is preserved as non-containment `cross_link`. Supported non-folder
  content types 1, 4, and 7 are ignored structurally; an unknown type fails.
  Simultaneous camelCase/snake_case aliases must be canonically identical.
- API SQLite must expose exactly one of normalized `trees` and `pearls` tables
  with the required columns or `api_responses(tree_id,response_json)`. It is
  opened immutable, read-only, and query-only; ambiguous schemas, WAL mode,
  failed `quick_check`, and `-wal`, `-shm`, or rollback-journal sidecars before
  or after parsing fail closed.
- Path JSONL requires one nonblank strict-JSON object per line with `tree_id`;
  consecutive numeric `path_ids` and optional consistent `parent_tree_id` emit
  `path` evidence. A leading `account:` marker (and the same marker in
  root-level `parent_tree_id`) is validated as root evidence, not a graph ID.

JSON and JSONL reject duplicate object keys, nonfinite constants, invalid UTF-8,
and excessive nesting. Canonical graph IDs are validated during every adapter.
Every declared source is frozen by byte length plus SHA-256 before parsing and
checked again afterward. Directory inventories are content-record sequences;
paths, member filenames, mtimes, and inodes do not enter the scientific
fingerprint. A changed source, unreadable member, schema mismatch, or directory
membership change fails the build.

The compiler records its schema/algorithm version, parsed relation policy,
repository commit, and source content records. An unavailable or malformed
repository commit fails preparation. The commit is bound inside the scientific
fingerprint core along with algorithm, policy, source contents, scientific
artifacts, and the resource ceiling.

## 4. Strict identities and source relations

The phase-one universe contains Pearltrees tree/collection nodes. Its canonical
identifier is `pt:<positive-decimal-id>`. Zero, signs, whitespace, leading-zero
aliases, bare IDs in compiled artifacts, title slugs, URLs used as final
identities, and cross-type reuse are rejected. Account, title, visibility, and
source location are attributes or evidence, never identity. The compiler never
repairs an ID from a title match.

Every accepted source relation is preserved in `edge_evidence.jsonl` before the
physical policy is applied. Canonical relation classes are:

| source evidence | canonical relation | directed containment for privacy? |
|---|---|---:|
| API `parentTree` or collection/content-tree membership | `collection` | yes |
| RDF `RefPearl` inside its `parentTree` | `ref` | yes |
| consecutive declared path IDs or `parent_tree_id` | `path` | yes |
| RDF `AliasPearl` | `alias` | no |
| API content-type-5 shortcut | `shortcut` | no |
| API content-type-6 unresolved content-tree reference | `cross_link` | no |
| reserved explicit associative evidence | `cross_link` | no |

`cross_link` is a conservative holding class for explicit non-containment
references whose topology is observed but whose containment semantics are not.
V1 maps repository-observed content type 6 here rather than guessing that it is
containment, and does not synthesize cross-links from titles, URLs, or generic
`seeAlso` observations. It is normally excluded from physics unless
prospectively licensed. Page pearls,
section labels, external URLs, and
titles do not become folder nodes or edges by inference.

`POLICY.json` has this exact shape and must name a Boolean decision for all six
relations, including reserved `cross_link`:

```json
{
  "schema": "pearltrees-diffusion-relation-policy-v1",
  "physical_edges": {
    "collection": true,
    "ref": true,
    "path": true,
    "alias": false,
    "shortcut": false,
    "cross_link": false
  }
}
```

There is no default or “all links” mode, and at least one relation must be
included. An included relation contributes one canonical undirected physical
edge; duplicate evidence remains provenance on that edge, not parallel unit
conductance. A self relation is an input-integrity failure. The resulting
`physical_edges.tsv` and `adjacency.jsonl` encode symmetric unit conductance for
the HOP operator.

Every source/record visibility claim, including an explicit missing/unknown
observation, is preserved in `visibility_evidence.jsonl`. Final node visibility
and certification are derived from that ledger and independently reconstructed
by `verify`. Conflicting titles, accounts, or known visibility observations are
recorded in `conflicts.jsonl`. Private visibility wins conservatively; malformed IDs,
unsupported relations/content types, ambiguous XML fields, and inconsistent API
ID fields fail during parsing rather than being guessed.

## 5. Privacy is computed before physics

Privacy uses the repository's scrub-everywhere policy and is a monotone fixed-point
calculation on the full typed evidence graph:

1. Union every known restricted/private visibility and the repository's explicit
   word-boundary `private` title marker for the same typed node across all sources.
   One private claim dominates public or unknown claims.
2. Build the directed containment relation from `collection`, `ref`, and `path`
   evidence, independent of whether those relations are admitted as physical
   edges.
3. Propagate private status from every seed to all containment descendants until
   the set stops changing. Cycles are handled by a visited-set closure; no edge is
   deleted to manufacture a tree.
4. Remove all private nodes and incident policy-admitted edges before components
   and anchor eligibility are constructed. Local node/evidence artifacts retain
   their identities only to audit the exclusion.

Privacy does **not** propagate through `alias`, `shortcut`, or `cross_link`. Those relations may
point at a node already private—in which case the identity union removes that node
everywhere—but their occurrence does not make an otherwise public target inherit
the referring node's privacy.

Unknown visibility is never interpreted as public. V1 **retains** unknown-
visibility nodes in the local graph and components, but marks them
`unknown_visibility` and ineligible in the anchor manifest. It records their count
and sets `privacy_certified=false`, which also makes `graph_asset_ready=false`. Unknown
is not a private seed and does not scrub its descendants. This preserves observed
topology for diagnosis without licensing
a confirmatory run. Detailed counts live in `scrub_manifest.json`; the release
candidate exposes only coarse retained/edge/component/anchor counts and the
certification flag.

## 6. Local-only artifact contract

A successful run contains exactly these files:

- `sources.json`
- `nodes.jsonl`
- `visibility_evidence.jsonl`
- `exclusions.jsonl`
- `conflicts.jsonl`
- `edge_evidence.jsonl`
- `physical_edges.tsv`
- `adjacency.jsonl`
- `components.jsonl`
- `anchor_eligibility.jsonl`
- `scrub_manifest.json`
- `legacy_parity.json`
- `aggregate_release_candidate.json`
- `manifest.json`
- `LOCAL_ONLY_DO_NOT_PUBLISH`

`manifest.json` and every detailed artifact are strictly local-only. Only
`aggregate_release_candidate.json` is structurally allowlisted for possible
release, and even it has `publishable=false` and requires explicit human approval.
It contains aggregate counts, certification, and approval flags only—never node
IDs, titles, accounts, paths, URLs, embeddings, source locators, or samples. The
preparer does not publish it.

JSON is canonical UTF-8 with sorted object keys; JSONL has one canonical object
per newline. Records and adjacency neighbours use stable numeric `pt:` ordering.
`physical_edges.tsv` is a canonically sorted undirected unit-conductance list.
Artifacts are mode 0600 inside a mode-0700 run directory.

`manifest.json` content-records every installed artifact so `verify` can detect
local tampering. Its **scientific** `fingerprint_core` is narrower and contains:

- schema, compiler algorithm, and repository commit;
- an implementation/parser content hash over the exact preparer contract bytes;
- logical source IDs, kinds, content lengths, and SHA-256 hashes;
- the complete relation and privacy policy;
- content records for the authoritative scientific artifacts, explicitly
  excluding `legacy_parity.json`;
- canonical hashes of the retained study universe and selected largest component;
  and
- both `resource_ceiling_bytes` and `observed_contract_bytes`.

The two graph-population hashes are SHA-256 over canonical sorted `pt:` membership,
not paths or titles. They pin the population from which components and anchors are
derived even when two snapshots happen to have the same aggregate counts.

### Frozen numeric and byte contract

The preparer is a single-threaded integer graph compiler. It emits exact unit
conductance `1`, integer counts, IDs, and hashes; it does not invoke BLAS, insert
jitter, solve a linear system, or select a machine-dependent thread count. The
consumer contract records downstream `float64` as the only decision-bearing
numeric dtype. The later diffusion runner must still record its actual backend,
thread settings, conditioning, residuals, and memory independently under the
fidelity protocol.

The declared raw-source byte total is checked against the ceiling before parsing;
the exact post-compilation prepare-time byte gate is

    observed_contract_bytes
      = total declared raw-source content bytes
        + total staged authoritative scientific artifact payload bytes
        (excluding `legacy_parity.json`),

where the diagnostic legacy payload, manifest, and marker bytes are excluded to avoid
circularity and preserve legacy noninterference. Both the
observed value and `--resource-ceiling-bytes` are recorded. Preparation fails
before installation when the observed value exceeds the ceiling. This is a
deterministic input/output byte contract—not peak RSS and not the projected
float64 dense-solve working set. Downstream resource arms remain governed by
`PROTOCOL_bounded_diffusion_fidelity.md`.

The snapshot fingerprint is SHA-256 of the canonical scientific core. Source and
member paths, filenames, mtimes, inodes, user/host names, temporary names,
minimum-anchor threshold, descriptive snapshot label, and legacy parity are
excluded. The repository commit is included; `snapshot_label_hash` remains
manifest provenance, while minimum-anchor coverage and `graph_asset_ready` are
manifest-level decisions. Byte-identical declarations and authoritative sources
under different local paths and the same commit produce the same graph artifacts
and scientific fingerprint.

## 7. Atomicity, verification, and failure semantics

`prepare` writes a mode-0700 sibling staging directory, writes and fsyncs mode-0600
artifacts, writes the marker and manifest, runs the self-contained verifier, fsyncs
the directory, and atomically installs it with Linux `renameat2(RENAME_NOREPLACE)`.
The sibling staging directory keeps the rename on one filesystem; a target that
appears concurrently is never replaced. An exception before promotion removes
staging output and never promotes a partial run. If the post-promotion parent
directory `fsync` fails, preparation reports failure but leaves the already
installed, independently verifiable run in place; recovery is `verify`/new-run,
never overwrite or rollback.

Preparation fails closed on malformed or changing sources, SQLite WAL state,
malformed or ambiguous `pt:` IDs, unsupported content/relation types, incomplete
policy, self relations, output/source overlap, an existing output, or a run outside
the approved local root. Verification fails closed on an unexpected run-file set,
marker mismatch, noncanonical JSON/JSONL, artifact content-record mismatch,
fingerprint mismatch, duplicate typed nodes or adjacency rows, excluded nodes in
adjacency, duplicate/unsorted neighbours, asymmetric adjacency, components that do
not partition retained adjacency nodes, or an eligible-anchor count mismatch.

`verify` validates the immutable compiled run; it does not promise to revalidate
raw sources that may later have moved. A future stronger verifier may replay the
compiler from the private source declaration, but v1's source mutation check is a
prepare-time transaction boundary.

`privacy_certified=false` or fewer than `--minimum-anchors` yields a valid
verified snapshot with `graph_asset_ready=false`; `prepare` returns exit 2. Exceeding
the resource byte ceiling is a hard prepare-time failure before installation.
Legacy parity never changes readiness. Integrity or operational failure returns
exit 1 with a generic fail-closed message that does not echo private values.
`verify` and `status` are read-only and expose only the fingerprint/readiness,
aggregate counts, and diagnostic parity status.

## 8. Legacy parity is causally disconnected

The optional `legacy_check.dag_path` is parsed only after the authoritative graph
has been compiled. `legacy_parity.json` compares the undirected legacy edge set
with the pre-privacy, policy-admitted authoritative edge set and records only
missing/extra counts plus the legacy content record. Legacy rows cannot create
nodes, relations, visibility, privacy seeds, physical edges, components, or
eligibility.

Legacy parity is a diagnostic sidecar, not scientific input. It is excluded from
the scientific artifact records and snapshot fingerprint, and match/mismatch does
not enter `graph_asset_ready`. Running with no legacy input, a matching input, or a
mismatching input must leave every authoritative graph artifact, population hash,
scientific fingerprint, and readiness decision identical. Only
`legacy_parity.json`, its tamper-detection bookkeeping, and the parity field exposed
by `status` may differ. A mismatch is evidence to investigate, never a repair
instruction or a reason to change the frozen graph.

## 9. Rejected alternatives

- **Legacy `assembled_dag.tsv` as truth:** rejected because it collapses relation
  provenance and carries neither visibility nor privacy-propagation semantics.
- **Title/slug inference:** rejected because titles are nonunique, mutable,
  privacy-sensitive, and can join unrelated nodes. Only typed source IDs identify.
- **Unknown visibility means public:** rejected because absence of evidence is not
  public evidence. V1 retains it only for local diagnosis and fails certification/readiness.
- **Propagate privacy through aliases/shortcuts/cross-links:** rejected because
  these are references, not containment. It can scrub unrelated public subtrees;
  direct private identity evidence is already handled by the conservative union.
- **Break cycles or choose one parent:** rejected because cycles are valid evidence
  convergence for HOP and privacy closure. Deleting an edge changes both the graph
  and scrub result. A later directed SKELETON phase may SCC-condense explicitly.
- **Use path or mtime as identity/freshness:** rejected because moving identical
  inputs would change the result and “newest” is environment-dependent. Content
  hashes and declared logical names define the snapshot.
- **Repair missing relations from titles, embeddings, or an LLM:** rejected because
  it mixes inference into raw evidence and can leak outcomes into geometry.
- **Publish detailed manifests automatically:** rejected because hashes do not make
  node-level metadata harmless. Release remains an explicit, aggregate-only human
  decision.

## 10. What this unlocks—and what it does not

A verified, privacy-certified snapshot with `graph_asset_ready=true` is only a
candidate for the graph-input gate in protocol §2. A second independent compilation must reproduce the snapshot fingerprint, and
that matching rerun hash must be frozen in the downstream run manifest. The
protocol must still freeze anchor batches, protected sets, resources, backend,
and downstream manifests before any real solve. The preparer
does not compute diffusion, inspect placement labels or judge outcomes, generate
embeddings, claim full-account coverage, or publish corpus artifacts. Until those
separate gates pass, the honest result remains: **no real-corpus outcome yet**.
