# Pearltrees diffusion source declaration and compilation consensus

**Status:** prospective gate between the snapshot compiler and the bounded
diffusion fidelity study. This gate declares the raw inputs explicitly and asks
the same frozen compiler to reproduce one graph in two fresh processes. It does
not run diffusion, select anchors, calibrate leakage, inspect filing labels, or
claim that two executions are two independent graph observations. Detailed
specifications, snapshots, and receipts remain local-only.

This document complements
[`DESIGN_pearltrees_diffusion_snapshot.md`](DESIGN_pearltrees_diffusion_snapshot.md)
and satisfies the deterministic-recompilation prerequisite in
[`PROTOCOL_bounded_diffusion_fidelity.md` Section 2](PROTOCOL_bounded_diffusion_fidelity.md#2-frozen-graph-and-conductance-regimes).

## 1. Why there are two stages

The snapshot compiler deliberately has no input discovery. The operator first
creates one explicit source declaration, then independently executes the frozen
compiler twice:

```text
human-reviewed explicit sources + relation policy
  -> immutable local source specification
  -> fresh compiler process A -> verify A
  -> fresh compiler process B -> verify B
  -> exact comparison -> immutable consensus receipt
```

Separating declaration from compilation prevents a second run from silently
finding a newer export, a different API cache, or a filename-dependent account.
Separating the executions into fresh processes tests deterministic repeatability
and source stability. It does **not** protect against a parser defect shared by
both executions, and it does not justify a two-sample uncertainty estimate.

## 2. Explicit source declaration

`declare_pearltrees_diffusion_sources.py` accepts only repeated, explicitly
named source arguments. Each argument supplies a stable `source_id` and path;
RDF additionally supplies its account. Supported kinds exactly match the frozen
snapshot compiler: RDF, API SQLite, an API JSON directory, and path JSONL. A
legacy assembled DAG may be named only as the optional parity diagnostic.

```text
declare_pearltrees_diffusion_sources.py \
  --snapshot-label LABEL --local-root LOCAL_ROOT \
  --output-dir NEW_DECLARATION_DIR --local-only \
  [--rdf SOURCE_ID ACCOUNT PATH]... \
  [--api-json-dir SOURCE_ID PATH]... \
  [--api-sqlite SOURCE_ID PATH]... \
  [--path-jsonl SOURCE_ID PATH]... \
  [--legacy-dag PATH]
```

The declaration tool must never:

- glob directories, choose the newest file, follow symlinks, or use an
  environment-dependent default;
- infer an RDF account from a filename or path;
- harvest, repair, synthesize, or parse source content; or
- treat the legacy DAG as an authoritative source.

The canonical team-space RDF account is the literal `groups`, including for a
historical source whose filename contains the misspelling `grous`. The account
is a human-reviewed declaration, not a filename repair. Source IDs are unique,
nonempty provenance labels; their ordering does not change the canonical
specification.

The tool checks only the declared filesystem type needed by the downstream
adapter. Content validation and hashing remain the compiler's job. It writes a
complete private declaration bundle into a new mode-0700 directory under an
explicitly approved local root. The bundle contains exactly mode-0600
`source_spec.json` (canonical `pearltrees-diffusion-source-spec-v1` JSON) and a
mode-0600 `LOCAL_ONLY_DO_NOT_PUBLISH` marker with fixed bytes. It installs
without replacement; the directory must be outside Git and no component may be
a symlink. The local specification necessarily contains private source paths;
standard output therefore reports only source-kind counts and success, never
paths, IDs, labels, titles, or content hashes.

Consensus accepts only this complete installed bundle, not an arbitrary
schema-shaped JSON file. Before either child process it reruns the declaration
validator over the directory/file modes, exact inventory and marker, canonical
JSON, supported source kinds, explicit RDF accounts, canonical ordering,
resolved absolute non-symlink paths of the declared types, unique IDs, and
path/inode non-aliasing. The validator is repeated after each attempt, and its
own content record is bound in the receipt.

The relation policy is intentionally not generated here. It is a scientific
choice and remains a separate, complete, human-reviewed input to the compiler.
An installed declaration proves only an explicit deterministic inventory. It
does not prove that the inventory is complete, current, authentic, or exhaustive
of the user's account.

## 3. Independent execution contract

`prepare_pearltrees_diffusion_consensus.py` receives the immutable source
specification, relation policy, approved local root, two new attempt directories,
a new receipt directory, minimum-anchor threshold, and resource ceiling. It
performs exactly these operations:

```text
prepare_pearltrees_diffusion_consensus.py prepare \
  --source-spec SPEC --relation-policy POLICY \
  --attempt-a-dir NEW_A --attempt-b-dir NEW_B --receipt-dir NEW_RECEIPT \
  --local-root LOCAL_ROOT --local-only --minimum-anchors N \
  --resource-ceiling-bytes N

prepare_pearltrees_diffusion_consensus.py verify \
  --receipt-dir RECEIPT --attempt-a-dir A --attempt-b-dir B \
  --source-spec SPEC --relation-policy POLICY
```

1. verify the complete private declaration bundle, then reject path aliases,
   symlinks, existing targets, in-repository outputs, and overlapping attempt
   or receipt directories;
2. invoke `prepare_pearltrees_diffusion_snapshot.py prepare` in a fresh process
   for attempt A through the current Python executable in isolated mode,
   accepting exit 0 or the documented scientific exit 2;
3. invoke the compiler's independent `verify` command in another process;
4. repeat the same pair of commands for attempt B without sharing artifacts;
5. compare the verified manifests exactly on the consensus contract; and
6. atomically install one local-only receipt and verify it against the frozen
   inputs and both installed attempt directories. There is no third execution,
   two-of-three rescue, or pooling of artifacts.

Compiler exit 2 is a valid fail-closed scientific result: a verified snapshot
may be privacy- or coverage-blocked. Operational failure, invalid output, or a
failed verification is not consensus evidence and exits 1. A successfully
written receipt exits 0 only when both snapshots are ready and exactly agree;
an honest blocked receipt exits 2.

Attempt A is the deterministic canonical downstream snapshot. Attempt B is
repeatability evidence only. Artifacts are never pooled, averaged, or counted as
two graphs. The compiler path is the fixed sibling implementation; an arbitrary
compiler executable or `PATH` lookup is not allowed. Device/inode identities of
the local root and every output parent are bound before execution and rechecked
around child invocations and receipt installation, so a replaced parent fails
closed.

## 4. Exact agreement contract

Both snapshots must independently verify. Consensus then requires equality of:

- the exact source-specification and relation-policy content records observed
  before attempt A and after attempt B;
- the verified manifests after removing only the causally disconnected legacy
  parity artifact record and aggregate status, while still requiring
  `snapshot_label_hash`;
- `snapshot_fingerprint` and the entire `fingerprint_core`;
- scientific artifact records and their authoritative set hash (legacy parity
  is excluded from the scientific fingerprint but its diagnostic status remains
  recorded separately);
- source-content and implementation records, repository commit, numeric
  contract, privacy policy, and relation policy;
- study-universe and largest-component hashes;
- resource ceiling and observed byte contract;
- aggregate counts, certification fields, and population hashes; and
- the separately supplied minimum-anchor threshold, coverage decision,
  `privacy_certified`, and `graph_asset_ready`.

Exact fingerprint equality already binds most of these fields, but not the
source-specification bytes, snapshot label, minimum-anchor threshold, or legacy
parity. In particular, the v1 fingerprint records whether a source declared an
account, not the declared account text itself. The exact specification content
record therefore binds the human-reviewed RDF account declaration. The other
explicit comparisons are retained as a defensive schema check and make
readiness settings outside the fingerprint core impossible to overlook. Any
scientific or readiness disagreement blocks; the gate does not select the more
favorable run.

Legacy parity remains causally disconnected. Its two artifact records and
statuses are recorded and compared, but a legacy-only difference is a warning,
not a graph-gate failure. It cannot make either a blocked graph ready or a ready
graph blocked.

## 5. Receipt and downstream chain

The receipt is canonical JSON under a local-only marker and contains no paths,
node IDs, titles, URLs, source IDs, or source labels. It records:

- schema and algorithm identifiers;
- exact source-specification and relation-policy content records;
- the declaration validator, compiler entry point, and consensus-orchestrator
  implementation records, plus the compiler-bound repository commit;
- non-path runtime identity: Python implementation/version, SQLite runtime
  version, Expat version, and Unicode-data version;
- attempt labels A and B with each verified manifest's SHA-256, snapshot
  fingerprint, observed byte contract, readiness fields, process exit codes,
  legacy artifact record, and legacy status;
- the common snapshot fingerprint and authoritative scientific artifact-set
  hash, fingerprint-core hash, study-universe hash, largest-component hash,
  compiler implementation records, numeric contract, and repository commit when
  equal;
- minimum anchors, resource ceiling, aggregate readiness/certification state,
  and legacy-parity comparison/warning state;
- an exact list of comparison gates and their Boolean results;
- separate `repeatability_verified` and `graph_gate_pass` decisions, `accepted`,
  one enumerated reason code, and the canonical-attempt label; and
- enough canonical content for the downstream manifest to bind the receipt's
  externally computed content record.

Receipt verification is deliberately not receipt-only. The verifier requires
the receipt directory, source specification, relation policy, and both attempt
directories. It revalidates the complete declaration bundle, binds the actual
input bytes, reruns the fixed snapshot verifier on both attempts, reloads both
actual manifests, and re-derives every per-attempt record, comparison, common
field, warning, and acceptance decision. It refuses extra fields, changed
implementation/runtime identity, invalid modes, or a substituted artifact. The
receipt's unkeyed self-hash detects accidental/internal mutation; it is an
integrity aid, not proof of authenticity.

The HOP study must run that full verification, then freeze the receipt hash and
the canonical attempt-A manifest hash in its no-solve planning manifest. A
path-free receipt alone cannot prove that a separately stored input or attempt
directory has not been replaced.

The intended immutable chain is:

```text
source specification + relation policy
  -> attempt A manifest + attempt B manifest
  -> snapshot consensus receipt
  -> no-solve HOP planning manifest
  -> calibration lock manifest
  -> untouched audit result
```

Changing any bound input or record, or rebuilding from changed raw sources,
policy, compiler commit, readiness threshold, resource ceiling, or receipt,
invalidates every downstream link. Verification does not rehash raw files after
their prepare-time transaction; calibration and audit may never silently rebuild
the graph.

## 6. Claims and non-claims

An accepted receipt supports only this claim: on the recorded host runtime, the
frozen compiler produced the same verified, privacy-certified, coverage-ready
snapshot twice from the same explicit frozen inputs under the same resource and
readiness contract.

It does not establish parser correctness, corpus completeness, graph-physics
validity, HOP convergence, selector quality, filing performance, covariance
deployment, or CUDA value. An accepted receipt authorizes construction of the
outcome-blind no-solve HOP plan; it does not by itself authorize calibration or
audit solves. A blocked receipt is a complete scientific gate result and must
not be repaired by changing inputs after inspecting which graph would pass.

## 7. Rejected alternatives

- **Automatic inventory discovery:** rejected because repeated runs could bind
  different raw evidence while appearing deterministic.
- **Inferring accounts from export names:** rejected because filenames are not
  source evidence and the observed `grous` spelling is ambiguous.
- **Two implementations as the required repeatability check:** rejected for this
  gate because the snapshot fingerprint intentionally binds the frozen compiler
  implementation. An orthogonal ledger validator is useful additional defense,
  but is not a substitute for exact same-code reproduction.
- **One process calling the compiler function twice:** rejected because module
  state and shared resources weaken the fresh-execution claim.
- **Two-of-three voting:** rejected because a nondeterministic compiler or moving
  source is itself a failed prerequisite.
- **Publishing detailed receipts:** rejected because hashes do not make private
  corpus provenance automatically safe to disclose.
