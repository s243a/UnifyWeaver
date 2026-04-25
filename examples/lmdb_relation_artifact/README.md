# LMDB Relation Artifact Prototype

Small Rust prototype for exact two-column relation artifacts backed by LMDB.

Current scope:

- build an LMDB artifact from a two-column TSV file
- emit a small `manifest.json` beside the LMDB environment
- support exact `arg1` lookup
- support full scan
- support `dupsort` for one-to-many relations

This is intentionally a prototype, not a committed cross-target artifact
format. The purpose is to validate that LMDB works as a local exact-artifact
backend in the current Termux environment and to give the shared
preprocess/manifest work a concrete non-C# backend to target.

## Usage

Build an artifact:

```sh
cargo run -- build edge/2 ./edge.tsv ./edge_artifact
```

Build a dupsort artifact:

```sh
cargo run -- build category_parent/2 ./category_parent.tsv ./category_parent_artifact --dupsort
```

Lookup one key:

```sh
cargo run -- get ./edge_artifact edge/2 a
```

Scan all rows:

```sh
cargo run -- scan ./edge_artifact edge/2
```

## Manifest

The prototype writes `manifest.json` with:

- predicate
- resolved mode
- artifact kind
- format version
- physical format
- exactness
- source path and SHA-256
- schema hash
- db name
- dupsort flag
- key/value encoding
- supported access contracts
- target capabilities
- declaration metadata
  - source
  - mode
  - kind
  - format
  - access contracts
  - options
- expected LMDB files

This is deliberately closer to the shared preprocess/artifact metadata seam than
the first smoke prototype, even though the declaration block is still
prototype-owned rather than coming from a real `preprocess/2` declaration.
