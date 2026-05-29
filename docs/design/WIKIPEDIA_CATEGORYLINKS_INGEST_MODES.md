# Wikipedia categorylinks ingest modes

**Status**: Implemented in `src/unifyweaver/runtime/rust/mysql_stream/src/categorylinks_resolve.rs`. See `lmdb_sink.rs` for the wiring.

**Date**: 2026-05-28
**Companion docs**:
- [`WAM_LMDB_RESIDENT_INTERNING_PHILOSOPHY.md`](WAM_LMDB_RESIDENT_INTERNING_PHILOSOPHY.md) — monotonic intern strategy used by mode=fallback/position.
- [`WAM_LMDB_RESIDENT_INTERNING_SPECIFICATION.md`](WAM_LMDB_RESIDENT_INTERNING_SPECIFICATION.md) — sub-db layout, `next_id` allocation rules.

## 1. The problem

MediaWiki refactored the `categorylinks` table around 2023. The
pre-2023 schema stored the parent category's **title** directly:

```
categorylinks: cl_from, cl_to (varbinary), cl_sortkey, ..., cl_type
```

`cl_to` was the parent's title string — a human-readable identifier
already in the same "space" as you could resolve through `page` to a
`page_id`.

The post-2023 schema replaced `cl_to` with an indirect reference into
a new `linktarget` table:

```
categorylinks: cl_from, cl_sortkey, cl_timestamp, cl_sortkey_prefix,
               cl_collation, cl_type, cl_target_id

linktarget:    lt_id, lt_namespace, lt_title
```

`cl_target_id` is a `linktarget_id`, **not a `page_id`**. The child
side (`cl_from`) is still a `page_id`. Multi-hop graph traversal on
just the categorylinks dump now mixes two ID namespaces — hop 2 needs
to look up the parent's `page_id` but only the parent's `linktarget_id`
is available.

## 2. Three ingest modes

The Rust ingester (`mysql_stream_lmdb`) now supports three modes via
the `--mode` flag, each with different correctness/dump-count trade-offs.

### 2.1 `--mode correct` (default, 3 dumps)

Required inputs:
- `categorylinks.sql.gz` (the edge dump)
- `--linktarget-dump linktarget.sql.gz`
- `--page-dump page.sql.gz`

Pipeline:

```
linktarget.sql.gz  →  HashMap<lt_id, title>          (ns=14 only)
page.sql.gz        →  HashMap<title, page_id>        (ns=14 only)
categorylinks.sql.gz, per subcat row (cl_from, cl_target_id):
    title = linktarget_map[cl_target_id]      // lt_id → title
    parent_page_id = page_map[title]          // title → page_id
    emit (cl_from, parent_page_id)
```

Output: every emitted edge is `(page_id, page_id)`. The graph is
fully walkable — given any node, you can look up its parents and
recurse without leaving the `page_id` namespace.

Cost: peak memory ≈ `O(|categories|)` for the two maps. For enwiki
this is ~50 MB for `lt_title_map` and ~30 MB for `page_title_to_id`
(roughly, depending on title length distribution). Both maps live
only for the duration of the categorylinks scan, then are dropped.

### 2.2 `--mode compromise` (2 dumps)

Required inputs:
- `categorylinks.sql.gz`
- `--linktarget-dump linktarget.sql.gz`

Pipeline:

```
linktarget.sql.gz  →  HashMap<lt_id, title>      (used only to gate edges)
categorylinks.sql.gz, per subcat row:
    if cl_target_id ∈ linktarget_map:
        emit (cl_from, cl_target_id)         // mixed namespaces, leaves opaque
    else:
        skip (count as unresolved)
```

Output: edges are `(page_id, lt_id)`. Walkable in the parent
direction only — given a parent `lt_id`, you can find ITS parents
(its lt_id will appear as a `cl_target_id` somewhere with the same
gate logic). Given a child `page_id`, you cannot recurse into the
child because its page_id doesn't appear as a `cl_target_id` unless
that page itself has subcategories — in which case its `lt_id`
(unknown to us) is what would appear, not its `page_id`.

The linktarget gate filters out spurious target_ids that don't
resolve to a category in the linktarget table. Without this gate,
post-2023 dumps would emit edges pointing at unresolvable identifiers.

### 2.3 `--mode fallback` (1 dump)

Required inputs:
- `categorylinks.sql.gz` only

Pipeline: per subcat row, transform `(cl_from, cl_target_id)`
according to `--id-method`.

This mode is for users who already have a pre-2023 dump (where the
ambiguity didn't exist) or who don't care whether the graph is
semantically correct, only that it has the right shape for
benchmark purposes.

## 3. ID-generation methods (fallback only)

`--id-method` selects how to handle the namespace mismatch in
single-dump ingest. Three strategies:

### 3.1 `raw` (preserve upstream IDs)

```
emit (cl_from, cl_target_id) unchanged
```

The simplest and most faithful — exactly what the pre-refactor
code path did. On a **pre-2023** dump where both columns were in
the page_id namespace, this produces a walkable graph. On a
**post-2023** dump, it produces a graph with mixed namespaces and
multi-hop traversal is broken at hop 2.

Use when: you control the source dump and know it pre-dates the
schema change, OR you only do single-hop queries.

### 3.2 `position` (monotonic intern, default)

```
let synthetic_id(raw) = intern.entry(raw).or_insert_with(|| {
    let id = next_id;
    next_id += 1;
    id
});
emit (synthetic_id(cl_from), synthetic_id(cl_target_id))
```

Every distinct raw integer (regardless of source column) gets the
next sequential ID. Both namespaces collapse into one synthetic
space. The graph is **structurally walkable**: if the same
underlying category appears as both `cl_from` (via some subcat
relationship) and `cl_target_id` (because something has it as a
parent), the two raw integers will be different — so the intern
table treats them as two distinct nodes. The graph will have
"orphan" nodes (a child that doesn't connect to its true parent
because the namespaces never met), but at least the IDs are
consistent.

Stable WITHIN a single run only. The synthetic ID assignment
depends on the order rows appear in the dump, which depends on
MySQL's clustered index order, which depends on `cl_from` ordering
— so across two runs of the same dump, IDs match. But across
different dumps (e.g. last-month's enwiki vs this-month's), the
IDs drift as new rows are added.

Use when: you want graph structure for benchmarks, don't care
about cross-run stability, and don't have the linktarget dump.

### 3.3 `hash` (deterministic hash, cross-run stable)

```
emit (hash64(0xA1, cl_from), hash64(0xB2, cl_target_id))
```

Each side is hashed with a different salt byte so the namespaces
stay separate (no accidental collision between `page_id=5` and
`lt_id=5`). FNV-1a is used — fast, well-distributed for integers,
no external dependencies.

Stable ACROSS runs because the hash is a pure function. Cross-run
identity is preserved as long as the upstream IDs don't change.

Collision rate: `n² / 2^64` for `n` ingested IDs (birthday
paradox). At `n = 10^6` this is ~2.7×10⁻⁸ expected collisions
within a column — negligible. At `n = 10^9` it climbs to ~0.027,
still acceptable for graph-structure benchmarks.

Use when: you want stable IDs across dump revisions but can't
get the linktarget dump.

### 3.4 Why not "use page_id directly when present"

A tempting fourth method would be to keep `cl_from` as-is (it's a
real `page_id`) and only synthesize for `cl_target_id`. But this
guarantees the namespaces don't meet — every parent gets a
different synthetic ID than the page_id its children would carry.
The resulting graph is exactly as broken as `raw`, just with one
side renumbered. Not added.

## 4. Manifest fields

The generated `.manifest.json` now records both:

```json
{
  ...
  "IngestMode": "correct" | "compromise" | "fallback",
  "IdMethod": "raw" | "position" | "hash"
}
```

Downstream consumers (Haskell, F#, C# query runtime) can read these
to know whether multi-hop traversal will work and whether IDs are
expected to be stable across runs.

## 5. What's NOT in this design

- **Disk-spilling intern table**: the `position` method holds the
  full `HashMap<i64, i64>` in memory. For enwiki at ~10M distinct
  raw IDs this is ~160 MB — fine on commodity hardware. If we
  later target multi-billion-edge dumps this would need to spill
  to an LMDB sub-db. Not motivated yet.
- **Bloom filter on hash collisions**: the `hash` method could
  detect collisions by also storing raw→hash in a bounded Bloom
  filter and flagging hash reuse. Useful for very large graphs
  but adds memory cost. Not motivated yet.
- **Schema auto-detection from `CREATE TABLE`**: the parser could
  read the `CREATE TABLE categorylinks` statement at the top of
  the dump and pick column indices accordingly. Current code
  assumes the post-2023 column layout. If pre-2023 dumps need
  proper support, add a `--schema-version` flag rather than
  auto-detect (more predictable for users).

## 6. References

- Implementation: `src/unifyweaver/runtime/rust/mysql_stream/src/categorylinks_resolve.rs`
- Wiring: `src/unifyweaver/runtime/rust/mysql_stream/src/lmdb_sink.rs`
- Existing intern philosophy: `docs/design/WAM_LMDB_RESIDENT_INTERNING_PHILOSOPHY.md`
- MediaWiki schema docs for linktarget refactor: see https://www.mediawiki.org/wiki/Manual:Linktarget_table
