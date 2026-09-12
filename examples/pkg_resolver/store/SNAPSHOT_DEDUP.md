<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (@s243a) -->

# Memory-efficient multi-snapshot store (dedup)

Loading many repository snapshots (snapshot.debian.org style) makes a store
large and realistic, but naively giving each snapshot its own catalog
duplicates every *unchanged* package — O(snapshots × packages). Since a
published `(Name, Ver)`'s dependencies/conflicts/provides never change, and
most packages don't change between snapshots, the bulk of that data is
redundant. This store shares it.

## Schema

| table | key | scope |
|---|---|---|
| `store_pkg` | `SnapId\|Name → Ver` | **per-snapshot** (membership: which versions a snapshot has) |
| `pool_dep` | `PoolId\|Name → Ver#Dep#Constraint` | **pooled** (stored once, shared across snapshots) |
| `pool_conflict` | `PoolId\|Name → Ver#Other` | pooled |
| `pool_provides` | `PoolId\|Virtual → Pkg#Ver#VirtualVer` | pooled |
| `pool_revdep` | `PoolId\|Name → Dependent#Ver#Constraint` | pooled |

`PoolId` names a family of snapshots sharing one package namespace; a single
store can host several unrelated pools side by side, exactly as `CatId|Name`
lets several catalogs share one index today. The payloads are **byte-identical**
to the frozen `resolver_store.pl` tables — only the key's leading id changes —
so the D43 seek codec (`uw_fact_codec.js`) indexes them unchanged.

## Query — `resolver_store_snapshot.pl`

A new module (the frozen `resolver.pl`/`resolver_store.pl` are **never**
touched). It carries an 8-ary `env_snap(SnapId, PoolId, Base, Installed,
Requested, Layers, Excluded, Aliases)`, splits the frozen `store_key/3` into
`store_key_snap` (SnapId, for `store_pkg`) and `store_key_pool` (PoolId, for
the four `pool_*`), and ports the control flow with `store_dep→pool_dep` etc.
`collect_deps`'s `unpack_dep(...), Ver0==Ver` filter is untouched: pooling
only changes which key selects the candidate rows; the version filter was
always a post-unpack check on the payload, so a snapshot still sees exactly
its selected version's deps even though the pool holds every version's.
`dependents_snap/3` adds the one genuinely new step — a
`package_in_store_snap` membership guard on each pooled-revdep hit — so a
dependent from another snapshot never leaks.

## Build & query

```bash
# Build a deduped store (100 snapshots, 5k base, default churn):
bash examples/pkg_resolver/store/build_dedup_store.sh DIR --base=5000 --n=100

# Query snapshot t against it:
STORE_DIR=DIR/snap-t UW_POOL_DIR=DIR/pool UW_SNAP_ID=snapt UW_POOL_ID=s5k-snap \
  swipl -q -g main -t halt examples/pkg_resolver/store_diff_runner_snap.pl < cases.jsonl
```

`gen_snapshots.mjs --materialize=<t>` projects one snapshot back to the classic
flat per-catalog `rich.jsonl` (the naive O(N×pkg) form) for compatibility
testing against the frozen lane.

## Correctness

`resolver_store_snapshot.pl` is validated against the **frozen**
`resolver_store.pl` by differential, byte-for-byte on the result stream:
- **N=1** (`PoolId == SnapId == CatId`, store isomorphic to a single catalog):
  0 divergences on the 51-scenario corpus and the 503-case 5k differential —
  proves the port is faithful.
- **N>1** (deduped store vs a materialized snapshot): 0 divergences across all
  6+ query kinds including `dependents` (exercising the membership guard) —
  proves the dedup scheme is answer-preserving.

## Dedup ratio

`DedupRatio(pool) = N / (1 + c·N) → 1/c` (N snapshots, churn rate `c`). Measured
at `N=50, base=2000, 2% churn`: unique versions **4,450 vs naive 107,387 (24×)**;
bulk pooled tables **~1.2 MB vs ~30 MB (24×)**; whole store **7.3 MB vs 37.6 MB
(5.1×)**. The per-snapshot `store_pkg` membership table is not deduped by this
design and becomes the size driver at large N — a **delta-chain membership
encoding** (store only each snapshot's change vs the previous) is the natural
next step and would push the overall ratio toward the 24× bulk ratio.

## Scope

This is the **store** (data layer + single-snapshot query). The cross-snapshot
"newest version compatible with the locked set across N snapshots" **resolver**
(a driver over `resolve_layered_snap` enumerating candidates over the union of
versions) is a follow-on that consumes this store, as is the C++ WAM compile of
`resolver_store_snapshot.pl` for the memory×scale crossover benchmark on a large
pooled store.
