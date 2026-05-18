# WAM-Rust LMDB crate decision: configurable, lmdb-zero default

**Status**: decided 2026-05-18. Both lmdb-zero and heed supported via codegen option; lmdb-zero is the default.

## TL;DR

The Rust LMDB arc supports **both** lmdb-zero and heed, selected via a Prolog codegen option `lmdb_crate(lmdb_zero | heed | auto)`. The `auto` resolver currently picks `lmdb_zero`. Both implementations are empirically validated against the 1k_cats fixture and produce identical counts.

## Decision path

| Date | Position | Why |
| --- | --- | --- |
| 2026-05-17 | lmdb-zero only (initial R1) | "Thinner over the C API, hypothetically faster" |
| 2026-05-18 morning | Considered heed-only swap | Concern about lmdb-zero parallelism safety at scale; thought heed had cleaner ergonomics |
| 2026-05-18 — first heed scratch | Heed fails in parallel | heed's `RoTxn::drop` aborts the txn, invalidating dbi handles. Single-threaded works (matches lmdb-zero counts: 3098/3098/5933/5933); parallel `env.read_txn()` from worker threads returns EINVAL. heed is NOT a drop-in replacement. |
| 2026-05-18 — second heed scratch | Heed works with WriteTxn bootstrap | A committed `WriteTxn` persists dbi handles across subsequent read txns. Required for the parallel pattern, but doable: 4 worker threads each opened own RoTxn and got correct 5933 counts. |
| 2026-05-18 final | Configurable, lmdb-zero default | User noted "we can't measure it if we don't have both options configurable" — building only one forecloses the comparison. Configurability also lowers the bar to adding *other* LMDB bindings in the future (e.g., when a new crate emerges). |

## Why lmdb-zero is the default

| Property | lmdb-zero | heed |
| --- | --- | --- |
| Read-only access | works directly with `Arc<Environment>` + per-thread RoTxn + parent-thread sub-db open + `Arc<Database>` | requires a brief `WriteTxn` at startup to persist dbi handles; env can't be opened with `EnvFlags::READ_ONLY` |
| Concurrent reads | each thread opens own `ReadTransaction`; no dbi-lifetime constraint | each thread opens own `RoTxn`; works only after the bootstrap WriteTxn commits |
| Read-only filesystem support | yes (no write lock taken if env is opened RDONLY) | no (WriteTxn requires write access to the env) |
| Empirical validation against 1k_cats | passes (4 threads × correct counts, 5933 entries) | passes (4 threads × correct counts, 5933 entries, with WriteTxn bootstrap) |
| Codebase footprint | ~150 lines of template | similar |
| Maintenance | older (last release ~2020), thinner | active, broader ecosystem |
| Type-state for txn/cursor safety | runtime panic | compile-time |
| `MDB_NOTLS` / `maxreaders` hygiene | manual | better defaults |

The deciding factor for the default: **lmdb-zero works on read-only fixtures without requiring write access to the LMDB env**. Our use case (pre-ingested category data, never modified at runtime) is more naturally expressed by RoTxn-only access. heed requires a write-txn dance at startup that touches the lock file even when no real writes happen.

heed's safety advantages (type-state, MDB_NOTLS hygiene) remain real and may matter at higher parallelism or for write-heavy workloads. The `lmdb_crate(heed)` option exists precisely so we can switch easily and measure when those advantages become relevant.

## Codegen option

```prolog
lmdb_crate(lmdb_zero)  % explicit lmdb-zero
lmdb_crate(heed)       % explicit heed
lmdb_crate(auto)       % auto-resolve via resolve_auto_lmdb_crate/2
```

`auto` currently resolves to `lmdb_zero` unconditionally. Future heuristics (target core count, fixture write-permissions, measured speed) would slot into `resolve_auto_lmdb_crate/2` following the existing resolver pattern from `resolve_auto_cache_strategy/2` and friends.

The codegen dispatches the option to:

- **Cargo.toml**: conditional `lmdb-zero = "0.4"` or `heed = "0.20"`
- **src/lmdb_fact_source.rs**: rendered from one of two mustache templates
  - `templates/targets/rust_wam/lmdb_fact_source_lmdb_zero.rs.mustache`
  - `templates/targets/rust_wam/lmdb_fact_source_heed.rs.mustache`
- **lib.rs**: `pub mod lmdb_fact_source;` (same line either way)

The Rust API surface is the same across both implementations (same struct name, same method signatures: `open`, `load_s2i`, `load_i2s`, `lookup_parents`, `lookup_children`) so callers in R2 / R4 don't see the difference.

## Why configurability beats "swap when measurements justify"

The user's observation that flipped this: **we can't measure what we don't have**. Picking heed-only or lmdb-zero-only forecloses the empirical comparison that would later justify the choice. Configurability inverts the cost: ~2 hours up-front to scaffold the option vs. potentially never collecting the data that would justify a change.

Secondary value: the scaffolding makes adding a *third* LMDB crate (or any other future binding) trivially small — the option pattern already exists and any new crate just needs its own template + a clause in the resolver.

## Empirical baseline

Both implementations open the same 1k_cats fixture (`data/benchmark/1k/lmdb_resident`, 5933 edges, 3098 atoms) and produce identical counts:

| Sub-DB | lmdb-zero (4-thread parallel) | heed (4-thread parallel, WriteTxn bootstrap) |
| --- | ---: | ---: |
| s2i | 3098 | 3098 |
| i2s | 3098 | 3098 |
| category_parent | 5933 × 4 threads | 5933 × 4 threads |
| category_child | 5933 | 5933 |

Bench-level head-to-head (read throughput, RSS, parallel scaling) is deferred to after R2 / R4 land — at that point we can measure both crates end-to-end and refine `resolve_auto_lmdb_crate/2` if one wins meaningfully at some scale.

## Cross-references

- Last lmdb-zero-only R1 commit: `21b65f31` on `feat/wam-rust-lmdb`
- Codegen option pattern this follows: `resolve_auto_cache_strategy/2`, `resolve_auto_lmdb_cache_mode/2`, `resolve_auto_demand_bfs_mode/2` (all in `src/unifyweaver/core/cost_model.pl`)
- Handoff doc: `context/wam_rust_lmdb_handoff.md` (gitignored)
- Haskell B1 PR equivalent (parallel pattern): `docs/design/WAM_PERF_OPTIMIZATION_LOG.md` Phase 2b.3
