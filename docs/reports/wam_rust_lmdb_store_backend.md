# wam_rust STORE lane — real LMDB backend, opt-in parity (D108)

**Question.** The Rust pkg-resolver STORE lane's `lmdb(Dir)` backend
(`SeekFactSource::new_lmdb`) was a fail-loud STUB: declaring
`UW_STORE_BACKEND=lmdb` compiled, but any query on an lmdb-declared source
panicked with `lmdb_seek_missing_error` — no reader was built in. This report
wires a real reader in, gated so it stays **non-default and opt-in**:
`indexed(Prefix)` (the dependency-free UWFI/UWIX seek store) remains the
default and is byte-unchanged; `lmdb(Dir)` becomes a real, byte-identical
PARITY tier that a caller must explicitly ask for (`UW_STORE_BACKEND=lmdb`).

**Framing — this is NOT a performance feature.** D107
(`docs/reports/wam_rust_store_scale_pressure_bench.md`) already proved there is
no wall-clock case for LMDB at the resolver's stores: they run at **~1
row/key**, so a keyed miss touches ~1 cold page on ANY backend (indexed,
clustered, or LMDB) and the LMDB speedup ceiling at rows-per-key=1 is **1.00×**
— confirmed here again, empirically, on synthetic drop_caches runs down to
122 MB. This work does not reopen that question or claim a speedup. The value
of a real LMDB reader is **maturity, crash-safety, and concurrent-reader
semantics** (a mature, ACID, MVCC B+-tree store vs. a bespoke binary-searched
flat file) — optionality for a deployment that already standardizes on LMDB
elsewhere, not throughput.

**Verdict: shipped.** Byte-identical to `indexed(Prefix)` on every gate run —
same 51/51 corpus, same 503/0 differential, `cmp`-clean on both raw output
files and the 5000-package B3 selection. Non-default: the default `indexed`
build carries zero LMDB dependency (no extra Cargo fetch, no `gcc` requirement,
identical binary to before this change). `resolver.pl` / `resolver_store.pl`
are unmodified.

---

## 1. Toolchain precondition (gating check, done first)

Per the task's STOP-if-it-can't-build precondition, this was verified before
writing the reader:

```
UW_LMDB_DATA_V1=1 bash examples/pkg_resolver/store/ensure_lmdb.sh   # -> uw_ensure_lmdb OK
```

This container has `gcc`, `g++`, `make`, `node` v22, `npm` 10, and a working
`HTTPS_PROXY` for the npm registry — the `LMDB_DATA_V1=true npm install
--build-from-source lmdb` rebuild (D43 policy: OpenLDAP-0.9.29-lineage,
vanilla-liblmdb-compatible page format, as opposed to the default `lmdb`
npm prebuilt, which is a Symas fork whose page format vanilla liblmdb rejects
with `MDB_INVALID`) succeeded, installing into `/tmp/uw-lmdb-pkg-v1`.
`build_lmdb_stores.sh` then built real per-store LMDB environments
(`examples/pkg_resolver/store/.out/corpus/lmdb/{pkg,dep,conflict,revdep,provide}/`)
from the corpus P/2 JSONL dumps, and a small hand-rolled `lmdb-zero` probe
crate confirmed those environments open under a from-source `liblmdb-sys`
build with no `MDB_INVALID` — i.e. the toolchain builds cleanly here, and the
work proceeded past the gating precondition.

## 2. Crate choice: lmdb-zero

`docs/design/WAM_RUST_LMDB_CRATE_DECISION.md` (written for the graph/edge
LMDB subsystem, `lmdb_fact_source_{lmdb_zero,heed}.rs.mustache`) already
settled this: **lmdb-zero is the project default**, specifically because it
"works on read-only fixtures without requiring write access to the LMDB env"
— heed needs a bootstrap `WriteTxn` at startup even for a pure read workload.
The STORE lane's `lmdb(Dir)` sources are exactly that read-only case (facts
never change; the store path is baked in at generation time), so this report
follows the existing default rather than re-litigating it:

- **lmdb-zero** (`Environment::open` with `RDONLY | NOTLS`, no bootstrap
  write transaction, no dbi-lifetime dance).
- Its dependency `liblmdb-sys` **vendors and compiles the OpenLDAP `mdb.c` /
  `midl.c` sources itself** via the `cc`/`gcc` crate at `cargo build` time —
  **no system `liblmdb` package is needed**, and critically, it is the SAME
  OpenLDAP lineage the `UW_LMDB_DATA_V1=1` npm rebuild targets, so the two
  sides of this reader (writer: `uw_fact_lmdb.js` via node's from-source
  `lmdb` package; reader: this Rust crate via `liblmdb-sys`) are page-format
  compatible by construction, not by luck. This was confirmed empirically: a
  standalone `lmdb-zero` probe crate opened the corpus LMDB stores built by
  `UW_LMDB_DATA_V1=1 build_lmdb_stores.sh` with no format error.

heed was not evaluated as an alternative for this reader — the existing
decision record's reasoning (read-only-friendly, no write-txn bootstrap)
applies directly and there was no new information to revisit it.

## 3. Store key/value schema mirrored

`scripts/js_wam/uw_fact_lmdb.js` writes each store as a single (default,
unnamed) LMDB database inside a directory environment, with **two** keyspaces
per row (`uw_fact_codec.js`):

```
seq (unbound enum, source order): 0x00 || u64be(seq)                        -> payload
a1  (bound lookup):                0x01 || u16be(key_len) || encodeIndexKey(a1) || u64be(seq) -> payload
```

`payload` is `recordPayload`: `u16le(a1_len) || u16le(a2_len) || a1 bytes ||
a2 bytes` — no outer length-prefix framing (LMDB tracks value length itself).
`encodeIndexKey` is the same D34-tagged key encoding the indexed `.idx`
binary search already uses (`0x41` atom / `0x53` string / `0x49` int / `0x46`
float), matching `SeekFactSource::encode_store_key` byte for byte.

The new reader (`SeekFactSource::rows_lmdb_bound` / `rows_lmdb_scan` in
`templates/targets/rust_wam/seek_fact_source.rs.mustache`) reads **only the
`0x01` a1-range keyspace** — every row is written under exactly one such key,
so it alone is sufficient for both query shapes; the `0x00` seq keyspace is
never read by this reader.

**Byte-identity is the actual engineering problem here, and it does not fall
out for free** — the two query shapes needed two different arguments:

- **Bound-key seek** (arg1 bound, the common case: `store_pkg(Name, Ver)`
  etc.): a cursor range `[a1_range_start(key), a1_range_end(key)]` (the same
  key/length, with the trailing `u64be(seq)` swept `0x00..0xFF`) visits
  exactly the rows for that key, in seq-ascending order. Since `seq` is the
  row's original position in the source JSONL, and `uw_fact_codec.js`'s
  `writeIndexedStore` clusters `.data` with a **stable** sort tie-broken by
  that same original order, the two per-key row orders coincide row for row
  — no re-sort needed. Verified directly against a real store: looking up
  `diamond|d` (which has two rows, `2.0.0` then `1.0.0` in source order)
  returns that exact order from both the `.idx` hit list and the LMDB range
  scan.
- **Unbound-arg1 scan** (the provides-style full walk): this is the part that
  is *not* a trivial "just use LMDB's own order" — the a1-range key's
  `u16be(key_len)` prefix comes **before** the key bytes, so LMDB's own
  B+-tree order sorts primarily by **key length**, not by the raw key bytes
  `bytes_compare` (and indexed's `#4272` clustering) use. A direct probe on
  the real corpus `pkg` store showed this concretely: walking the raw a1
  keyspace yields `linear|a` (9-byte key) before `diamond|a` (10-byte key)
  before `alts_first|a` (13-byte key) — length-grouped, not the
  `bytes_compare`-sorted order `indexed(Prefix)`'s `.data` file is in. The
  fix: `rows_lmdb_scan` walks the full a1 keyspace in whatever order LMDB
  gives it, then **stable-sorts by `bytes_compare` on the extracted key bytes
  alone** — exactly indexed's own clustering rule. Stability is what makes
  this safe: within one exact key's group, the raw LMDB traversal order is
  already seq-ascending (only the trailing seq differs among ties), so
  re-sorting by key alone leaves that intra-key order untouched. This was
  verified independently with a standalone probe: reconstructing the pkg
  store's 116-row scan this way and diffing against a direct parse of the
  real `pkg.data` UWFI file gave a byte-for-byte identical 116-row list.

`decode_payload` (the `u16/u16/bytes/bytes` -> `(Value, Value)` decode) is now
a single shared function used by **both** backends — the indexed `.data`
reader strips its own record's 4-byte length prefix first, then calls the
same `decode_payload` the LMDB reader calls directly on its LMDB value — so
the two backends cannot silently diverge on how bytes become a row.

## 4. Codegen wiring (opt-in, no dependency for `indexed`)

`seek_fact_source.rs` is emitted into **every** wam_rust-generated crate
unconditionally (term catalogs, other graph targets, not just the store
lane), most of which have nothing to do with LMDB. To keep `indexed` (and
every other lane) free of any new dependency:

- A new Cargo feature `store_lmdb` is declared in **every** generated
  crate's `[features]` table (`store_lmdb = []`, empty when unused) — the
  same "always declare, rarely enable" pattern already used for `parallel`.
  The real reader code in `seek_fact_source.rs` is behind
  `#[cfg(feature = "store_lmdb")]`; without it, `rows()` on an `lmdb(Dir)`
  source still panics loudly via `lmdb_seek_missing_error` (updated to name
  the feature), exactly as before this change.
- `wam_rust_target.pl` computes, per generated project, whether its
  `rust_wam_fact_sources` option declares any `source(_, lmdb(Dir))` entry
  (`build.pl`'s `store_sources/3` makes a build all-`indexed` or all-`lmdb`,
  never mixed, so this is a per-project fact). When true: `store_lmdb` is
  added to the crate's `default` feature list, **and** the Cargo.toml
  dependency block gains a real, non-optional `lmdb-zero = "0.4"` line
  (reusing the existing `use_lmdb_zero`-gated Cargo.toml block the graph/edge
  LMDB subsystem already has, via a separate boolean so the graph/edge
  `lmdb_mode(cursor)` option and this store-lane trigger stay independent —
  a plain `indexed`-backend or store-less project's Cargo.toml is untouched
  by this change and still declares no LMDB dependency of any kind).
- `rust_store/build.sh` and `run_differential_rust_store.sh`'s `lmdb` branch
  now `export UW_LMDB_DATA_V1=1` before sourcing `ensure_lmdb.sh` /
  `build_lmdb_stores.sh`, mirroring `cpp_store/build.sh` — without it, a
  plain `npm install lmdb` reuses the cached Symas-fork prebuilt and the
  resulting store is unreadable by vanilla `liblmdb-sys` (`MDB_INVALID`).

No changes were needed to `rust_store/build.pl`, `state.rs.mustache`'s
`register_lmdb_seek_fact2` / `execute_seek_fact_source`, or the JSON shim —
every caller already routed through `SeekFactSource::rows(key)` uniformly, so
implementing the lmdb-kind branch of `rows()` was the entire integration
surface.

## 5. Gate matrix (`LC_ALL=C.UTF-8`)

| Gate | indexed (default) | lmdb (opt-in) |
| --- | --- | --- |
| Store corpus (`run_corpus_rust_store.sh`) | 51/51 matched SWI, identical to term corpus | 51/51 matched SWI, identical to term corpus |
| Store differential (`run_differential_rust_store.sh`, 5k catalog) | 503 cases / 0 divergences / 0 crashes | 503 cases / 0 divergences / 0 crashes |
| Term corpus (`run_corpus_rust.sh`, regression) | 51/51 | n/a (term lane never uses store sources) |
| Term differential (`run_differential_rust.sh`, regression) | 2600 cases / 0 divergences / 0 crashes | n/a |
| `cargo test --lib` (store crate) | 232 passed | 234 passed (232 + 2 new `lmdb_reader_tests`) |
| `cargo test --lib` (term crate) | 232 passed (unaffected by this change) | n/a |
| Cargo.toml LMDB dependency | none | `lmdb-zero = "0.4"` (hard dep, this project only) |
| Fail-loud check (`lmdb(Dir)` declared, `store_lmdb` feature manually forced off) | n/a | panics with the updated `lmdb_seek_missing_error` message; never falls back to `indexed` |

**Byte-identity, `cmp`-clean:**

- `rust_store/.diff_out/rust.jsonl` (503-case differential output):
  `indexed` and `lmdb` runs are byte-for-byte identical.
- `rust_store/.corpus_out/rust_store.jsonl` (51-case corpus output):
  `indexed` and `lmdb` runs are byte-for-byte identical.
- 5000-package B3 selection (`uw_resolve_store --scale-probe`'s
  `rust_store_result` line, the bound `resolve_layered(p30, ...)` answer over
  the 5k catalog): identical between backends — only the D43 I/O counters
  differ (`rust_store_bytes_read`/`rust_store_n_reads`; the `lmdb` build
  currently reports these as 0 since the D43 byte-counting instrumentation is
  wired to the indexed `.data`/`.idx` file reads only — an informational gap,
  not a correctness one, and consistent with this work's framing that LMDB is
  not being measured or sold on I/O efficiency here).

The two build-time gotchas called out in the task were re-confirmed and
guarded against on every run:

1. **STORE_DIR path-baking** — the generated `uw_resolve_wam_store` crate
   bakes the absolute `STORE_DIR` into `setup_foreign_predicates` at
   `swipl`/`build.pl` time. Every indexed-vs-lmdb / corpus-vs-scale
   comparison in this report re-ran `build.sh` with the matching `STORE_DIR`
   and `UW_STORE_BACKEND` immediately before gating, per D106's recorded
   gotcha.
2. **`dump_store_data` clobber** — `rust_store/build.sh` only skips the
   corpus dump when `cases.jsonl` already exists in `STORE_DIR`; the corpus
   dir already carried one from earlier gating in this session, so this
   never fired destructively here, but every build in this report went
   through `build.sh`'s own guard rather than a hand-rolled shortcut.

## 6. Files changed

- `templates/targets/rust_wam/seek_fact_source.rs.mustache` — the real
  `lmdb(Dir)` reader (`ensure_open_lmdb`, `rows_lmdb` /
  `rows_lmdb_bound` / `rows_lmdb_scan`, `a1_range_start` / `a1_range_end` /
  `a1_range_key_bytes`), a shared `decode_payload` (factored out of
  `read_record`), and a new `#[cfg(feature = "store_lmdb")] mod
  lmdb_reader_tests` with two tests that build a real `lmdb-zero`
  environment and assert its `rows()` output is `assert_eq!`-identical to
  the existing hand-built indexed fixture, for both a bound-key seek and an
  unbound-arg1 scan.
- `src/unifyweaver/targets/wam_rust_target.pl` — `store_lmdb` Cargo feature
  wiring (declared in every crate's `[features]`, auto-defaulted only for a
  project whose `rust_wam_fact_sources` are `lmdb(Dir)`) and the matching
  `lmdb-zero` Cargo dependency line, independent of the pre-existing
  graph/edge `lmdb_mode(cursor)` option.
- `examples/pkg_resolver/rust_store/build.sh`,
  `examples/pkg_resolver/rust_store/run_differential_rust_store.sh` —
  `export UW_LMDB_DATA_V1=1` in the `lmdb` branch (vanilla-format store
  build), mirroring `cpp_store/build.sh`.
- `examples/pkg_resolver/rust/uw_resolve_wam/{Cargo.toml,src/seek_fact_source.rs}`
  — the committed term crate regenerated to pick up the shared template/
  Cargo-feature change (`store_lmdb = []` declared, never enabled — the term
  lane has no store-backed fact sources, so `StoreUsesLmdb` is false and no
  LMDB dependency is added). Every other regenerated file in that crate
  (`lib.rs`, `state.rs`, `value.rs`, `instructions.rs`, `main.rs`,
  `par_aggregate.rs`, `boundary_cache.rs`) reverted to its committed content
  after regeneration — those diffs were pure `// Date:` timestamp churn plus
  pre-existing WAM label-renumbering nondeterminism unrelated to this change,
  not confined to it, so they were not committed.
- `examples/pkg_resolver/rust_store/uw_resolve_wam_store/` — gitignored
  per-build artifact; not committed.

`resolver.pl` / `resolver_store.pl` are unmodified.

## 7. What this is not

Per D107's framing, this is explicitly **not** a performance change. The
rows-per-key=1 structure of every resolver store (pkg/dep/conflict/revdep/
provide) means a keyed miss touches ~1 cold page on `indexed`, clustered
`indexed`, or `lmdb` alike — there is no scatter for LMDB to remove here, and
this report makes no wall-clock claim. The reader exists so that a
deployment that wants LMDB's maturity (crash-safety via its ACID/MVCC
B+-tree design, concurrent-reader semantics, an on-disk format broadly used
elsewhere) has a real, byte-identical, opt-in way to get it, at the cost of
one extra `gcc`-compiled dependency it only pays when it asks for it.
