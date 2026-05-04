# WAM-Elixir LMDB int-id FactSource

**Status**: design proposal + emit-and-grep code stub. Runtime
validation deferred until `:elmdb` is installable.

**Companion to**: PR #1792 (initial `WamRuntime.FactSource.Lmdb`
adaptor) and PR #1815 (int-tuple FactSource pattern, the single
largest perf lever in the cross-target benchmark sequence).

## What this proposal addresses

PR #1815 found that for the WAM-Elixir CategoryAncestor kernel,
storing the FactSource as a tuple-as-array indexed by contiguous
integer IDs is ~2× faster than an atom-keyed `Map` at 10k scale.
The win came from `elem/2` being O(1) without hashing.

That recipe assumed an upstream **interning pass** — at TSV-load
time the bench harness builds a string→int map and then a tuple
of dest-lists indexed by the int ID. The cross-target benchmark
doc documents this as "the recommended scale-up recipe."

But: at production scale (~1M unique categories — the full
Wikipedia data) two things break:

1. The intern pass + tuple build is in-memory only. No
   persistence; restart the BEAM and you re-intern from scratch.
2. The int-tuple holds the entire fact set in BEAM's heap.
   For 1M categories with ~10M edges, that's ~160-1600 MB,
   uncomfortable on small machines.

LMDB is the existing answer to (2) for the other targets: a
memory-mapped K/V store that lets the OS page hot data while keeping
the cold majority on disk. The PR #1792 Elixir LMDB adaptor stores
binary keys (UTF-8 category names) and binary values. That solves
(2) but loses the integer-key win from PR #1815: every kernel
neighbor lookup pays the binary-string `Map.get` hash cost.

The earlier kernel docstring claimed Haskell already solved this by
using LMDB record IDs directly as comparison keys. Investigation
found that's **incorrect**. Haskell's intern table
(`atom_intern_id/2`, `wam_haskell_target.pl` ~line 75) is built at
codegen, the integer IDs come from emission order, and LMDB itself
stores binary keys. Scala's `InternTable` runs at runtime per
FactSource lookup. Neither pushes interning into the storage layer.
The architecture this proposal describes is genuinely new for the
LMDB-using targets.

## Goal

Produce an LMDB FactSource adaptor for WAM-Elixir that:

- Stores facts as integer-id pairs (so neighbor lookups return
  bare integers, ready for the int-tuple kernel path).
- Persists the string↔id mapping in the same LMDB env, surviving
  process restart.
- Is consistent across multiple readers / cross-process (LMDB
  is multi-reader-safe).
- Scales to arbitrary fact-set size — only hot pages live in RAM,
  cold pages stay on disk.
- Reuses the safe Elmdb API surface that PR #1792's existing
  `WamRuntime.FactSource.Lmdb` already targets.

## Architecture

Three LMDB sub-databases inside one env:

| Sub-DB | Key | Value | Notes |
|---|---|---|---|
| `facts_dbi` | 8-byte BE u64 (arg1 id) | 8-byte BE u64 (arg2 id) | `MDB_DUPSORT` if a single arg1 maps to multiple arg2s. Both key and value are integer IDs. |
| `key_to_id_dbi` | binary (UTF-8 string) | 8-byte BE u64 | Input translation: caller has a node name, gets the ID. Unique. |
| `id_to_key_dbi` | 8-byte BE u64 | binary | Output translation: caller has an ID, wants the original string for display. Unique. |

A sentinel key `"__next_id__"` in `id_to_key_dbi` (or in a separate
metadata sub-DB) tracks the next unassigned ID. The ingestion
routine reads it, assigns IDs sequentially, writes back.

ID assignment is **insert-time** (driver responsibility). The
ingestion routine takes raw `(arg1_str, arg2_str)` pairs and
populates all three sub-databases consistently. This is more
efficient than open-time enumeration because:

- Open is now O(1) — just opening LMDB handles. No cursor walk to
  rebuild the id map.
- Ingestion already needs to walk the input data once anyway, so
  building the maps adds no extra pass.

For migration of an existing string-keyed LMDB (PR #1792's shape)
to int-id form, a one-time `migrate_to_int_ids/1` helper would
cursor-walk the existing fact DB and emit the three sub-DBs.

## Public API surface

```elixir
defmodule WamRuntime.FactSource.LmdbIntIds do
  @behaviour WamRuntime.FactSource

  defstruct [:env, :facts_dbi, :key_to_id_dbi, :id_to_key_dbi, :arity, :dupsort]

  # Open the FactSource against a pre-opened LMDB env + the three
  # sub-DBs. Driver is responsible for creating the env and DBs and
  # populating them via insert-time ID assignment.
  @impl true
  def open(spec, pred_arity, state)

  # Returns [{key_id :: integer, value_id :: integer}, ...].
  # The fast path — bypasses both string keys AND the existing
  # PR #1792 `lookup_by_arg1` adapter. Pair this with the int-tuple
  # kernel path: `dests_fn = fn id -> for {_, v_id} <- lookup_by_arg1_id(...), do: v_id end`.
  def lookup_by_arg1_id(handle, key_id, state) when is_integer(key_id)

  # Backwards-compatible binary-key entry. Translates key → id,
  # delegates to lookup_by_arg1_id, translates value ids → strings
  # on the way out. Two extra LMDB reads per lookup, useful for
  # gradual migration but slower than the int-id path.
  @impl true
  def lookup_by_arg1(handle, key, state) when is_binary(key)

  # Boundary translators. Caller usually invokes these once at the
  # query boundary (input: parse user-supplied article name to id;
  # output: format result ids back to display names).
  def lookup_id(handle, key) :: integer | nil
  def lookup_key(handle, id) :: binary | nil

  # Cursor-walk all (key_id, value_id) pairs in facts_dbi.
  @impl true
  def stream_all(handle, state)

  @impl true
  def close(handle, state)
end
```

The kernel-dispatch path uses `lookup_by_arg1_id`. The string-key
entry exists only for callers that haven't migrated to int IDs yet.

## Comparison with existing FactSource adapters

| FactSource | Storage | Key type | Cap | Scale-up |
|---|---|---|---|---|
| `Tsv` | in-memory list-of-tuples | binary | RAM | not for production |
| `Ets` | ETS bag | binary | RAM | up to ~RAM/2 |
| `Sqlite` | sqlite3 file | binary | disk | scales but slow |
| `Lmdb` (PR #1792) | LMDB env | binary | disk | scales, but pays binary-key cost on every kernel lookup |
| **`LmdbIntIds`** | LMDB env (3 sub-DBs) | **integer** | disk | **scales AND keeps the PR #1815 int-tuple perf** |

## What the code stub ships

A new module `WamRuntime.FactSource.LmdbIntIds` emitted by
`wam_elixir_target.pl` with:

- The full public API surface above.
- Bodies that call `Module.concat([Elmdb])` for the actual LMDB
  primitives (`txn_get`, `ro_txn_cursor_get`, etc.) — same indirect
  resolution pattern as the existing `Lmdb` adaptor, so the runtime
  emits and compiles without `:elmdb` installed.
- Big-endian 8-byte u64 encoding helpers (`encode_id/1`, `decode_id/1`)
  for the integer-key marshalling.

## What still needs runtime validation

Without `:elmdb` we can't actually exercise:
- The three-sub-DB ingestion path.
- The cursor walk in `lookup_by_arg1_id` for `MDB_DUPSORT`.
- The id↔key translation latency vs the binary-key adapter.
- End-to-end perf parity with the in-memory int-tuple recipe.

The emit-and-grep tests in
`tests/test_wam_elixir_target.pl` validate the structural shape:
the module emits, the API functions are present, the moduledoc
documents the architecture. Same test discipline as PR #1792 used
for the original Lmdb adaptor.

## Open questions for runtime validation

1. **Cost of integer encoding/decoding per lookup.** Each
   `lookup_by_arg1_id` produces N `<<id::64-big-unsigned>>` decodes
   for the value list. Maybe negligible, maybe dominant — only
   measurement will tell.

2. **Whether dupsort comparator order matches integer order.** LMDB
   uses memcmp by default; for big-endian u64 that *should* match
   integer order but we need to verify (especially for cursor-based
   range scans).

3. **Per-transaction vs per-call txn lifecycle.** The PR #1792
   adapter opens a fresh ro_txn per `lookup_by_arg1` call. For the
   kernel walk that does many lookups in tight succession, batching
   into a single txn (e.g., a `with_txn/2` wrapper) could matter.

4. **Whether LMDB's page cache + memcmp on 8-byte keys actually
   matches the in-memory int-tuple `elem/2` win at scale.** This is
   the big empirical question. The hypothesis is yes (memcmp on a
   fixed-width 8-byte key + a B+tree lookup ≈ HAMT lookup, but with
   disk-backing for free); the proof requires measurement.

## Status

- [x] Doc fix (the inaccurate "Haskell uses LMDB IDs" claim
  removed from kernel docstring + benchmark doc + roadmap)
  — PR #1819.
- [x] Design proposal (this doc) — PR #1819.
- [x] Code stub: `WamRuntime.FactSource.LmdbIntIds` emitted by
  `wam_elixir_target.pl`. Compiles, doesn't run without `:elmdb`
  — PR #1819.
- [x] Emit-and-grep tests asserting the API surface emits — PR #1819.
- [x] **Driver ingestion helper**: `ingest_pairs/3` populates all
  three sub-databases consistently with insert-time ID assignment.
  Idempotent for previously-seen strings. Returns `{:ok, %{pairs_seen,
  new_ids, next_id}}` for batched ingestion. — this PR.
- [x] **Migration helper**: `migrate_from_string_keyed/3` cursor-walks
  an existing PR #1792 `Lmdb` env, batches into `ingest_pairs/3` calls,
  populates the destination's three sub-databases with sequential
  IDs in encounter order. — this PR.
- [ ] `:elmdb`-backed integration test (requires Hex.pm reachability).
- [ ] Cross-target benchmark with the int-id LMDB FactSource against
  the in-memory int-tuple recipe (requires the integration test).
- [ ] Mock-`Elmdb` end-to-end test that exercises the adaptor's logic
  (id encoding/decoding, dupsort cursor walk, round-trip) without
  requiring real `:elmdb`. ~300 lines of Elixir for a fake module
  backed by an in-memory map; would catch the kinds of bugs
  emit-and-grep can't (encoding off-by-ones, dupsort comparator
  ordering, txn lifecycle). Deferred.

## Driver-side recipe

With the helpers shipped, a driver looks like:

```elixir
# 1. Open env + three sub-DBs (driver responsibility).
{:ok, env} = :elmdb.env_open("/path/to/db.lmdb", maxdbs: 3, ...)
{:ok, facts}    = :elmdb.db_open(env, "facts",    [:create, :dupsort])
{:ok, k_to_id}  = :elmdb.db_open(env, "key_to_id",  [:create])
{:ok, id_to_k}  = :elmdb.db_open(env, "id_to_key",  [:create])

handle = WamRuntime.FactSource.LmdbIntIds.open(
  %{env: env, facts_dbi: facts, key_to_id_dbi: k_to_id,
    id_to_key_dbi: id_to_k, arity: 2, dupsort: true},
  2, nil)

# 2. Ingest data (insert-time ID assignment).
{:ok, %{next_id: nid}} =
  WamRuntime.FactSource.LmdbIntIds.ingest_pairs(
    handle,
    [{"alpha", "beta"}, {"alpha", "gamma"}, {"delta", "epsilon"}],
    start_id: 0)

# 3. Persist next_id somewhere (sentinel key, separate metadata DB,
#    application config) for the next ingest batch.

# 4. Register with the kernel-dispatch FactSourceRegistry.
WamRuntime.FactSourceRegistry.register("category_parent/2", handle)

# 5. Now the kernel sees integer IDs end-to-end:
#    dispatch_wrapper.run/1 calls handle.lookup_by_arg1_id/3 which
#    returns [{cat_id, parent_id}, ...] -- no atom-table pressure,
#    no per-call hashing, fact data on disk via LMDB.
```

For migration from an existing PR #1792 string-keyed env:

```elixir
{:ok, %{pairs_migrated: n, ids_assigned: m}} =
  WamRuntime.FactSource.LmdbIntIds.migrate_from_string_keyed(
    old_string_handle, new_int_id_handle,
    start_id: 0, batch_size: 10_000)
```
