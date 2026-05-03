# LMDB benchmark recipe - Scala / Haskell / Elixir cross-target

This document explains how to run the effective-distance benchmark
with an LMDB-backed `category_parent/2` against the three WAM
targets that ship an LMDB FactSource adaptor: Scala (PR #1804),
Haskell (`generate_wam_haskell_enwiki_lmdb_benchmark.pl`), and Elixir
(PR #1792).

The point is to compare materialisation cost: above ~100k facts the
JVM/BEAM/GHC heap pressure of holding all rows in memory dominates
runtime. LMDB's memory-mapped file lets each target read straight
from the page cache.

> **Status:** infrastructure-only. The three targets use *different
> key encodings* on disk - Haskell uses int32 IDs, Scala/Elixir use
> UTF-8 strings. Reconciling those is out of scope for this PR; see
> *Caveats* at the bottom. The recipe below produces three separate
> envs and three separate benchmarks. Treat the timings as
> indicative, not apples-to-apples, until the encoding is unified.

## Workload

`effective_distance.pl` driving `category_ancestor(Cat, Root, Hops,
Visited)` over Wikipedia category hierarchy facts. The bottleneck
predicate is `category_parent(Child, Parent)`: the WAM body queries
it with `Child` ground (single-key probe) and the multi-parent shape
naturally maps to LMDB `MDB_DUPSORT`.

## Setup

You need the LMDB env populated before any of the three targets can
read it. The simplest loader is a small program that reads the
`category_parent/2` facts from the workload `.pl` and writes them as
`Child -> Parent` rows under `MDB_DUPSORT`.

There isn't a single canonical loader in this repo; each target's
existing benchmark has its own. For a single-string-key env that
both Scala and Elixir can read, a one-liner with the `mdb_dump`
companion tooling works; for the int32-key env the Haskell pipeline
expects, see `examples/benchmark/generate_wam_haskell_enwiki_lmdb_benchmark.pl`'s
docstring (it documents `seed_ids.txt` / `root_ids.txt`
companion files).

## Running each target

### Scala (this PR)

```sh
# 1. Generate the project with lmdb data mode.
WAM_SCALA_LMDB_ENV=/path/to/lmdb_env \
  swipl -q -s examples/benchmark/generate_wam_scala_effective_distance_benchmark.pl -- \
    data/benchmark/dev/facts.pl /tmp/scala-bench-lmdb \
    accumulated kernels_on lmdb

# 2. Build with lmdbjava on the classpath.
cd /tmp/scala-bench-lmdb
sbt 'set libraryDependencies += "org.lmdbjava" % "lmdbjava" % "0.9.0"' compile

# 3. Run with the same classpath.
sbt 'run --bench 100 wam_effective_distance_q/4 ...'
```

### Haskell

See `examples/benchmark/generate_wam_haskell_enwiki_lmdb_benchmark.pl`.
The Haskell pipeline targets int32 IDs and the LMDB layout matches
that - `seed_ids.txt` + `root_ids.txt` companion files are required.

### Elixir

See PR #1792 (`feat/wam-elixir-lmdb-fact-source`). Same
single-string-key shape as Scala. Driver responsibility: open env +
dbi via `:elmdb`, populate, pass handles to
`WamRuntime.FactSource.Lmdb.open/3` via the spec map.

## Comparing the timings

Each target's bench mode emits a `BENCH n=<N> elapsed=<sec>` line.
Capture three runs of equal `N` against the same workload and
compare. Beware: the encoding mismatch means each target is reading
*its own* LMDB, not the same bytes - so the comparison is bound by
loader cost, not just runtime cost. A fair side-by-side requires the
encoding reconciliation in *Caveats*.

## Caveats

- **Key encoding mismatch.** Scala/Elixir use UTF-8 strings; Haskell
  uses int32 IDs. To make all three read the *same* env you'd
  either: (a) teach Scala/Elixir an int32-key mode, or (b) teach
  Haskell a string-key mode. Either is a separate PR.
- **MDB_DUPSORT byte order.** Lmdbjava and `:elmdb` both expose the
  `MDB_DUPSORT` flag, but their default key/value comparator
  functions may differ from the C `lmdb` library Haskell uses
  through `lmdb-bindings`. Verify before drawing conclusions about
  per-key fanout.
- **Cache modes.** The Haskell pipeline has multiple `lmdb_cache_mode`
  options (`memoize` / `per_hec` / `sharded`); Scala/Elixir don't yet.
  Disable Haskell caching (or pick the no-cache default) for an
  apples-to-apples comparison.

## See also

- `PR_DESCRIPTION_WAM_SCALA_LMDB_FACT_SOURCE.md` - the Scala
  adaptor's design contract.
- `tests/test_wam_scala_lmdb_runtime_smoke.pl` - runtime smoke for
  the Scala adaptor (gated on `SCALA_LMDB_TESTS=1` and
  `LMDBJAVA_CLASSPATH`).
- `docs/WAM_TARGET_ROADMAP.md` - the original materialisation-cost
  bottleneck writeup that motivated all three LMDB adaptors.
