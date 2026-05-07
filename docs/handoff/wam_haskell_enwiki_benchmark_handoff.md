# WAM-Haskell Enwiki Benchmark Handoff

## Context

Recent merged work ported one WAM-Elixir lesson back to WAM-Haskell:

- `8975364f` merged PR #1876.
- `cb57ca71 bench(wam-haskell): gate demand-filtered seeds`.
- The generated Haskell effective-distance benchmark now skips seeds outside
  the root-bound structural demand set before constructing WAM state or calling
  the kernel.
- It also replaced quadratic seed de-duplication and added
  `demand_skipped_seeds`.

Merged-main verification after that PR:

| scale | mode | time | hash | notes |
|---|---:|---:|---:|---|
| `50k_cats` | Haskell `-N1` | `3.422s` | `9ada853b4403` | `demand_skipped_seeds=49989` |
| `50k_cats` | Haskell `-N2` | `3.459s` | `9ada853b4403` | no useful speedup |
| `50k_cats` | Haskell `-N4` | `5.326s` | `9ada853b4403` | worse |
| `100k_cats` | Haskell `-N1` | `9.657s` | `9ada853b4403` | `demand_skipped_seeds=84125` |
| `100k_cats` | Haskell `-N2` | `6.254s` | `9ada853b4403` | useful speedup |
| `100k_cats` | Haskell `-N4` | `7.021s` | `9ada853b4403` | worse than `-N2` |
| `50k_cats` | Rust accumulated | `0.446s` | `9ada853b4403` | still much faster |
| `100k_cats` | Rust accumulated | `0.599s` | `9ada853b4403` | still much faster |
| `50k_cats` | Elixir LMDB int IDs | `1.260s` | `9ada853b4403` | faster than Haskell |
| `100k_cats` | Elixir LMDB int IDs | `1.643s` | `9ada853b4403` | faster than Haskell |

The important uncertainty: Haskell used to scale more linearly with cores before
the demand-gate and setup optimizations. It is not clear yet whether the lack
of scaling at 50k/100k is a real limitation or simply because the benchmark has
too little remaining query work after most seeds are skipped.

## Existing Full-Enwiki Assets

There is already a full-enwiki LMDB artifact. Do not regenerate a TSV fixture
without first checking this path:

```text
/home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_cats/lmdb_proj/lmdb
```

It contains `data.mdb` and `lock.mdb`. Existing companion files:

```text
/home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_cats/lmdb_proj/seed_ids.txt      # 1,000 seeds
/home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_cats/lmdb_proj/root_ids.txt      # 1 root
/home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_cats/lmdb_proj_10k/seed_ids.txt  # 10,000 seeds
/home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_cats/lmdb_proj_10k/root_ids.txt  # 1 root
```

There is also:

```text
/home/s243a/Projects/UnifyWeaver/data/wikipedia_categories.db
```

I started to inspect this DB, but that was probably the wrong direction because
the LMDB artifact already exists.

## Current WAM-Haskell Enwiki Path

The current WAM-Haskell generator for this family is:

```text
examples/benchmark/generate_wam_haskell_enwiki_lmdb_benchmark.pl
```

It generates a project that:

- uses `use_lmdb(true)`;
- uses `lmdb_layout(dupsort)`;
- uses `int_atom_seeds(true)`;
- expects `seed_ids.txt`, `root_ids.txt`, and `lmdb/` in the runtime facts dir;
- skips WAM compilation for `category_parent/2` through the LMDB
  `external_source` path.

Critical point: it currently sets:

```prolog
demand_filter(false)
```

The comment says demand filtering relies on `parentsIndexInterned`, which is
empty in `int_atom_seeds` mode because edges live in LMDB. That means the recent
Haskell demand-gate optimization probably does not apply to the full-enwiki
LMDB/int-ID path.

## Old Standalone Probe To Treat Carefully

There is also:

```text
examples/benchmark/enwiki_dfs_benchmark/
```

This is an older standalone Haskell DFS probe. It compares IntMap vs LMDB raw
against the same LMDB graph, but it uses `Database.LMDB.Raw` and pointer-level
LMDB calls. I briefly ran it, then stopped because it may conflict with the
current target direction away from raw pointer handling.

Observed from that old probe before stopping:

- `lmdb` backend at 10k seeds, `-N1`: `0.0616s` measured DFS time.
- `lmdb` backend at 1M seeds, `-N1`: `4.610s` measured DFS time after a temporary
  sampler fix.
- Raw `lmdb` backend crashed at `-N2` with an LMDB cursor assertion, likely
  because the raw cursor backend is not thread-safe.
- `lmdb_cached` backend hit `MDB_READERS_FULL` under `-N2`.

Do not use those results as evidence for the current WAM-Haskell target without
first deciding that the old raw-pointer benchmark is still an acceptable probe.

## Benchmarking I Ran

### Merged Current WAM-Haskell Effective-Distance Harness

These runs used the current merged WAM-Haskell target via:

```sh
python3 examples/benchmark/benchmark_effective_distance.py \
  --scales 50k_cats \
  --repetitions 1 \
  --targets haskell-wam-accumulated \
  --build-root /tmp/uw-main-haskell-verify
```

and:

```sh
HASKELL_RTS="+RTS -N2 -RTS" python3 examples/benchmark/benchmark_effective_distance.py \
  --scales 50k_cats,100k_cats \
  --repetitions 1 \
  --targets haskell-wam-accumulated \
  --build-root /tmp/uw-main-haskell-n2
```

plus the same command with `HASKELL_RTS="+RTS -N4 -RTS"` and
`--build-root /tmp/uw-main-haskell-n4`.

Results:

| scale | RTS | time | rows | hash | query metric | skipped seeds |
|---|---:|---:|---:|---:|---:|---:|
| `50k_cats` | default / `-N1` | `3.422s` | 11 | `9ada853b4403` | `query_ms=141` | `49989` |
| `50k_cats` | `-N2` | `3.459s` | 11 | `9ada853b4403` | `query_ms=113` | `49989` |
| `50k_cats` | `-N4` | `5.326s` | 11 | `9ada853b4403` | `query_ms=179` | `49989` |
| `100k_cats` | default / `-N1` | `9.657s` | 11 | `9ada853b4403` | `query_ms=240` | `84125` |
| `100k_cats` | `-N2` | `6.254s` | 11 | `9ada853b4403` | `query_ms=172` | `84125` |
| `100k_cats` | `-N4` | `7.021s` | 11 | `9ada853b4403` | `query_ms=220` | `84125` |

Interpretation:

- `-N2` helps at `100k_cats`.
- `-N4` is worse than `-N2`.
- `50k_cats` has too little remaining query work after demand gating for
  parallelism to help.
- These are authoritative for the current WAM-Haskell harness, but they are
  not full-enwiki runs.

### Rust And Elixir Comparison On The Same 50k/100k Fixtures

Command:

```sh
python3 examples/benchmark/benchmark_effective_distance.py \
  --scales 50k_cats,100k_cats \
  --repetitions 1 \
  --targets wam-rust-accumulated,wam-elixir-int-tuple,wam-elixir-lmdb-int-ids \
  --build-root /tmp/uw-main-rust-elixir-compare
```

Results:

| scale | target | time | rows | hash |
|---|---:|---:|---:|---:|
| `50k_cats` | `wam-rust-accumulated` | `0.446s` | 11 | `9ada853b4403` |
| `50k_cats` | `wam-elixir-int-tuple` | `2.061s` | 11 | `9ada853b4403` |
| `50k_cats` | `wam-elixir-lmdb-int-ids` | `1.260s` | 11 | `9ada853b4403` |
| `100k_cats` | `wam-rust-accumulated` | `0.599s` | 11 | `9ada853b4403` |
| `100k_cats` | `wam-elixir-int-tuple` | `2.307s` | 11 | `9ada853b4403` |
| `100k_cats` | `wam-elixir-lmdb-int-ids` | `1.643s` | 11 | `9ada853b4403` |

Interpretation:

- Rust is still fastest on these fixtures.
- Elixir LMDB int IDs beat Haskell on these fixtures after the Haskell demand
  gate, mostly because Haskell still has setup cost that is not query work.

### Old Standalone Enwiki DFS Probe

These runs used:

```sh
cd examples/benchmark/enwiki_dfs_benchmark
cabal v2-build
./dist-newstyle/build/x86_64-linux/ghc-8.6.5/enwiki-dfs-benchmark-0.1.0.0/x/enwiki-dfs/build/enwiki-dfs/enwiki-dfs \
  /home/s243a/Projects/UnifyWeaver/data/benchmark/enwiki_cats/lmdb_proj/lmdb \
  lmdb 10000 +RTS -N1 -RTS
```

10k result:

```text
backend=lmdb
seeds=10000
total_elapsed_sec=6.162695e-2
per_seed_mean_ms=6.162695e-3
seeds_per_sec=162266.67066924454
total_nodes_visited=50163
avg_nodes_per_seed=5.0163
max_depth_observed=4
```

I then attempted 1M seeds with the same old probe. The first run was stopped
because the benchmark's reservoir sampler used linked-list replacement and was
spending time in sampling rather than DFS. I made a temporary local branch
change to use `IntMap` in the sampler, then reverted it because this old probe
uses raw pointer LMDB calls and should not be mixed into the current target
narrative.

Temporary 1M observations from the old probe:

```text
backend=lmdb
RTS=-N1
seeds=1000000
total_elapsed_sec=4.61041658
per_seed_mean_ms=4.610416579999999e-3
seeds_per_sec=216900.1396398761
total_nodes_visited=5025034
avg_nodes_per_seed=5.025034
max_depth_observed=5
```

Failed old-probe multi-core attempts:

```text
backend=lmdb
RTS=-N2
mdb.c:6140: Assertion 'IS_LEAF(mp)' failed in mdb_cursor_set()
```

```text
backend=lmdb_cached
RTS=-N2
MDB_READERS_FULL: Environment maxreaders limit reached
```

Interpretation:

- These old-probe results should be treated as diagnostic only.
- The old `lmdb` backend is not thread-safe.
- The old `lmdb_cached` backend was not immediately usable under `-N2`.
- Most importantly, this code path uses `Database.LMDB.Raw` and pointer-level
  APIs, so it should not be presented as the current WAM-Haskell LMDB target.

## Questions For Claude

1. What is the canonical current benchmark for full-enwiki WAM-Haskell
   effective-distance after the recent demand-gate PR?

2. Should the next run use
   `generate_wam_haskell_enwiki_lmdb_benchmark.pl`, and if so, what is the
   correct facts dir: `lmdb_proj`, `lmdb_proj_10k`, or a new companion dir with
   many more precomputed `seed_ids.txt` entries?

3. Is `demand_filter(false)` still required in `int_atom_seeds` mode, or should
   Haskell learn to compute/root-bound prepopulate demand data from LMDB without
   materializing all facts into the generated program?

4. Is there already a preprocessing command that generates large
   `seed_ids.txt` files from the existing enwiki LMDB? If yes, use that instead
   of sampling keys in Haskell or generating TSV fixtures.

5. For a fair multi-core test, should we benchmark:
   - current WAM-Haskell LMDB with `lmdb_cache_mode(per_hec)`;
   - `lmdb_cache_mode(sharded)`;
   - `lmdb_cache_mode(two_level)`;
   - no cache;
   - or all of the above?

6. Are Rust and Elixir expected to have comparable full-enwiki LMDB/int-ID
   benchmark paths, or should the immediate comparison stay Haskell-only until
   key encodings and fixture shape are aligned?

## Suggested Next Step

Start by asking the codebase what existing command produced
`data/benchmark/enwiki_cats/lmdb_proj` and `lmdb_proj_10k`. If a generator for a
larger seed companion directory already exists, use it. If not, add a small
preprocessing utility for seed/root companion files that reuses the existing
LMDB artifact and does not touch raw-pointer benchmark code.

Then run the current WAM-Haskell enwiki LMDB project at a larger seed count with
`HASKELL_RTS` set explicitly:

```sh
HASKELL_RTS="+RTS -N1 -RTS" ...
HASKELL_RTS="+RTS -N2 -RTS" ...
HASKELL_RTS="+RTS -N4 -RTS" ...
```

Record whether the result is WAM-Haskell current target code or the old
standalone DFS probe. That distinction matters for the PR narrative.
