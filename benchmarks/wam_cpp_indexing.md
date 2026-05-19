# WAM-cpp first-argument indexing microbenchmark

This document captures the dispatch-cost findings for the indexing
work in PRs #2296, #2297, #2298, #2301, #2303 (consolidated A1/A2,
homogeneous + mixed-mode).

Reproduce with:

```
swipl -q -g run -t halt tests/benchmarks/bench_wam_cpp_indexing.pl
swipl -q -g "run([n_clauses(10),   n_iters(200000)])" -t halt \
      tests/benchmarks/bench_wam_cpp_indexing.pl
swipl -q -g "run([n_clauses(500),  n_iters(20000)])"  -t halt \
      tests/benchmarks/bench_wam_cpp_indexing.pl
```

## What's measured

Three predicate shapes, each with N clauses; the head-unification
work is identical between shapes, only the dispatch path differs:

| Shape   | Source                                 | Dispatch                              |
| ------- | -------------------------------------- | ------------------------------------- |
| `idx`   | `p(k1, v1). p(k2, v2). ...`            | `switch_on_constant` (homogeneous)    |
| `chain` | `p(K, V) :- K = k1, V = v1. ...`       | none — plain `try_me_else` walk       |
| `mma`   | `p(k1, v1). ... p(_, default_val).`    | `switch_on_constant_fallthrough` (#2301) |

`chain` is the pre-indexing baseline: every clause has a variable
A1 (and a variable A2), so neither A1 nor A2 indexing applies.

For each shape, three query positions:

- **first** — hits clause 1 (best case for everyone)
- **last**  — hits clause N (worst case for `chain`'s linear walk)
- **miss**  — matches no clause (best case for indexed, full walk for `chain`)

Per-call cost is averaged over `n_iters` calls inside a tight
tail-recursive Prolog loop. `once/1` wraps hit-position queries to
exclude backtracking cost; miss-position queries are wrapped in
`(G -> true ; true)` to swallow failure.

## Headline result (N=100, 100,000 iters per cell)

```
shape                    first hit    last hit     miss
--------------------------------------------------------------
idx (constant)           4328.290     3287.180     3014.400
chain (no index)         5537.240     64840.160    63666.770
mma (fallthrough)        5780.220     6351.850     20645.210

speedup vs chain (no-index baseline):
  last-hit  idx 19.73x,  mma 10.21x
  miss      idx 21.12x,  mma 3.08x
```

The cost of a worst-case hit drops from ~65 μs (chain walks every
clause) to ~3 μs (constant indexing) — a **19.7× speedup**.

Misses see the same magnitude of speedup for indexed predicates
(21×): the switch finds no entry and returns false immediately,
whereas the chain walks every clause looking for a match.

Mixed-mode (the `_` catch-all default pattern from #2301) keeps the
indexed-hit speed (10× over chain) but pays for the miss case
because the fallthrough still walks the chain to find the variable
clause. Even so, it's 3× faster than no indexing at all because
the indexed clauses' constant heads fail unification fast (one
constant comparison) vs the chain version where the variable head
always unifies and the body has to run.

## Scaling

At N=10 (200,000 iters per cell):

```
shape                    first hit    last hit     miss
--------------------------------------------------------------
idx (constant)           4540.530     2972.480     2959.360
chain (no index)         5916.940     9577.750     8918.750
mma (fallthrough)        6285.180     6418.250     5629.920

speedup vs chain (no-index baseline):
  last-hit  idx 3.22x,  mma 1.49x
  miss      idx 3.01x,  mma 1.58x
```

Scaling matches expectations:

| N    | idx last (ns) | chain last (ns) | speedup |
| ---- | ------------- | --------------- | ------- |
| 10   | 2972          |  9578           |  3.2×   |
| 100  | 3287          | 64840           | 19.7×   |

## Hash-based const_table

The first version of the runtime used a linear scan of the
`const_table` vector. At N=100 that was already fast (~3 μs/call
is mostly the recursive Prolog loop overhead, not the switch
itself), but at N=500 the scan cost became visible: `idx_last`
rose to ~4.7 μs vs ~3.4 μs for `idx_first` — a clear ~1.3 μs of
"scanning 500 entries to find the matching one" cost.

The follow-up replaced the scan with a parallel
`std::unordered_map<Value, std::size_t>` built once in the
factory. `try_emplace` gives first-insert-wins semantics, matching
the previous behaviour when the WAM compiler emits duplicate keys.

Before / after at N=500 (20,000 iters per cell):

| metric         | linear scan | hash map | change           |
| -------------- | ----------- | -------- | ---------------- |
| idx_first (ns) |  3370       | 3821     | within noise     |
| idx_last  (ns) |  4652       | 2494     | **1.87× faster** |
| idx_miss  (ns) |  3396       | 2409     | **1.41× faster** |

The hash flattens the curve: `idx_last` is now indistinguishable
from `idx_first`. The speedup vs the no-index chain baseline grew
from 47.9× to **88.4×** (last-hit) and 65.4× to **93.1×** (miss).

At N=100 the change is within noise (~3 μs total dispatch is
already dominated by loop overhead, not table scan), so this is a
"no regression at small N, growing wins at large N" change rather
than a uniform speedup.

`idx_last` is essentially flat — the switch table is a linear scan
in the runtime (`for (auto& kv : instr.const_table)`), but the
per-key work is just a `Value::operator==`, much cheaper than full
WAM head unification. `chain_last` scales roughly linearly with N,
as the try_me_else chain walks every clause.

## What this doesn't measure

- **Hash-based switch.** The C++ runtime still uses a linear scan
  through `const_table`. For very large N (1000+ keys) a hash map
  would help. The current data shows this isn't yet a bottleneck
  at N=100, where idx is already fast enough that loop overhead
  dominates (~3 μs/call is mostly the recursive Prolog loop, not
  the switch itself).
- **Compile-time cost.** Generating + compiling N=100 clauses with
  `g++ -O2` on the full file takes ~75s. The benchmark works
  around this with a split build (-O2 only on `wam_runtime.cpp`,
  -O0 on the auto-generated setup file) and gets identical runtime
  numbers in ~30s.
- **Structure / term indexing.** Only `switch_on_constant` and its
  fallthrough variant are exercised here. `switch_on_structure`,
  `switch_on_term`, and the A2 variants are functionally tested in
  the e2e suite but not benchmarked yet.
- **Cache effects.** A 100-entry const_table fits comfortably in
  L1; behavior at 10k+ entries (or with sparse memory layout) may
  differ. The runtime is single-threaded so we don't measure any
  contention.

## Next bottlenecks

The 3 μs/call floor for `idx_first` is dominated by the recursive
Prolog loop overhead (decrement, allocate frame, call), not the
indexing path itself. Reducing dispatch further would mean
optimizing the per-call WAM overhead, not the switch.
