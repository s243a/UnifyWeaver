# WAM-Rust Boundary Distribution Cache — Implementation Plan

Status: design / groundwork. Scopes porting the **boundary-splice / suffix-distribution
cache** (prototyped and validated correct in Python by the `codex/*` distribution
line) into the Rust WAM graph-search target. Companion to
`DISTRIBUTION_CACHE_BENCHMARK_PLAN.md`, `DISTRIBUTIONAL_COMPRESSION_THEORY.md`,
`RECURRENCE_EVALUATION_STRATEGY_*.md`, and the EnWiki splice-validation reports.

This plan deliberately **re-derives the cost tradeoffs for compiled Rust** rather
than inheriting the Python prototype's performance conclusions — because, as with
the edge cache (see below), several of those conclusions are interpreter-overhead
artifacts, not properties of the algorithm.

## 1. What the boundary cache is, and why the splice is valid

The effective-distance query aggregates over all bounded paths from a seed to a
root up `category_parent`:

```
d_eff = (Σ_paths  Hops(path)^(-N))^(-1/N)        (effective_distance.pl:90,108-114)
WeightSum = Σ_paths Hops^(-N)
```

`WeightSum` is a **linear functional of the path-length histogram**
`H[L] = #{paths of length L}`: `WeightSum = Σ_L H[L]·L^(-N)`
(cf. `weighted_power_cdf`, `tools/distribution_cache_support.py:303-304`).

Path-length histograms compose by **convolution** over a cut node B (boundary):
every seed→root path through B is a seed→B prefix path concatenated with a B→root
suffix path, and lengths add:

```
H_seed→root[L] = Σ_{a+b=L} H_seed→B[a] · H_B→root[b]            (boundary splice)
```

So if we precompute the **suffix** histogram `H_B→root` at boundary nodes B near
the root, a query that walks up and reaches B can **splice** the cached suffix
over its remaining budget instead of continuing to enumerate B→root for every
seed. The Python prototype proved this **exact** on EnWiki MTC — zero delta on
mass, value-sum, and mean length vs full DFS
(`enwiki_mtc_shallow_splice_validation_2026-06-14.md`). The algorithm is correct;
this plan ports it.

## 2. What does NOT transfer from the Python prototype (the interpreter-overhead trap)

The same dynamic we already proved on the **edge** cache applies here. The Python
edge-cache benchmark concluded "cached is *slower*" (ratio 1.66); the compiled
Rust port measured cached **2.5–3.5× faster** (`wam_rust_cached_scaling_sweep`),
because the residual that dominated Python was interpreter overhead, not algorithm
cost. The boundary-cache prototype shows the **identical signature**:

| Python conclusion | Source | Cost basis | Transfers to Rust? |
|---|---|---|---|
| "Cached search slower (ratio 1.66) despite many hits" | `enwiki_mtc_boundary_cache_runtime_attribution` | **75.6% unattributed traversal/interpreter overhead**; splice 0.1%, decode 0.4% | **No** — pure Python residual |
| "Still ~11.6% slower after decode optimisation" | `enwiki_mtc_boundary_cache_hit_overhead` | per-node Python recursion/bookkeeping | **No** |
| "Fewer nodes (0.82×) but slower (1.33×)" | `..._boundary_aggregate_root_cone_b2_t4...` | Python per-node cost > node savings | **No** — the node savings are real |
| Splice/decode are negligible | attribution buckets (0.1% / 0.4%) | algorithm cost | **Yes** (and even cheaper in Rust — see §3) |
| Boundary-hit fraction ~24% on EnWiki MTC | coverage probe | graph structure / reuse | **Yes** (durable) |
| 50-point exact→approximate threshold | `DISTRIBUTIONAL_COMPRESSION_THEORY` | **storage** bytes | **Yes** (durable; see §4) |
| `D_pre` / marginal-speedup crossovers | `DISTRIBUTION_CACHE_BENCHMARK_PLAN` ("hypotheses, not commitments") | **Python wall-time** — overhead so large the curves are unreadable | **No** — must be re-measured in Rust |

**Conclusion:** keep the algorithm, the storage-based decisions, and the
structural reuse facts. Re-derive every wall-time/compute crossover for Rust.

## 3. The sharp Rust criterion (the non-obvious part)

In Rust the boundary cache does **not** automatically win — and not for Python's
reasons. The kernel DFS is already native (`collect_native_category_ancestor_hops`,
`wam_rust_target.pl:2433-2499`) and, after the edge cache, parent lookups are
~L1-cache reads. So in Rust **both** the suffix walk *and* the splice are cheap.
The cache wins iff:

```
(native DFS work avoided in the B→root cone) × (reuse across seeds)  >  splice cost
```

The splice cost is essentially zero. Measured here: caching the suffix as a
**pre-weighted vector** `g_B[a] = Σ_b H_B→root[b]·(a+b)^(-N)` makes the query-time
splice a dot product `WeightSum += Σ_a H_seed→B[a]·g_B[a]` —

```
micro-bench (this machine): query-time dot-product splice = 1.23 ns
                            (raw-histogram convolve + powf  = 5.85 µs;
                             Python dict convolve            = 35 µs)
```

So the criterion reduces to: **the cached suffix cone must represent enough avoided
native DFS, reused enough times, to beat ~1 ns.** That is overwhelmingly true for
**root-near** boundaries (the B→root cone is the whole shared upper graph, traversed
by *every* seed — exactly the hub-reuse the cache-scaling work measured) and a wash
for deep boundaries (tiny cone, low reuse). Two consequences that differ from the
Python framing:

- **Cache only shallow / root-near boundaries.** The optimal `D_pre` is likely
  *shallower* in Rust than the Python curves suggest: Python overhead made even
  small caches look marginal, so its "diminishing returns past D_pre 3–4" is
  overhead-shaped. In Rust only the top reuse band clears the ~1 ns splice — but it
  clears it by orders of magnitude.
- **The boundary cache is a layer *on top of* the edge cache, not a replacement.**
  The edge cache removes LMDB-seek cost but each seed still *walks* the suffix cone
  (native DFS over cached edges). The boundary cache removes the walk itself. So its
  remaining headroom is exactly the native-DFS time left after the edge cache — to
  be measured, not assumed.

## 4. Representation: pre-weighted vector, exact by default

Cache, per boundary node and per hot functional, the **cumulative basis** vector
`g_B[a]` for `a = 0..max_depth` (the "cumulative bases" of
`DISTRIBUTION_CACHE_BENCHMARK_PLAN.md` — `mass`, `moment(1)`, `weighted_power(N)`).
For `d_eff` that is `g_B[a] = Σ_b H_B→root[b]·(a+b)^(-N)`, truncated at the budget.
This is:

- **Cheaper at query time** than caching the raw histogram and shift-adding
  (`add_shifted_hist`, `lmdb_parent_boundary_cache_benchmark.py:517-525`): a dot
  product, no convolution, no `powf`.
- **Smaller**: one `f64` vector of length `≤ max_depth` (~10–20) per boundary per
  functional, vs a variable-support histogram.

Because the exact splice is ~1 ns, the **exact→approximate ladder is justified by
storage only** (the 50-point/byte argument), never by compute. The Rust default is
**exact** `g_B`; fall back to a fitted form (binomial / discretised GMM, per the
compression theory) only when a node's basis exceeds the storage budget — decoupling
the two concerns the Python policy had to entangle.

(Keep the raw suffix histogram cacheable too, for cold/ad-hoc functionals; the
`g_B` vectors are the hot-path optimisation.)

## 5. Rust integration (reuses existing scaffolding)

From the runtime map, the plug-in points already exist:

1. **Storage** — add a `boundary_basis` LMDB sub-db (node id → packed `g_B` /
   histogram bytes), alongside `s2i/i2s/category_parent/category_child`
   (`lmdb_fact_source_heed.rs.mustache:42-84`). Load once into a side-table
   (`HashMap<i32, Box<[f32]>>`), mirroring `load_s2i`. Built by a new ingest pass
   (the recurrence in `DISTRIBUTIONAL_COMPRESSION_THEORY.md:15-27`, computed in
   Rust or written by the existing Python precompute and just *read* by Rust).
2. **Kernel** — a `category_ancestor_boundary` native kind: walk up as today, but
   when a node is in the boundary side-table, **splice `g_B` and stop** instead of
   recursing. Register via `kernel_native_kind/2` +
   `recursive_kernel_detection.pl:54-68`; add the match arm in
   `execute_foreign_predicate` (`wam_rust_target.pl:1894-2346`); reuse the existing
   `min_dist` A* prune (`:2451-2460`).
3. **Result** — emit the same aggregate the current kernel does (so it drops into
   the existing `effective_distance` pipeline unchanged), gated so default output
   is identical to today.
4. **Value/side-table** — no `Value` change needed; the basis lives in a native
   side-table (like `min_dist`), not as heap `Value::List`.

Phasing: (P1) exact `g_B` side-table + boundary kernel arm + exec test asserting
boundary-spliced result == full-DFS result on a synthetic graph (the splice
identity, in Rust). (P2) wire the LMDB `boundary_basis` sub-db + a precompute pass.
(P3) the measurement in §6. (P4) storage-gated approximate bases.

## 6. What to measure (re-derive, don't inherit)

Using the synthetic generator (`generate_synthetic_category_graph.py`) and the
cached/lazy harness, add a **boundary-cached** variant and measure, at enwiki-ish
scale, the three-way `lazy` vs `edge-cached` vs `edge-cached + boundary-cached`:

- Does the boundary cache add measurable wall-time **on top of** the edge cache,
  and from what `D_pre`? (The Rust crossover, expected shallower than Python's.)
- Boundary-hit fraction vs `D_pre` (durable from Python ~24%, confirm in Rust).
- Correctness invariant: boundary-cached output == edge-cached output, exactly
  (the Rust analog of the Python exact-match validation).

## 7. Risks / open questions

- **Headroom may be modest.** After the edge cache, the suffix walk is cheap native
  code; the boundary cache's win is bounded by that residual. P3 measures it before
  P4 is worth building. (This is the honest version of "Rust makes the cache win" —
  it might be a small win here precisely because Rust already made the *edge* cache
  win.)
- **Cycle / filter policy must match** between precompute and query (the Python
  validation's exactness depended on identical root, filter, and cycle handling) —
  port those invariants, not just the splice.
- **Multi-functional caching** multiplies storage by #hot-functionals; gate by the
  same demand/storage budget the edge cache uses (R8b).
