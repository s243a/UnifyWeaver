# Cross-target effective_distance: Prolog, Rust kernel, Elixir kernel

Companion to `benchmarks/wam_elixir_graph_kernels.md`. That doc
measured the kernel-vs-WAM baseline on a synthetic chain graph
(982× at N=1000); this doc measures the same kernel against **other
targets' implementations** of the same real-world query —
`effective_distance/3` over Wikipedia category graphs.

## TL;DR

Kernel-based lowering reproduces correct effective_distance values
across targets. Rust's native-compiled kernel is the clear perf
winner at scale; SWI-Prolog's native indexing puts it ahead of the
BEAM-interpreted Elixir kernel on this path-enumeration-heavy
workload at all sizes ≥ 300 nodes.

## What this measures

Same `effective_distance/3` query (`d_eff = (Σ d^(-N))^(-1/N)`,
N=5) over four scales of Wikipedia category data, with **Physics**
as the single root category.

| Dataset | category_parent edges | article_category edges | output rows |
| --- | ---: | ---: | ---: |
| dev | 198 | 31 | 19 |
| 300 | 6,008 | 232 | 271 |
| 1k | 6,943 | 549 | 580 |
| 10k | (large) | (large) | 5,192 |

## Results

Wall-clock for the full pipeline (read TSVs, compute all
`(article, root, distance)` rows, output). Single-machine, 4-vCPU
Intel Xeon @ 2.10 GHz.

| Scale | Prolog (SWI) | Rust | Elixir orig.² | + struct² | + bare-dests³ | + walker⁴ | + fold⁵ | + int-tuple⁶ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev (~200 edges) | 20 ms | 3 ms | 4 ms¹ | 2 ms¹ | 0.5 ms¹ | 0.3 ms¹ | 0.2 ms¹ | **0.15 ms¹** |
| 300 (~6k edges) | 111 ms | 23 ms | 680 ms | 213 ms | 76 ms | 38 ms | 31 ms | **18 ms** |
| 1k (~7k edges) | 115 ms | 28 ms | 5,078 ms | 886 ms | 258 ms | 146 ms | 125 ms | **68 ms** |
| 10k | 567 ms | 177 ms | 60,710 ms | 9,806 ms | 3,089 ms | 1,624 ms | 1,400 ms | **764 ms** |

¹ Elixir timings exclude script startup; Prolog timings include
SWI startup (~10 ms); Rust includes binary startup (~1 ms). At dev
scale the startup dominates; at larger scales the work dominates.

² All Elixir columns measured on the same hardware. "Original" is
the kernel as initially shipped (PR #1799); "struct" is post-PR
#1809; "bare-dests" is post-PR #1810; "walker" is post-PR #1811;
"fold" is post-PR #1812; "int-tuple" is this PR.

³ The bare-dests column uses three composable wins (any one alone
helps; together they compound to ~3× over the structural-fix kernel):
- `collect_hops_with_dests/4` — kernel API where `dests_fn`
  returns bare destinations `[to1, to2, ...]` instead of
  `{from, to}` tuples. Saves the per-neighbor tuple wrap+unwrap
  that was ~22% of runtime under the neighbors_fn API.
- Atom-interned keys at FactSource load time — atom-compare is
  pointer-compare on BEAM (vs binary byte-compare). Cuts member-check
  cost on the visited list by ~40%.
- Direct `Map` for the edge index instead of `:ets.lookup` —
  eliminates ETS message-passing overhead per neighbor query.

⁴ The walker-fusion column adds three more compounding wins (~1.8×
over bare-dests):
- Drop the invariant `root_blocked? = :lists.member(root, visited)`
  check. Recursion never adds `root` to `visited` (the `to == root`
  branch pushes a hop and stops without recursing), so `root_blocked?`
  was always false. Saves ~6.5% (halves the lists:member call count).
- Replace `Enum.reduce` with hand-recursive `walk_*_recurse/N` and
  `walk_*_leaf/N`. The `can_recurse?` check is invariant for the
  whole dests list at one recursion level, so it's hoisted ABOVE the
  loop (recurse-phase walker can descend; leaf-phase walker only
  counts direct hits). Eliminates the `Enum.reduce` closure
  allocation + per-call dispatch (~22% of runtime in the post-bare-dests
  profile, plus ~14% on the closure body).
- Fuse the `collect_d/7` recursion-step trampoline into the
  `walk_d_recurse/7` rec branch. The trampoline (compute next_depth,
  branch on bound, dispatch to walker) was ~20% of runtime per
  recursion step. Inlining saves one function dispatch per step.

⁵ The fold column adds the BEAM-native producer/consumer fusion
primitive: `fold_hops_with_dests/6`. The walker calls a caller-supplied
`hop_fn.(hop, acc)` on each direct hit instead of consing onto a list,
so the consumer fold is invoked INSIDE the producer recursion frame.
No intermediate hop list is ever allocated. Modest perf win
(~5-15% across scales — most of the bench timing is in the kernel
walk itself, not the aggregation), but the structural significance
is larger: this is the BEAM analogue of Rust iterator monomorphization
and Haskell GHC deforestation.

⁶ The int-tuple column changes the FactSource representation, not
the kernel. The kernel itself is term-agnostic — `==`, `:lists.member`,
and `Map.get` all work on whatever the caller supplies. Three
patterns measured:

| Scale | atom (Map) | int (Map) | int (tuple) |
| ---: | ---: | ---: | ---: |
| dev | 247 μs | 317 μs | **151 μs** |
| 300 | 34,814 μs | 34,009 μs | **18,410 μs** |
| 1k | 127,761 μs | 139,312 μs | **68,103 μs** |
| 10k | 1,505,144 μs | 1,659,082 μs | **763,930 μs** |

Atom-Map and int-Map are within 10-15% of each other (atom-table
hash precomputation gives atoms a small per-call edge that
disappears as Map lookups dominate). Int-tuple wins ~2× over both
because `elem(cp_tuple, id)` is O(1) without hashing — the
contiguous integer IDs from a one-shot intern pass (or from an
LMDB-backed FactSource that already keys nodes by integer ID, the
shape the Haskell target uses) let us replace the HAMT with a flat
indexed structure.

Why this matters beyond perf: BEAM's atom table is bounded
(~1M default, shared with the rest of the VM). Production-scale
Wikipedia category data has ~1M unique categories — atom-interning
is not viable there. The int-tuple path is the recommended scale-up
recipe AND happens to be the fastest path at our measured scales.

**Structural-fix changes** (PR #1809) — see
`WamRuntime.GraphKernel.CategoryAncestor.collect_n/7`
in `src/unifyweaver/targets/wam_elixir_target.pl`:

1. Pass an integer `depth` through the recursion and push `depth + 1`
   directly when a direct edge to root is found. Eliminates the
   post-recursion `length(ac)` + `length(rec_acc)` + `Enum.split` +
   `Enum.map(&(&1+1))` + `++` pattern that was ~28% of runtime.
2. Fuse the direct-edge `Enum.any?` check into the same `Enum.reduce`
   that recurses, so neighbors are scanned once per step instead of twice.
3. Skip recursion through `to == root` — those subtrees can't add new
   entries (the next step would have `root in visited` blocking every
   downstream direct-edge hit).
4. Bound semantics fix: align with Rust's reference implementation
   (`visited.len() >= max_depth` with `[cat]` seed). The previous Elixir
   kernel started with `visited = []` AND used `length(visited) >= max_depth`,
   which allowed one extra level of direct check vs Rust and over-counted
   paths by ~7% near the depth boundary on Wikipedia category data
   (e.g., 621 rows at scale-1k vs 580 in Rust/Prolog/`reference_output.tsv`).
   The fix uses `next_depth < max_depth`, restoring exact agreement.

**Bare-dests path changes** (PR #1810):

5. Add `collect_hops_with_dests/4` and `collect_d/7` — a parallel API
   where `dests_fn` returns bare destinations. Profile-driven: at
   scale-1k the previous neighbors_fn API spent ~19% of runtime on
   `Enum.map` wrapping/unwrapping `{from, to}` tuples that the kernel
   immediately discarded `from` from. The new path takes destinations
   directly and the bench harness demonstrates the recipe.
6. Replace `to in visited` with `:lists.member(to, visited)` directly,
   skipping the `Enum.member?` protocol-dispatch overhead (~6-7% of
   runtime at 1k-scale on top of the underlying `:lists.member` call).

**Walker-fusion changes** (PR #1811). After PR #1810 the eprof
profile showed two structurally removable costs: `Enum.reduce/3
lists^foldl/2-0` at ~22% (closure machinery) plus the closure body
at ~14%, and `collect_d/7` recursion entry at ~20% (a thin
trampoline). Together that was ~56% of runtime spent on dispatch
and iteration overhead, not on the algorithm itself.

7. Replace each `Enum.reduce(dests, acc, fn ... end)` with
   tail-recursive `walk_n_recurse/7` + `walk_n_leaf/4` (and the same
   for `walk_d_*`). Each pair splits the per-element conditional:
   recurse-phase iterates and may descend; leaf-phase only counts
   direct hits at depth==max_depth. That hoists the
   `next_depth < max_depth` check ABOVE the loop instead of paying
   it per neighbor.
8. Drop the invariant `root_blocked?` check — recursion never adds
   `root` to `visited`, so it was always false.
9. Fuse the `collect_d/7` recursion-step trampoline directly into the
   `walk_d_recurse/7` "true" branch. One function dispatch per
   recursion step instead of two.

**Fold path changes** (this PR — see footnote ⁵). Audit of which
graph kernels Rust + Haskell have that Elixir does not surfaced a
deeper structural mismatch on `effective_distance/3`: Haskell emits
the `category_ancestor/4` call inside a single GHC list-comprehension
that fuses with the power-sum aggregation
(`!weightSum = sum [((hops + 1) ** negN) | hops <- solutions]`),
so deforestation collapses producer + consumer into one tight loop
with no intermediate sequence. Rust gets the same thing via
iterator monomorphization + inlining. Elixir was building a
materialised list of hops, then iterating it through `Enum.map(+1)`,
`Enum.group_by`, `Enum.concat`, `Enum.reduce(:math.pow)` — multiple
passes plus heavy GC pressure on the cons cells.

`Stream.unfold/2` is *not* the BEAM analogue (runtime laziness via
closure trampolining adds per-yield overhead — for tight numeric
loops a `Stream`-based path is ~2-4× slower than the plain list).
The BEAM-native primitive is **callback-fold**: pass the consumer
fold INTO the producer.

10. Add `fold_hops_with_dests/6` —
    `(dests_fn, cat, root, max_depth, init_acc, hop_fn) -> acc`. The
    walker calls `hop_fn.(hop, acc)` on each direct hit instead of
    consing onto a list. No intermediate sequence is allocated; the
    init_acc threads through every recursion frame. `collect_hops_with_dests/4`
    is kept as a separate inlined defp (not a wrapper around
    `fold_hops_with_dests`) so the non-aggregating path keeps its
    `[hop | acc]` and pays no closure dispatch.
11. Update the bench harness to use `fold_hops_with_dests/6` for
    `effective_distance` weight-sum: per (article, cat, root) tuple
    the kernel folds directly to a partial weight contribution
    (`acc + :math.pow(hop + 1, -n)`), then partial sums combine over
    `(article, root)` groups by `Map.update`. No per-pair list ever
    exists.

**Emitter dispatch-fold integration** (this PR — first emitter-side
step toward the producer-consumer fusion that Rust + Haskell get for
free from monomorphisation / GHC deforestation). The kernel-dispatch
wrapper that the WAM-Elixir target generates for kernel-recognised
predicates (`compile_category_ancestor_dispatch_module/5`) used to
build a hop list via `collect_hops/4` and then iterate it through
`merge_into_aggregate` whenever the call site was inside an
aggregate frame. The new path uses fold-form directly:

12. Add `fold_hops/6` (tuple-input variant of `fold_hops_with_dests/6`)
    that pairs with `collect_hops/4`. Same semantics as
    `fold_hops_with_dests/6` but the neighbours come back as
    `{from, to}` tuples — the contract `WamRuntime.FactSource.lookup_by_arg1`
    returns and the kernel-dispatch wrapper sees. Implementation is
    parallel `fold_n` / `fold_n_recurse` / `fold_n_leaf` mirroring
    the existing `fold_d_*` walkers.
13. Add `WamRuntime.aggregate_push_one/2` runtime helper. Same role
    as `aggregate_collect/2` but takes the value directly instead of
    reading from a register. Convention matches `aggregate_collect/2`
    (prepend, O(1)) — the existing `Enum.reverse` in
    `finalise_aggregate/4` restores encounter order.
14. Modify the dispatch wrapper template: when
    `in_forkable_aggregate_frame?(state)` is true, fold each hop
    directly into the aggregate frame. No intermediate hop list is
    built. The non-aggregate (backtracking) branch keeps
    `collect_hops/4` since it needs to bind individual solutions into
    the hops register one at a time.

**Dispatch-fold regression fix** (this PR — measured by a fresh
micro-bench that simulates the full aggregate-frame path through the
runtime APIs at varying cp-stack depths). The first cut of the
dispatch-fold integration used `aggregate_push_one/2` per hit, which
walks `state.choice_points` to find the aggregate frame on every
push. That made the per-kernel-call cost O(N×D) where the legacy
collect+merge path was O(D + N) — a measurable regression at
non-trivial cp-stack depth (D≥5). Numbers from
`bench_dispatch.exs` at scale-300:

| cp-stack depth | collect+merge (μs) | fold+push_one (μs) | fold+split (μs) |
| ---: | ---: | ---: | ---: |
| 0 | 63,041 | 68,898 | **61,083** |
| 5 | 61,190 | 62,829 | **60,874** |
| 20 | 62,029 | 80,314 | 67,530 |

15. Add `WamRuntime.split_at_aggregate_cp/1` — one-pass cp-stack
    walk that returns `{above_cps_in_order, agg_cp, below_cps}`.
    Walks the cp list once instead of N times.
16. Modify the dispatch wrapper to **extract the agg cp once via
    `split_at_aggregate_cp/1`, fold against `agg_cp.agg_accum`
    directly with `[hop | accum]`, then reassemble**
    `state.choice_points` as `above ++ [updated_agg_cp | below]`.
    Total cp work per kernel call is O(D) instead of O(N×D).
    The fold itself stays the same; the closure no longer touches
    the cp stack. At typical Prolog cp-stack depth (≤5) this matches
    the legacy collect+merge path. (True O(1) would require a
    structural separation of the active-aggregate-frames stack from
    `choice_points`; BEAM has no equivalent of Go's compiled `switch`
    jump table to provide constant-time dispatch for free.)

**Scope of the dispatch-fold change.** Equivalent semantics to the
legacy collect+merge path for `findall`, `bag`, `set`, `count`,
`sum`, `max`, `min` aggregates whose value register IS the kernel's
hop output — the most common shapes. **NOT yet handled**:
transform-aware aggregates such as
`aggregate_all(sum(W), (Goal, W is Hops ** -n), R)` from
`effective_distance/3` — the wrapper does not see the `W is f(Hops)`
arithmetic step, so it would push raw hops where `f(hops)` values
are expected. Those queries currently bypass kernel dispatch and run
on plain WAM bytecode; lifting them into fold-form requires the WAM
IR pattern-recognition pass that compiles the per-solution
arithmetic into a closure passed as `hop_fn`. That is the natural
next emitter step — same architectural shape as this PR but with
Prolog-arithmetic-to-Elixir compilation added.

**Output cross-validation**: at every scale (dev/300/1k/10k) the
patched kernel produces identical row counts AND identical
effective_distance values (max delta < 1e-4) to
`data/benchmark/*/reference_output.tsv` and to Rust's output. The
specific value `Bose-Einstein_statistics → Physics → 0.999179`
matches across Rust, Prolog, and patched Elixir at dev. The off-by-one
correctness bug was masked at dev (no paths hit the boundary) and
only became visible at scale.

eprof on the fold path at scale=1k shows the residual hot spots
are essentially identical to the walker-fusion baseline:
`fold_d_recurse/8` at ~51% (the inner cond+pattern-match loop),
`Map.get/3` at ~13%, `:lists.member/2` at ~12%, closure dispatch
glue at ~11%, `fold_d_leaf/5` at ~12%. The hop_fn closure (called
~101k times across the workload at 1k vs ~1.93M recursion steps —
roughly 19:1) plus `:math.pow/2` together account for only ~2% of
runtime, confirming the per-HIT closure cost is small.

The kernel-side BEAM-only wins are essentially exhausted at this
point. Remaining levers are (a) a third kernel variant specialised
on a Map-typed source — ~30 lines of near-duplicate walker code,
~10% headroom from skipping closure dispatch on `Map.get`; or (b)
NIF-backed kernels via Rustler — port the same kernel to native
code, dispatch through the same FactSource contract.

The structurally bigger follow-up is the **WAM-Elixir emitter
pass** that detects `findall(H, Goal, L), sum_list(L, S)` /
`aggregate_all(sum(W), Goal, S)` shapes in the WAM IR and lowers
them automatically to `fold_hops_with_dests/6` instead of
`collect_hops_with_dests/4` followed by an `Enum.reduce`. That
restores architectural symmetry with how Rust and Haskell handle
producer-consumer composition for free (iterator monomorphization,
GHC deforestation), and generalizes to count, max, and any future
aggregation kernel without per-aggregation special-casing.

## Reading the gap

- **Rust > everyone**. Native compilation + arena-style bookkeeping
  + interned atom IDs as `u32`. The category_ancestor FFI kernel
  here is the same shape that the roadmap doc cited as the
  existence proof of kernel-based-lowering wins.
- **Prolog (SWI direct) beats Elixir kernel above 300 edges**. SWI's
  first-arg indexing on `category_parent/2` is highly tuned C code;
  per-step neighbor lookup is faster than the BEAM `:ets.lookup`
  the Elixir kernel uses. Path enumeration multiplies the per-step
  cost by `O(num_simple_paths)`, so the constant-factor difference
  blows up.
- **Elixir kernel scales poorly on path-enumeration**. Stacking
  the structural fix + bare-dests + atom interning + Map index +
  walker fusion + fold closes the gap from ~343× over Rust at 10k
  (original) to ~8× (this PR). BEAM-interpreted recursion still
  loses to Rust's native code on the same algorithm — final gap
  is dominated by the recursion+iteration loop itself and the
  closure-dispatched `Map.get`.

## Why the kernel still matters for Elixir

The earlier benchmark (`benchmarks/wam_elixir_graph_kernels.md`)
measured kernel-Elixir vs **WAM-Elixir-emitted** `tc/2` and got
982× at N=1000. This benchmark compares against **other targets'
implementations**, which is a different yardstick:

- vs WAM-Elixir-emitted: kernel ≈ 1000× faster (chain graph, n=1000)
- vs Prolog-direct: int-tuple path ≈ 1.7× as fast (effective_distance, 1k)
  — was ≈ 0.92× with fold, ≈ 0.79× with walker-fusion,
  ≈ 0.45× with bare-dests, ≈ 0.13× with structural-fix only,
  ≈ 0.02× original
- vs Rust-native-kernel: int-tuple path ≈ 0.23× as fast (10k)
  — was ≈ 0.13× with fold, ≈ 0.11× with walker-fusion,
  ≈ 0.057× with bare-dests, ≈ 0.018× with structural-fix only,
  ≈ 0.003× original

Useful framing: the kernel makes the BEAM-deployed Elixir version
competitive with itself, not with native targets. For applications
that need to deploy on the BEAM ecosystem (Elixir/Erlang runtime),
the kernel is the fast path THERE. Applications that have a choice
of runtime should pick Rust for path-heavy graph workloads.

## What this measurement does NOT cover

- **LMDB-as-storage benefit**. Per `docs/WAM_TARGET_ROADMAP.md`,
  LMDB integration is the highest-priority Elixir follow-up for
  workloads above 100k facts. PR #1792 shipped the FactSource
  adaptor with emit-and-grep tests only; **runtime validation
  requires the `:elmdb` Hex package**, which couldn't be installed
  in the sandbox where these measurements were taken (Hex.pm
  returns 403). Recommend running the equivalent measurement in a
  driver project that has `:elmdb` installed; expect a meaningful
  improvement at 100k+ scale where ETS memory pressure starts to
  dominate.

- **Haskell side**. GHC was installed and Hackage is reachable, but
  the WAM-Haskell pipeline pulls 9 cabal packages
  (containers/array/time/unordered-containers/hashable/deepseq/
  parallel/async + base) — substantial setup-time investment.
  Haskell with LMDB is the comparison the roadmap doc most wants;
  defer to a session with that infrastructure pre-built.

- **Memory-mapped file path** (vs ETS). This entire benchmark uses
  ETS bag tables for Elixir's edge source. The dominant cost in
  Elixir is per-step `:ets.lookup`; LMDB might be slower per-call
  but better per-byte at scale. Untestable here.

## Reproducing

Setup:
- SWI-Prolog (any recent version — these used 9.0.4)
- Elixir 1.14+ (sandbox: 1.14.0)
- Rust + Cargo (sandbox: cargo std-only build, no crates.io needed)
- Workload data: `data/benchmark/dev/`, `300/`, `1k/`, `10k/`

Steps:

```sh
# Rust (cargo std-only)
swipl -q examples/benchmark/generate_wam_rust_matrix_benchmark.pl \
      -- data/benchmark/dev/facts.pl /tmp/rust_ed seeded interpreter kernels_on
cd /tmp/rust_ed && cargo build --offline --release
time ./target/release/wam_rust_matrix_bench \
     data/benchmark/dev/category_parent.tsv \
     data/benchmark/dev/article_category.tsv > /tmp/rust_out.tsv

# Prolog
time swipl -q -g 'consult("examples/benchmark/effective_distance.pl"), \
                  [user:"data/benchmark/dev/facts.pl"], \
                  assertz(user:dimension_n(5)), \
                  findall(A-R-D, user:effective_distance(A,R,D), L), \
                  length(L, N), format("~w~n", [N]), halt'

# Elixir kernel — bare-dests recipe (recommended fast path):
# 1. Atom-intern category names at TSV-load time:
#       intern = fn s -> String.to_atom(s) end
# 2. Build a process-local Map of cat -> [parent_atoms] (no ETS):
#       cp_map = ... |> Enum.reduce(%{}, fn ... -> Map.update(...) end)
# 3. Use bare-destinations FactSource (returns a flat list, no tuples):
#       dests_fn = fn n -> Map.get(cp_map, n, []) end
# 4. For aggregations (sum, count, max, power-sum), use
#    WamRuntime.GraphKernel.CategoryAncestor.fold_hops_with_dests/6 to
#    fuse the consumer fold INTO the producer recursion — no
#    intermediate hop list:
#       fold_hops_with_dests(dests_fn, cat, root, 10, 0.0,
#         fn hop, acc -> acc + :math.pow(hop + 1, -n) end)
# 5. For consumers that genuinely need the materialised list, use
#    WamRuntime.GraphKernel.CategoryAncestor.collect_hops_with_dests/4
#    instead. (See bench_fold.exs / bench_dests.exs in PRs #1810/#1812.)
```

## Conclusion + next steps

The kernel-based-lowering work for Elixir (PRs #1799/#1800/#1801/
#1803) reproduces correct effective_distance answers and runs the
same algorithm as Rust's reference implementation. **It does not
make BEAM-Elixir competitive with native-compiled targets on path-
enumeration-heavy workloads** — that requires going to the BEAM
NIF layer (Rustler-backed kernels) or accepting the JIT penalty
relative to Rust.

What the kernel DOES achieve:
1. Correct path-enumeration semantics (cross-validated vs Rust at
   every scale; the patched kernel matches `reference_output.tsv` to
   < 1e-4 absolute on every (article, root) pair, fixing a previous
   off-by-one at the depth boundary).
2. Compositional with FactSource adaptors (ETS today, LMDB once
   `:elmdb` is installable).
3. Massive speedup over WAM-emitted recursive Prolog (982× at N=1000
   on chain graph).
4. Pattern-recognition auto-routing: user writes the canonical
   shape, the kernel fires.
5. Constant-factor cleanup: ~80× over the original kernel
   stacking the structural fix (PR #1809), the bare-destinations
   API path (PR #1810), walker fusion (PR #1811), the fold path
   (PR #1812), and the int-tuple FactSource pattern (this PR —
   integer node IDs in a tuple-as-array, O(1) `elem/2` reads,
   no atom-table pressure). **Final gap to Rust at 10k: ~4.3×,
   down from ~343×.**
6. Architectural symmetry with Rust + Haskell on producer/consumer
   composition: `fold_hops_with_dests/6` is the BEAM analogue of
   Rust iterator monomorphization and Haskell GHC deforestation.
7. Production-scale viability: integer-keyed FactSource path
   matches the architecture the Haskell target uses (LMDB-derived
   integer IDs as comparison keys). Scales to ~1M unique nodes
   without atom-table cap concerns.

For BEAM-targeted deployments at scale, the remaining perf levers
are (a) a WAM-Elixir emitter pass to auto-route `findall + sum_list`
/ `aggregate_all(sum/count/max, ...)` shapes to fold-form lowering
(restoring the symmetry to the codegen layer, not just the kernel
API), or (b) **NIF-backed kernels** (Rustler or Zigler) — port the
kernel to native code, dispatch through the same FactSource
contract. Different work items; not in this measurement.

## Cross-references

- `docs/WAM_TARGET_ROADMAP.md` — strategic context.
- `benchmarks/wam_elixir_graph_kernels.md` — Elixir kernel vs
  WAM-Elixir-baseline on synthetic chain graph.
- `examples/benchmark/effective_distance.pl` — Prolog source.
- `src/unifyweaver/targets/wam_rust_target.pl:1797` —
  `collect_native_category_ancestor_hops` reference impl.
- `src/unifyweaver/core/recursive_kernel_detection.pl` — shared
  kernel detector (target-neutral).
