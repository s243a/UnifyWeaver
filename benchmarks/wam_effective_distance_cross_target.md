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

| Scale | Prolog (SWI, direct) | Rust (WAM + FFI kernel) | Elixir kernel (pre-fix)² | Elixir kernel (patched)² |
| ---: | ---: | ---: | ---: | ---: |
| dev (~200 edges) | 20 ms | 3 ms | 4 ms¹ | **2 ms¹** |
| 300 (~6k edges) | 111 ms | 23 ms | 680 ms | **213 ms** |
| 1k (~7k edges) | 115 ms | 28 ms | 5,078 ms | **886 ms** |
| 10k | 567 ms | 177 ms | 60,710 ms | **9,806 ms** |

¹ Elixir timings exclude script startup; Prolog timings include
SWI startup (~10 ms); Rust includes binary startup (~1 ms). At dev
scale the startup dominates; at larger scales the work dominates.

² Pre-fix and patched columns measured on the same hardware. The
patched kernel ships in this PR.

**Patched kernel changes** — see `WamRuntime.GraphKernel.CategoryAncestor.collect/7`
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

**Output cross-validation**: at every scale (dev/300/1k/10k) the
patched kernel produces identical row counts AND identical
effective_distance values (max delta < 1e-4) to
`data/benchmark/*/reference_output.tsv` and to Rust's output. The
specific value `Bose-Einstein_statistics → Physics → 0.999179`
matches across Rust, Prolog, and patched Elixir at dev. The off-by-one
correctness bug was masked at dev (no paths hit the boundary) and
only became visible at scale.

The eprof profile of the patched kernel at scale=300 shows the new
hot path: `Enum.member?`/`lists:member` (visited-list scan) at ~31%,
`ets:lookup` at ~13%, recursion-body lambda at ~12%, `Enum.reduce`
machinery at ~18%. Member-check dominates because the visited list
holds binary strings; switching to a `MapSet` doesn't help at
max_depth=10 (lists win on short-N membership). String→integer
interning at FactSource load time would help but is a larger refactor.

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
- **Elixir kernel scales poorly on path-enumeration**. Even after
  the patched-kernel cleanup (5-6× wins from removing per-step
  `length/1`/`Enum.split`/`++`), BEAM-interpreted recursion with
  per-call list copies for the visited list and per-step
  `Enum.member?` over binary strings is much slower than either
  Prolog's WAM or Rust's native code on the same algorithm. The
  patched kernel narrows the gap (~50× over Rust at 10k vs ~350×
  pre-fix), but it's still BEAM-interpreted vs native.

## Why the kernel still matters for Elixir

The earlier benchmark (`benchmarks/wam_elixir_graph_kernels.md`)
measured kernel-Elixir vs **WAM-Elixir-emitted** `tc/2` and got
982× at N=1000. This benchmark compares against **other targets'
implementations**, which is a different yardstick:

- vs WAM-Elixir-emitted: kernel ≈ 1000× faster (chain graph, n=1000)
- vs Prolog-direct: patched kernel ≈ 0.06× as fast (effective_distance, 1k)
  — was ≈ 0.01× pre-fix
- vs Rust-native-kernel: patched kernel ≈ 0.018× as fast (10k)
  — was ≈ 0.003× pre-fix

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

# Elixir kernel (TC + CategoryAncestor in WamRuntime; ETS-backed FactSource)
# Generate any minimal Elixir project to get wam_runtime.ex emitted, then
# write a driver script that loads TSVs into ETS and calls
# WamRuntime.GraphKernel.CategoryAncestor.collect_hops per (article, root) pair.
# (See benchmarks/wam_elixir_graph_kernels.md for the kernel API surface.)
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
5. Constant-factor cleanup: 5-6× over the original kernel by
   eliminating `length/1`/`Enum.split`/`++` from the recursion hot
   path and fusing the direct-edge check into the recursion reduce.

For BEAM-targeted deployments at scale, the next perf lever is
likely **NIF-backed kernels** (Rustler or Zigler) — port the same
kernel to native code and dispatch through the same FactSource
contract. Different work item; not in this measurement.

## Cross-references

- `docs/WAM_TARGET_ROADMAP.md` — strategic context.
- `benchmarks/wam_elixir_graph_kernels.md` — Elixir kernel vs
  WAM-Elixir-baseline on synthetic chain graph.
- `examples/benchmark/effective_distance.pl` — Prolog source.
- `src/unifyweaver/targets/wam_rust_target.pl:1797` —
  `collect_native_category_ancestor_hops` reference impl.
- `src/unifyweaver/core/recursive_kernel_detection.pl` — shared
  kernel detector (target-neutral).
