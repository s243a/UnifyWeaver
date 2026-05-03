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
| 1k | 6,943 | 549 | 621 |
| 10k | (large) | (large) | 5,193 |

## Results

Wall-clock for the full pipeline (read TSVs, compute all
`(article, root, distance)` rows, output). Single-machine, 4-vCPU
Intel Xeon @ 2.10 GHz.

| Scale | Prolog (SWI, direct) | Rust (WAM + FFI kernel) | Elixir (kernel-dispatched) |
| ---: | ---: | ---: | ---: |
| dev (~200 edges) | 20 ms | 3 ms | 4 ms (kernel only)¹ |
| 300 (~6k edges) | 111 ms | 23 ms | 667 ms |
| 1k (~7k edges) | 115 ms | 28 ms | 4,977 ms |
| 10k | 567 ms | 177 ms | 61,541 ms |

¹ Elixir timings exclude script startup; Prolog timings include
SWI startup (~10 ms); Rust includes binary startup (~1 ms). At dev
scale the startup dominates; at larger scales the work dominates.

**Output cross-validation**: at dev scale all three targets produce
identical effective_distance values (e.g.
`Bose-Einstein_statistics → Physics → 0.999179`). Confirms the
Elixir CategoryAncestor kernel (PR #1803) faithfully implements
the same path-enumeration semantics as Rust's
`collect_native_category_ancestor_hops`.

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
- **Elixir kernel scales poorly on path-enumeration**. It IS the
  faithful path-enumeration semantics — same algorithm Rust uses —
  but BEAM-interpreted recursion with per-call list copies for the
  visited list is much slower than either Prolog's WAM or Rust's
  native code on the same algorithm.

## Why the kernel still matters for Elixir

The earlier benchmark (`benchmarks/wam_elixir_graph_kernels.md`)
measured kernel-Elixir vs **WAM-Elixir-emitted** `tc/2` and got
982× at N=1000. This benchmark compares against **other targets'
implementations**, which is a different yardstick:

- vs WAM-Elixir-emitted: kernel ≈ 1000× faster (chain graph, n=1000)
- vs Prolog-direct: kernel ≈ 0.01× as fast (effective_distance, 1k+ nodes)
- vs Rust-native-kernel: kernel ≈ 0.005× as fast (10k)

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
1. Correct path-enumeration semantics (cross-validated vs Rust).
2. Compositional with FactSource adaptors (ETS today, LMDB once
   `:elmdb` is installable).
3. Massive speedup over WAM-emitted recursive Prolog (982× at N=1000
   on chain graph).
4. Pattern-recognition auto-routing: user writes the canonical
   shape, the kernel fires.

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
