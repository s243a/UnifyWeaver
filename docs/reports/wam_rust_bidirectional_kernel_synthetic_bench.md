# WAM Rust: bidirectional ancestor kernel — first benchmark (synthetic 1k/10k)

Date: 2026-06-12. Follows the P1 port of the F# bidirectional kernel to
the Rust hybrid-WAM target (see CHANGELOG "bidirectional ancestor
kernel (F# parity port)" and `docs/handoff/rust_fsharp_parity_campaign.md`).

## Setup

- Generated two crates from the same kernel-shaped `category_ancestor/4`
  source (max_depth 10): default emission (upward-only kernel) and
  `kernel_mode(bidirectional)` (5-ary kernel, costs 1.0/3.0, budget 10.0).
- Harness: `templates/targets/rust_wam/main.rs.mustache` effective-distance
  workload — eager in-memory edges, demand-filtered to root-reachable,
  seeds = distinct article categories. The bidirectional branch hoists
  calibration out of the seed loop (graph and root are loop constants)
  and applies the F# direction-weighted power mean
  `w = (1/D)^parentHops · (1/(b·D))^childHops`,
  per-seed weight `Σ(w·(total+1)^-n)/Σw`.
- Fixtures: `data/benchmark/1k` (5,933 edges) and `data/benchmark/10k`
  (25,227 edges), root `Physics`. Single run each (process-level
  variance at these scales is small; rerun for publication numbers).
- `cargo build --release`, container CPU.

## Results

| Variant | Scale | query_ms | total_ms | seed_count | tuple_count | D | b |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| upward-only | 1k | 5 | 14 | 89 | 48 | — | — |
| bidirectional | 1k | 106 | 113 | 89 | 48 | 2.4377 | 1.2977 |
| upward-only | 10k | 49 | 87 | 888 | 462 | — | — |
| bidirectional | 10k | 319 | 356 | 888 | 462 | 2.8963 | 2.6559 |

## Reading

- **Correctness signal**: `tuple_count` (seeds with a root-reaching
  path) is identical between variants at both scales — the
  bidirectional kernel finds at least the pure-parent paths, and the
  extra child-detour paths change weights, not reachability.
- **Cost shape**: ~6.5× query time over upward-only at 10k. Expected —
  the kernel enumerates every budget-feasible path in both directions
  (childCost 3.0 admits up to 3 child hops within budget 10), which is
  the price of the direction-weighted metric. The F# simplewiki report
  (`bidirectional_kernel_simplewiki_benchmark.md`) has the same shape.
- **Calibration is hoisted once** per (graph, root) in the harness;
  the WAM-dispatch path (`execute_foreign_predicate`) still calibrates
  per call — fine for single queries, not for hot WAM-routed loops.
  A state-level calibration cache keyed by (edge_pred, child_pred,
  root), invalidated on fact registration (the C target's shape), is
  the known follow-up if that path becomes hot.

## Next steps (campaign ladder)

1. Simplewiki-scale run (needs the dump ingested — fixture is
   gitignored; recovery procedure in
   `docs/handoff/wam_rust_simplewiki_blocker.md`), comparing against
   the F# numbers in `bidirectional_kernel_simplewiki_benchmark.md`.
2. Reverse-CSR child index (the C target has a reader to model on;
   today Rust derives the child index by reversing the eager parent
   table or reading LMDB `category_child`).
3. Builtins parity sweep (succ/2, between/3, sort/msort, atom/string
   ops, term-order ops, catch/throw).
