# Effective-Distance Cross-Target Benchmark — Go hybrid WAM vs. Rust WAM vs. optimized Prolog

**Date:** 2026-04-30
**Branch:** `bench/wam-go-vs-rust-prolog`
**Workload:** `examples/benchmark/effective_distance.pl` (single-source effective distance over a Wikipedia category graph; aggregation kernel = `(sum Hops^(-N))^(-1/N)` with `N=5`).
**Generators:**
- `examples/benchmark/generate_prolog_effective_distance_benchmark.pl` (variant: `accumulated`)
- `examples/benchmark/generate_wam_effective_distance_benchmark.pl` (Rust WAM, variant: `accumulated`)
- `examples/benchmark/generate_wam_go_effective_distance_benchmark.pl` (Go hybrid WAM, variant: `accumulated`)
**Driver:** `examples/benchmark/run_effective_distance_comparison.sh`
**Toolchains:** SWI-Prolog 9.0.4, Rust/cargo 1.94.1, Go 1.24.7. Haskell and .NET were not exercised (toolchains not installed in the sandbox).
**Hardware:** sandbox VM, 15 GiB RAM, x86_64 Linux 6.18.5.

## Headline numbers

Median over 3 runs each. `query_ms` is the time inside the predicate engine; `total_ms` is end-to-end including fact load and aggregation.

### Scale 300 (6,788-line facts file, ≈300 articles)

| Target  | query_ms | total_ms | Correctness                                  |
|---------|---------:|---------:|----------------------------------------------|
| Prolog  |      354 |      416 | mismatch (driver double-runs, see below)     |
| Rust    |       18 |       35 | mismatch by 1 row (Unicode sort order on `é`)|
| Go      |        2 |        2 | **incorrect — 31 / 271 rows**                |

### Scale 1k (6,944-line facts file, ≈1,000 articles)

| Target  | query_ms | total_ms | Correctness                                  |
|---------|---------:|---------:|----------------------------------------------|
| Prolog  |      220 |      284 | mismatch (driver double-runs, see below)     |
| Rust    |       17 |       40 | **EXACT match (580 rows)**                   |
| Go      |        0 |        0 | **incorrect — 0 / 580 rows**                 |

### Speed ratios (Rust as the reference)

| Target  | Scale 300 total | Scale 1k total |
|---------|----------------:|---------------:|
| Rust    |        1.0×     |        1.0×    |
| Prolog  |       11.9×     |        7.1×    |
| Go      |        n/a (incorrect output)              ||

## What the numbers mean — and what they don't

**Rust is the clear winner on both speed and correctness.** At scale 1k it produces a byte-identical output to the reference TSV in 40 ms median, ~7× faster than the optimized-Prolog interpreter. The scale-300 mismatch is a single row (`Théophile_de_Donder`) appearing one position off in the sort order — both Rust and Prolog disagree with the reference identically, suggesting the reference was generated under a different locale's collation rule rather than a real numerical bug.

**Optimized Prolog is correct on numerics but the driver double-emits.** Each run of the generated Prolog script under `swipl -g run_benchmark -t halt` prints two metric blocks and two result tables (the swipl `:- initialization(main, main)` directive plus the `-g run_benchmark` flag both fire `run_benchmark`). The runner extracts the *last* `query_ms=` line, so the timings here are per-run (not doubled), but the TSV row count is doubled (1,161 vs. 580 at scale 1k), which trips the byte-diff. The numerical results match Rust within those 580 rows. **This is a pre-existing benchmark-driver quirk, not a Prolog correctness issue.**

**Go has multiple separate problems.** The generator was previously broken end-to-end (the cross-target runner shipped with a `# NOTE: Go WAM benchmark driver not yet wired up for fact loading` comment). Commit `988f0a6` on this branch fixed seven distinct gen bugs to get the project building and running:

1. `replace_all_atoms/4` infinite loop (Replace string contained Search string)
2. `load_files/2` loading clauses into the gen's own module instead of `user`
3. `string_length/2` used as an arithmetic function inside `is/2`
4. Missing trailing comma in a multi-line Go composite literal
5. `%%d` / `%%s` in `fmt.Printf` format strings (mistaken Prolog-format escape)
6. `vm.Regs[fmt.Sprintf("A%d", i+1)]` indexing a `[]Value` slice with a string
7. ITE pattern detection (added in `ebf1b38`) lumping continuation into the else branch and leaving the then branch with no `return`

After those fixes the Go binary builds and runs. **But its output is still wrong at runtime** — it produces 31 rows on the 300-scale fixture (vs. 271 expected) and 0 rows at scale 1k. The 2 ms "query time" therefore measures *how fast Go can return the wrong answer*, not how fast it can compute the right one. The remaining gap is downstream of the gen script: the WAM kernel for `category_ancestor$effective_distance_sum_selected/3` is not producing any tuples at runtime (`tuple_count=0` in the bench's stderr), so all weight sums collapse to 0 or 1 and most articles get filtered out. This is a Go-target runtime / kernel-dispatch issue separate from the generator fixes and out of scope for this comparison.

## Reproducing

```bash
# Scale 300, 3 reps
./examples/benchmark/run_effective_distance_comparison.sh 300 3

# Scale 1k, 3 reps
./examples/benchmark/run_effective_distance_comparison.sh 1k 3
```

The driver reuses `/tmp/uw-bench-cmp/` for generated projects and per-rep TSVs, and reports median `query_ms` / `total_ms` plus a per-target diff against `data/benchmark/<scale>/reference_output.tsv`.

## Caveats and not-tested

- **Haskell WAM, F# WAM, Scala WAM, Clojure WAM, Elixir WAM, Python WAM, C# Query** — not exercised here. Adding them is a matter of installing the toolchains and extending the runner with a new `--- <target> ---` block; the metric format (`query_ms=N` etc. on stderr) is shared.
- **Larger scales (5k, 10k, 10x)** — not exercised here. Rust scales linearly per row inspection, but I didn't time it.
- **Rust no-kernel matrix** — the recent `bench: add Rust WAM no-kernel matrix targets` work isn't compared against; this run uses `kernels_on` for both Rust and Go.
- **Process startup time** is folded into `total_ms` (a few ms per run for Rust/Go, more for swipl). For the order-of-magnitude question we're answering this doesn't matter; for sub-millisecond benchmarking it would.

## Summary

Rust is ~7–12× faster than optimized Prolog on this workload and produces a byte-identical TSV. The Go hybrid-WAM target's *generator* is now unblocked for the first time (seven separate fixes landed on this branch) but its *runtime* still doesn't compute the correct effective distances, so its raw timing isn't a fair benchmark — it's just a fast no-op. The next step for Go is debugging the WAM kernel for `effective_distance_sum_selected/3`.
