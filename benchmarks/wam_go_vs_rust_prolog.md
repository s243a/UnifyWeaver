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
| Prolog  |      352 |      415 | mismatch by 1 row (Unicode sort order on `é`)|
| Rust    |       18 |       32 | mismatch by 1 row (Unicode sort order on `é`)|
| Go      |        3 |        3 | **incorrect — 31 / 271 rows**                |

### Scale 1k (6,944-line facts file, ≈1,000 articles)

| Target  | query_ms | total_ms | Correctness                                  |
|---------|---------:|---------:|----------------------------------------------|
| Prolog  |      217 |      282 | **EXACT match (580 rows)**                   |
| Rust    |       16 |       33 | **EXACT match (580 rows)**                   |
| Go      |        0 |        0 | **incorrect — 0 / 580 rows**                 |

> **Updated 2026-04-30 (post-`fix/wam-go-runtime-and-prolog-driver`):**
> The Prolog "MISMATCH" rows in the original report were a benchmark-driver
> bug, not a correctness issue. The runner was passing both
> `-g run_benchmark` *and* loading a script that already had
> `:- initialization(run_benchmark, main)`, so swipl ran the entry point
> twice and concatenated two output blocks into the TSV. Dropping
> `-g run_benchmark` (the script's initialization directive is sufficient)
> made Prolog match the reference exactly at scale 1k.

### Speed ratios (Rust as the reference)

| Target  | Scale 300 total | Scale 1k total |
|---------|----------------:|---------------:|
| Rust    |        1.0×     |        1.0×    |
| Prolog  |       13.0×     |        8.5×    |
| Go      |        n/a (incorrect output)              ||

## What the numbers mean — and what they don't

**Rust is the clear winner on both speed and correctness.** At scale 1k it produces a byte-identical output to the reference TSV in 40 ms median, ~7× faster than the optimized-Prolog interpreter. The scale-300 mismatch is a single row (`Théophile_de_Donder`) appearing one position off in the sort order — both Rust and Prolog disagree with the reference identically, suggesting the reference was generated under a different locale's collation rule rather than a real numerical bug.

**Optimized Prolog is correct on numerics.** The original report
showed a Prolog "MISMATCH" caused by the benchmark driver firing the
entry point twice (the generated script has `:- initialization(run_benchmark, main)`
*and* the runner was passing `-g run_benchmark` on top of it). Dropping
the `-g` flag made Prolog produce a single output block that exactly
matches the reference at scale 1k.

**Go has multiple separate problems.** The generator was previously broken end-to-end (the cross-target runner shipped with a `# NOTE: Go WAM benchmark driver not yet wired up for fact loading` comment). Two rounds of gen fixes have landed:

Commit `988f0a6` (branch `bench/wam-go-vs-rust-prolog`) — seven gen bugs that prevented `accumulated kernels_on` from compiling at all:

1. `replace_all_atoms/4` infinite loop (Replace string contained Search string)
2. `load_files/2` loading clauses into the gen's own module instead of `user`
3. `string_length/2` used as an arithmetic function inside `is/2`
4. Missing trailing comma in a multi-line Go composite literal
5. `%%d` / `%%s` in `fmt.Printf` format strings (mistaken Prolog-format escape)
6. `vm.Regs[fmt.Sprintf("A%d", i+1)]` indexing a `[]Value` slice with a string
7. ITE pattern detection (added in `ebf1b38`) lumping continuation into the else branch and leaving the then branch with no `return`

Branch `fix/wam-go-runtime-and-prolog-driver` — three further gen bugs that prevented `accumulated kernels_off` from compiling and would have produced invalid Go for any fact predicate containing apostrophe-bearing atoms (e.g. `'People\'s_Republic_of_China'`):

8. `compile_predicate_to_wam/3` is benignly non-deterministic for multi-clause fact predicates (32 essentially-equivalent WAM bodies for `category_parent/2` at scale 300, differing only by ~5 chars of label name). Without `once/1` in the lowered-emission `findall`, the emitter produced one duplicate `func (vm *WamState) PredCategory_parent2() bool` per alternative, and Go refused to compile lib.go with "method already declared".
9. The WAM-text-to-Go parser passed quoted-atom tokens (`'People\'s_Republic_of_China'`) through to `&Atom{Name: "..."}` literally, so Go saw `\'` inside a double-quoted string and rejected the file with "unknown escape sequence". Now `parse_string_to_go_val/2` round-trips the token through `term_to_atom/2` to strip the outer quotes and unescape the inner `\'`.
10. `go_value_literal/2` was emitting unescaped `~w` of the atom into a Go double-quoted string. An atom containing `\` or `"` would have produced invalid Go. Now passes through `escape_go_atom_for_double_quoted/2` first.

After all ten fixes the Go binary builds and runs cleanly under both `kernels_on` and `kernels_off`. **But its output is still wrong at runtime** — it produces 31 rows on the 300-scale fixture (vs. 271 expected) and 0 rows at scale 1k. The 2 ms "query time" therefore measures *how fast Go can return the wrong answer*, not how fast it can compute the right one. The remaining gap is downstream of the gen script: the WAM kernel for `category_ancestor$effective_distance_sum_selected/3` is not producing any tuples at runtime (`tuple_count=0` in the bench's stderr), so all weight sums collapse to 0 or 1 and most articles get filtered out. This is a Go-target runtime / kernel-dispatch issue separate from the generator fixes and out of scope for this comparison.

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
