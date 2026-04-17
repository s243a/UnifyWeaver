# Benchmark Target Matrix

## Goal

Benchmark comparisons need to separate execution families that look similar at the command line but are not the same compilation path.

The important distinction is whether a target is:

- optimized Prolog transpilation
- hybrid WAM execution
- direct target-language pipeline generation
- a query engine

If those are mixed without labeling, the numbers are hard to interpret.

## Current Effective-Distance Paths

### Haskell

The Haskell effective-distance benchmark can start from optimized Prolog.

Path:

1. Load the workload Prolog.
2. Run `prolog_target` optimization passes.
3. Load the generated optimized predicates back into Prolog.
4. Emit a WAM Haskell project with `write_wam_haskell_project/3`.

That means Haskell is not "always WAM first, then Haskell" in a naive sense. The benchmark can start from optimized Prolog, and WAM remains the lowering substrate and fallback path.

The relevant modes are:

- `interpreter + no_kernels(true)` -> pure interpreter baseline
- `interpreter + kernels enabled` -> hybrid WAM + FFI
- `functions + no_kernels(true)` -> lowered-only
- `functions + kernels enabled` -> lowered functions with WAM fallback + FFI

### Rust

There are two materially different Rust benchmark paths today.

Direct pipeline path:

1. `generate_pipeline.py`
2. Emit a direct Rust executable

This is not the same thing as compiling optimized Prolog into a WAM Rust project.

Hybrid WAM Rust path:

1. Load the workload Prolog.
2. Optionally run `prolog_target` optimization passes.
3. Compile selected predicates to WAM.
4. Emit a merged-code Rust WAM benchmark project.

So Rust already has both:

- a direct pipeline benchmark path
- a hybrid WAM benchmark path

### Go

The current effective-distance benchmark surface for Go is the direct pipeline path from `generate_pipeline.py`.

That is useful for comparison, but it is not yet the same kind of optimized-Prolog-to-hybrid-WAM benchmark path that exists for Haskell and WAM Rust.

### C#

The C# query runtime is still a useful comparison point because it is heavily optimized, but it belongs in its own category.

It should not be treated as just another transpiled target.

## Target Categories

The new matrix script uses these categories:

- `optimized-prolog`
- `hybrid-wam`
- `direct-pipeline`
- `query-engine`

The default presets are:

- `portable-default`
- `desktop-default`
- `optimized-prolog`
- `hybrid-wam`
- `direct-pipeline`
- `query-engine`
- `all`

## Termux Rule

On Termux, the default set excludes C#.

Reason:

- running C# through `proot` Debian adds an environment penalty
- that penalty is not part of the query engine itself
- including it in default local comparisons would bias the numbers

C# stays available in the harness as an explicit opt-in target for environments where it can run natively.

## Scripts

New scripts:

- `examples/benchmark/benchmark_effective_distance_matrix.py`
- `examples/benchmark/generate_wam_haskell_matrix_benchmark.pl`

Examples:

```bash
python examples/benchmark/benchmark_effective_distance_matrix.py --list-targets
```

```bash
python examples/benchmark/benchmark_effective_distance_matrix.py --target-sets optimized-prolog,hybrid-wam --scales 300,1k
```

```bash
python examples/benchmark/benchmark_effective_distance_matrix.py --targets prolog-accumulated,wam-rust-accumulated,haskell-lowered-ffi
```

```bash
python examples/benchmark/benchmark_effective_distance_matrix.py --target-sets portable-default --include-targets csharp-query
```
