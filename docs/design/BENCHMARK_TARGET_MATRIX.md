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

Rust now has a hybrid WAM benchmark path with full support for the lowered emitter
(`emit_mode(functions)`) and Rayon-based parallel execution (`parallel(true)`)
via `run_parallel` using `rayon::par_iter` (behind `#[cfg(feature = "parallel")]`).

Performance optimizations:

- Register Vec: `Vec<Value>` with encoded indices (A1→0, X1→100, Y1→200) instead of `HashMap<String, Value>`
- Lowered emitter: deterministic predicates compiled to plain Rust functions
- Rayon parallel: `par_iter` for intra-query parallelism (optional feature gate)

Direct pipeline path:

1. `generate_pipeline.py`
2. Emit a direct Rust executable

This is not the same thing as compiling optimized Prolog into a WAM Rust project.

Hybrid WAM Rust path:

1. Load the workload Prolog and benchmark facts.
2. Run `prolog_target` optimization passes.
3. Force the selected predicates through the shared Rust WAM path.
4. Emit a Rust benchmark driver that queries the compiled VM directly.

The relevant modes are (mirroring Haskell/Go/Python/F#):

- `interpreter + no_kernels(true)` -> pure interpreter baseline
- `interpreter + kernels enabled` -> hybrid WAM + FFI
- `functions + no_kernels(true)` -> lowered-only (deterministic predicate-as-function)
- `functions + kernels enabled` -> lowered functions with WAM fallback + FFI

Optimized benchmark pipeline (mirrors Go, Haskell, Python, F#):

1. Load the workload Prolog.
2. Run `prolog_target` optimization passes (seeded accumulation).
3. Load the generated optimized predicates back into Prolog.
4. Emit a WAM Rust project with `write_wam_rust_project/3` using
   `emit_mode(functions)` and `parallel(true)`.

Script: `examples/benchmark/generate_wam_rust_optimized_benchmark.pl`

Profiling matrix configs (Q-T) in `gen_prof_matrix.pl`:

- Q: `rust-pure-interp` — interpreter + no_kernels
- R: `rust-interp-ffi` — interpreter + kernels
- S: `rust-lowered-only` — functions + no_kernels
- T: `rust-lowered-ffi` — functions + kernels

That means Rust can now participate in:

- `direct-pipeline`
- `hybrid-wam`

### Go

Go now has two effective-distance benchmark paths, with full support for the
lowered emitter (`emit_mode(functions)`) and goroutine-based parallel execution
(`parallel(true)`) via the package-level `RunParallel` on `WamContext`.

Direct pipeline path:

1. `generate_pipeline.py`
2. Emit a direct Go executable

Hybrid WAM Go path:

1. Load the workload Prolog and benchmark facts.
2. Run `prolog_target` optimization passes.
3. Force the selected predicates through the shared Go WAM path.
4. Emit a Go benchmark driver that queries the compiled VM directly.

The relevant modes are (mirroring Haskell):

- `interpreter + no_kernels(true)` -> pure interpreter baseline
- `interpreter + kernels enabled` -> hybrid WAM + FFI
- `functions + no_kernels(true)` -> lowered-only (deterministic predicate-as-function)
- `functions + kernels enabled` -> lowered functions with WAM fallback + FFI

Optimized benchmark pipeline (mirrors `generate_wam_haskell_optimized_benchmark.pl`):

1. Load the workload Prolog.
2. Run `prolog_target` optimization passes (seeded accumulation).
3. Load the generated optimized predicates back into Prolog.
4. Emit a WAM Go project with `write_wam_go_project/3` using
   `emit_mode(functions)` and `parallel(true)`.

Script: `examples/benchmark/generate_wam_go_optimized_benchmark.pl`

That means Go can now participate in:

- `direct-pipeline`
- `hybrid-wam`

### Python

Python now has a hybrid WAM benchmark path with support for the lowered emitter
(`emit_mode(functions)`) and process-based parallel execution (`parallel(true)`)
via `run_parallel` in `WamRuntime.py` (using `ProcessPoolExecutor` to bypass the GIL).

Hybrid WAM Python path:

1. Load the workload Prolog and benchmark facts.
2. Run `prolog_target` optimization passes.
3. Force the selected predicates through the shared Python WAM path.
4. Emit a Python benchmark driver that queries the compiled VM directly.

The relevant modes are (mirroring Haskell/Go):

- `interpreter + no_kernels(true)` -> pure interpreter baseline
- `interpreter + kernels enabled` -> hybrid WAM + FFI
- `functions + no_kernels(true)` -> lowered-only (deterministic predicate-as-function)
- `functions + kernels enabled` -> lowered functions with WAM fallback + FFI

Optimized benchmark pipeline (mirrors Go and Haskell):

1. Load the workload Prolog.
2. Run `prolog_target` optimization passes (seeded accumulation).
3. Load the generated optimized predicates back into Prolog.
4. Emit a WAM Python project with `write_wam_python_project/3` using
   `emit_mode(functions)` and `parallel(true)`.

Script: `examples/benchmark/generate_wam_python_optimized_benchmark.pl`

That means Python can now participate in:

- `hybrid-wam`

### F#

F# now has a hybrid WAM benchmark path with full support for the lowered emitter
(`emit_mode(functions)`) and TPL-based parallel execution (`parallel(true)`)
via `runParallel` using `Array.Parallel.map`.

Performance optimizations (closing gaps vs Haskell/Go/Rust):

- Register array: `Value array` (O(1) access) instead of `Map<int,Value>`
- Binary search for `SwitchOnConstantPc`: sorted `(string * int) array` with binary search
- FFI-owned fact skip: predicates handled entirely by FFI kernel path are excluded from WAM compilation
- Atom interning: `buildAtomInternTable` populates `WcAtomIntern`/`WcAtomDeintern` at startup
- Real parallel execution: `Array.Parallel.map` (TPL) for intra-query parallelism

Hybrid WAM F# path:

1. Load the workload Prolog and benchmark facts.
2. Run `prolog_target` optimization passes.
3. Force the selected predicates through the shared F# WAM path.
4. Emit an F# benchmark driver that queries the compiled VM directly.

The relevant modes are (mirroring Haskell/Go):

- `interpreter + no_kernels(true)` -> pure interpreter baseline
- `interpreter + kernels enabled` -> hybrid WAM + FFI
- `functions + no_kernels(true)` -> lowered-only (deterministic predicate-as-function)
- `functions + kernels enabled` -> lowered functions with WAM fallback + FFI

Optimized benchmark pipeline (mirrors Go and Haskell):

1. Load the workload Prolog.
2. Run `prolog_target` optimization passes (seeded accumulation).
3. Load the generated optimized predicates back into Prolog.
4. Emit a WAM F# project with `write_wam_fsharp_project/3` using
   `emit_mode(functions)` and `parallel(true)`.

Script: `examples/benchmark/generate_wam_fsharp_optimized_benchmark.pl`

Profiling matrix configs (M-P) in `gen_prof_matrix.pl`:

- M: `fsharp-pure-interp` — interpreter + no_kernels
- N: `fsharp-interp-ffi` — interpreter + kernels
- O: `fsharp-lowered-only` — functions + no_kernels
- P: `fsharp-lowered-ffi` — functions + kernels

That means F# can now participate in:

- `hybrid-wam`

### C#

The C# query runtime is still a useful comparison point because it is heavily optimized, but it belongs in its own category.

It should not be treated as just another transpiled target.

## Target Categories

The new matrix script uses these categories:

- `optimized-prolog`
- `hybrid-wam`
- `hybrid-wam-scaffold`
- `direct-pipeline`
- `query-engine`

The default presets are:

- `termux-smoke`
- `portable-default`
- `desktop-default`
- `optimized-prolog`
- `hybrid-wam`
- `clojure-wam`
- `clojure-wam-scaffold`
- `direct-pipeline`
- `query-engine`
- `all`

`hybrid-wam-scaffold` is for generated targets that have benchmark-generation
shape and smoke coverage but do not yet emit the common effective-distance
result table. Clojure WAM is split between one executable target and the
remaining scaffold targets:

- `clojure-wam-accumulated` — executable `hybrid-wam`; emits the common result
  table and matches `prolog-accumulated` on the `dev` scale

- `clojure-wam-seeded`
- `clojure-wam-seeded-no-kernels`
- `clojure-wam-accumulated-no-kernels`

The effective-distance matrix lists and resolves all of these targets, but
skips the scaffold-only modes with an explicit message when a result-producing
benchmark run is requested. This avoids comparing a predicate-level smoke path
against full effective-distance table producers.

## Termux Rule

On Termux, the default local mode is intentionally smaller:

- default target set: `termux-smoke`
- default scales: `dev,10x`

Reason:

- larger effective-distance runs can destabilize the Termux session
- running C# through `proot` Debian adds an environment penalty
- that penalty is not part of the query engine itself
- including it in default local comparisons would bias the numbers

So:

- C# stays available as an explicit opt-in target for native desktop environments
- larger benchmark scales should be run outside Termux
- Termux should be treated as a smoke-test environment for benchmark correctness, not the primary profiling environment
- the matrix script rejects larger Termux scales unless `--allow-large-termux-scales` is passed explicitly

## Scripts

New scripts:

- `examples/benchmark/benchmark_effective_distance_matrix.py`
- `examples/benchmark/generate_wam_haskell_matrix_benchmark.pl`
- `examples/benchmark/generate_wam_go_effective_distance_benchmark.pl`
- `examples/benchmark/generate_wam_go_optimized_benchmark.pl`
- `examples/benchmark/generate_wam_python_optimized_benchmark.pl`
- `examples/benchmark/generate_wam_fsharp_optimized_benchmark.pl`
- `examples/benchmark/generate_wam_rust_optimized_benchmark.pl`
- `examples/benchmark/generate_wam_clojure_optimized_benchmark.pl`
- `tests/bench_wam_fsharp.pl` (F# compilation throughput benchmarks)
- `tests/bench_wam_rust.pl` (Rust compilation throughput benchmarks)
- `examples/benchmark/gen_prof_matrix.pl` (4 Haskell + 4 Go + 4 Python + 4 F# + 4 Rust profiling configs)

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
