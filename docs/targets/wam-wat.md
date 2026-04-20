<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2025-2026 John William Creighton (@s243a)
-->
# WAM-WAT Target Overview (`target(wam_wat)`)

The WAM-WAT target compiles Prolog predicates to WebAssembly
(`.wasm` via `.wat`), producing self-contained modules that
interpret a WAM bytecode stream at runtime. Unlike the other
targets documented here — which emit source code for a specific
host language — WAM-WAT emits a compact binary artifact that
runs anywhere with a WebAssembly runtime: the browser, Node.js,
Wasmtime, Wasmer, or other embedders.

For implementation and architecture details, see
[`docs/design/WAM_WAT_ARCHITECTURE.md`](../design/WAM_WAT_ARCHITECTURE.md).

## Design Highlights

- **WAM-first pipeline**: Predicates compile to a textual WAM
  intermediate form via `wam_target.pl`, then to a
  bytecode-oriented `.wat` module via `wam_wat_target.pl`. The
  two layers share the instruction vocabulary with the broader
  WAM family (Go, Rust, LLVM, ILAsm, Elixir, JVM backends).
- **73-instruction bytecode**: Includes the base WAM instruction
  set plus fused instructions (arg/3 fast paths, tail-call
  collapses, clause-end combinations) and first-argument
  indexing. The fused instructions are produced by an 8-pass
  peephole optimizer during compilation.
- **Self-contained output**: A single `.wat` file is emitted.
  Running `wat2wasm bench_suite.wat -o bench_suite.wasm`
  produces a `.wasm` module with no external dependencies beyond
  three small host imports (`print_i64`, `print_char`,
  `print_newline`) used only for optional output.
- **V8 JIT-friendly dispatch**: The runtime uses a `br_table`
  dispatch in a single `$step` function that returns after each
  instruction, paired with an outer `$run_loop` driver. V8's
  JIT inlines and tiers this pattern effectively.
- **Structured control flow throughout**: No indirect control
  flow primitives are used beyond `br_table` — every dispatch
  is via a tagged instruction that the peephole layer has
  already specialized.

## Strengths

- **Portable**: The compiled `.wasm` runs in any wasm runtime.
  Ideal for embedding Prolog-derived logic in browsers, sandboxed
  environments, or cross-platform applications.
- **Predictable performance**: Benchmarks on the bundled
  `bench_suite.wasm` (13 workloads covering arithmetic, unify,
  term inspection, tree walking, and fib) show competitive times
  vs SWI-Prolog on Termux/V8, with ~55% cumulative improvement
  on term-walking workloads since the Phase 6 baseline (April
  2026).
- **Deterministic binary output**: The compiled module is a
  pure function of the Prolog source plus the (versioned)
  peephole pipeline. No host-toolchain variability.
- **No runtime unification explosion**: The trail-aware
  `neck_cut_test` and first-arg `type_dispatch_a1` eliminate
  choice-point pushes for common cut-deterministic and
  type-dispatched patterns.

## Limitations

- **Interpreter, not native code**: The runtime still executes
  the WAM bytecode via a dispatch loop — not a transpile-to-host
  like the Go/Rust/C# targets. Faster than naive WAM
  implementations thanks to aggressive instruction fusion, but
  the dispatch cost is still the dominant hot-loop overhead
  (~50% of `bench_sum_big`).
- **Limited to the WAM subset**: Features outside the WAM
  intermediate form (e.g., the full query runtime, aggregate
  semantics) aren't available through this target. Use one of
  the higher-level targets (Go, C# query runtime, SQL) for
  those workloads.
- **WASM runtime required**: The output needs a V8 / Wasmtime /
  Wasmer / equivalent to run. For pipeline integration on a
  bare Unix box without one, the Bash target is more convenient.
- **Peephole coverage is bench-driven**: The fused instructions
  were motivated by specific bench workload patterns (sum-over-
  compound, term depth, fib). Predicates with very different
  structure will hit fewer optimized paths and may be closer
  to the unoptimized baseline.

## Usage

```prolog
% Compile a list of predicates to a WAM-WAT module
:- use_module('src/unifyweaver/targets/wam_wat_target').

?- write_wam_wat_project(
       [bench_suite:sum_ints/3,
        bench_suite:term_depth/2],
       [module_name(my_module)],
       'output.wat').
```

Then:

```bash
wat2wasm output.wat -o output.wasm

# In Node.js:
node -e '
  const {instance} = await WebAssembly.instantiate(
    require("fs").readFileSync("output.wasm"),
    {env: {print_i64:()=>{}, print_char:()=>{}, print_newline:()=>{}}}
  );
  console.log(instance.exports.sum_ints_3(...));
'
```

### Benchmark suite

The bundled benchmark lives at
`examples/wam_term_builtins_bench/`. Regenerate and run with:

```bash
cd examples/wam_term_builtins_bench
swipl ../../src/unifyweaver/targets/wam_wat_target.pl  # (sanity check)
swipl generate_wat_bench.pl                             # regenerate bench_suite.wat
wat2wasm bench_suite.wat -o bench_suite.wasm
node run_bench.js bench_suite.wasm                      # runs all 13 workloads
```

For single-workload focused profiling:

```bash
node --cpu-prof profile_one.js bench_suite.wasm bench_sum_big_0 300000
```

## Relation to the WAM family

UnifyWeaver has several WAM-based targets that share the same
WAM text IR via `wam_target.pl`:

| Backend | Host output | Use case |
|---------|-------------|----------|
| `wam_go_target` | Go source | Compiled native binaries |
| `wam_rust_target` | Rust source | Memory-safe native binaries |
| `wam_llvm_target` | LLVM IR | Cross-platform native code |
| `wam_ilasm_target` | CIL assembly | .NET runtime |
| `wam_elixir_target` | Elixir source | BEAM VM / distributed apps |
| `wam_jvm_target` | JVM bytecode | Java runtime (Jamaica/Krakatau) |
| **`wam_wat_target`** | **WebAssembly** | **Browser / cross-platform runtime** |

The WAM-WAT peephole optimizations (first-arg tag dispatch,
neck-cut-test, tail-call fusion, clause-end fusion) are currently
WAM-WAT-specific; moving them to the shared `wam_target.pl` layer
so other backends benefit is a listed follow-up in the
architecture doc.

## See Also

- [`docs/design/WAM_WAT_ARCHITECTURE.md`](../design/WAM_WAT_ARCHITECTURE.md)
  — pipeline, instruction set, runtime data structures,
  optimization family walkthroughs, performance trajectory.
- [`docs/design/WAM_TERM_BUILTINS_PHASE_6_PERF.md`](../design/WAM_TERM_BUILTINS_PHASE_6_PERF.md)
  — historical baseline with SWI-Prolog comparisons before the
  recent optimization work.
- [`examples/wam_term_builtins_bench/`](../../examples/wam_term_builtins_bench/)
  — 13-workload benchmark suite used for A/B measurement.
