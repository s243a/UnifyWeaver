# WAM Scala Target — Usage Guide

The WAM Scala target compiles Prolog predicates to a self-contained
SBT project that runs on the JVM. It implements a hybrid WAM:
predicates and queries are interpreted by a stepping engine, but the
instruction array, label maps, intern table, and dispatch tables are
all compiled at codegen time and shared across queries.

This document is a getting-started reference for using the target.
For the high-level design, see:

- [WAM_SCALA_HYBRID_SPEC.md](proposals/WAM_SCALA_HYBRID_SPEC.md) — runtime data shape
- [WAM_SCALA_HYBRID_PHILOSOPHY.md](proposals/WAM_SCALA_HYBRID_PHILOSOPHY.md) — what's mutable vs immutable
- [WAM_SCALA_HYBRID_IMPL_PLAN.md](proposals/WAM_SCALA_HYBRID_IMPL_PLAN.md) — phase plan

## Quick start

```prolog
:- use_module('src/unifyweaver/targets/wam_scala_target').

:- dynamic user:greet/1.
user:greet(world).
user:greet(scala).

:- initialization((
    write_wam_scala_project(
        [user:greet/1],
        [ package('demo.greet'),
          runtime_package('demo.greet'),
          module_name('greet-demo'),
          intern_atoms([world, scala])
        ],
        '/tmp/greet_demo'),
    halt
)).
```

After running this script:

```bash
cd /tmp/greet_demo
mkdir classes
scalac -d classes src/main/scala/demo/greet/*.scala
scala -classpath classes demo.greet.GeneratedProgram 'greet/1' world
# → true
scala -classpath classes demo.greet.GeneratedProgram 'greet/1' mars
# → false
```

A worked end-to-end example with a recursive ancestor relation lives
at [examples/wam_scala_demo/](../examples/wam_scala_demo/).

## API

### `write_wam_scala_project(+Predicates, +Options, +ProjectDir)`

Generates a complete Scala project for a list of Prolog predicates.

**Predicates** — list of `Module:Pred/Arity` or `Pred/Arity` indicators.
Predicates that aren't declared as dynamic must be defined in the
calling file.

**ProjectDir** — output directory (created if needed). Layout:

```
<ProjectDir>/
├── build.sbt
├── project/
│   └── build.properties
└── src/main/scala/<package-path>/
    ├── WamRuntime.scala      ← stepping engine, ADTs, helpers
    └── GeneratedProgram.scala ← intern table, instructions, dispatch, main
```

**Options**:

| Option | Meaning |
|---|---|
| `package(Pkg)` | Scala package for `GeneratedProgram` (default `generated.wam_scala.core`). |
| `runtime_package(Pkg)` | Package for `WamRuntime` (typically same as `package`). |
| `module_name(Name)` | SBT `name :=` value. |
| `foreign_predicates([P/A, ...])` | These predicates' WAM bodies are replaced by a `CallForeign` stub. |
| `scala_foreign_handlers([handler(P/A, "<scala expr>"), ...])` | Inline Scala source for each foreign handler. |
| `scala_fact_sources([source(P/A, Spec), ...])` | Declarative fact-source — auto-expands to `foreign_predicates` + `scala_foreign_handlers` + `intern_atoms`. See below. |
| `intern_atoms([atom1, atom2, ...])` | Pre-intern atoms whose runtime identity matters but which don't appear in any compiled WAM body. |
| `emit_mode(Mode)` | Code-generation mode. `interpreter` (default) — every predicate runs in the step-loop interpreter. `functions` — every lowerable predicate *also* gets a native Scala fast-path function. `mixed([P/A, ...])` — only the listed predicates are lowered. See [Lowered functions](#lowered-functions-emit_mode). |
| `kernel_dispatch(true)` | Opt-in hot-path graph kernels. Predicates matching a recognised recursive graph shape are replaced by a native Scala traversal handler that bypasses the WAM loop. See [Graph kernels](#graph-kernels-kernel_dispatch). |

You can also set the mode globally without touching `Options` by asserting
`user:wam_scala_emit_mode(functions).` before generating.

### Lowered functions (`emit_mode`)

In the default `interpreter` mode the generated program is a pure
stepping WAM: every predicate is dispatched through `WamRuntime.step`.
`emit_mode(functions)` brings the Scala target to parity with the
Haskell / Rust / C++ / F# / Go / Clojure targets, all of which ship a
per-predicate **lowered emitter**
([wam_scala_lowered_emitter.pl](../src/unifyweaver/targets/wam_scala_lowered_emitter.pl)).

For every lowerable predicate the codegen emits a native Scala function
`lowered_<pred>_<arity>(s, program): Boolean` that runs the predicate's
deterministic clause 1 directly — simple register operations are inlined
as in-place mutations of the mutable `WamState`; failure-capable head
unification and structure ops delegate to small `lo*` helpers on
`WamRuntime`; deterministic builtins (`=/2`, `is/2`, the arithmetic
comparisons, type checks, `!/0`) route through `loBuiltin`. The generated
`loweredEntries` map registers an entry wrapper per predicate; `runEntry`
tries the fast path first and **falls back to a fresh interpreter run
when clause 1 misses**, so results are identical to the pure interpreter
for any boolean query (a lowered `true` is always a real solution; a
lowered `false` defers to the complete step loop, preserving
first-argument indexing, clause 2+, and backtracking into
nondeterministic sub-goals).

```prolog
write_wam_scala_project([user:ancestor/2],
    [ package('demo.anc'), emit_mode(functions) ], '/tmp/anc').
```

A predicate is lowered only if its clause 1 is deterministic (no
`try_me_else` / `retry_me_else` / `trust_me` inside the clause body) and
every clause-1 instruction is supported; predicates whose clause 1 uses a
nondeterministic builtin (`member/2`, `between/3`, `sort/2`, `findall/3`,
…) stay in the interpreter. This mirrors the deterministic-clause-1
contract of the Rust and Clojure lowered emitters.

### Graph kernels (`kernel_dispatch`)

`kernel_dispatch(true)` brings the Scala target onto the
Rust/Haskell/Elixir/Go **hot-path kernel** route. When set, the codegen
runs the shared recursive-kernel detector
([recursive_kernel_detection.pl](../src/unifyweaver/core/recursive_kernel_detection.pl))
over the predicates; any predicate matching a recognised graph shape is
replaced by a synthesized Scala `ForeignHandler` that performs the
traversal natively, bypassing the WAM step loop entirely. The handler
builds its adjacency map by enumerating the kernel's edge relation
through `WamRuntime.collectBinarySolutions/2`, so it works whether the
edges are WAM-compiled facts or a declarative fact source.

```prolog
% tc/2 is detected as transitive_closure2 and lowered to a native BFS handler;
% edge/2 stays WAM-compiled and supplies the adjacency.
write_wam_scala_project([user:tc/2, user:edge/2],
    [ package('demo.tc'), kernel_dispatch(true) ], '/tmp/tc').
```

Currently implemented: **`transitive_closure2`**,
**`transitive_distance3`** (BFS shortest-path distance),
**`transitive_parent_distance4`** (target + immediate predecessor on the
shortest path + distance), **`transitive_step_parent_distance5`**
(target + first hop from source + immediate predecessor + distance), and
**`category_ancestor`** (depth-bounded ancestor search with a visited
list; config carries `max_depth`). The remaining two kinds the detector
recognises (`weighted_shortest_path3`, `astar_shortest_path4`) follow the
same pattern and are slated as follow-ups; until then those predicates
fall back to ordinary WAM compilation (correct, just not accelerated).

> Note: `category_ancestor`'s recursive clause (`length/2` + cut + `\+` +
> recursion) exceeds what the plain step-loop interpreter currently runs
> correctly, so it should be run with `kernel_dispatch(true)` — which is
> precisely the depth-bounded search the cross-target `effective_distance`
> benchmark relies on. The native kernel's results are verified against
> SWI-Prolog ground truth.

The distance kernel returns the **shortest** path length per reachable
node (matching the Haskell/Rust/Elixir kernels). The Prolog source
enumerates one solution per path, so kernel and interpreter agree on
graphs where each reachable node has a single path length (trees / DAGs
without length-divergent alternate paths); on graphs with multiple path
lengths to the same node the kernel reports only the shortest.

### Fact-source spec forms

`scala_fact_sources([source(P/A, <Spec>)])` accepts:

- **Inline tuples**: `source(p/2, [[a, b], [b, c], ...])` — a list of
  `Arity`-element lists. Codegen synthesises a handler that returns
  every tuple as a `ForeignMulti` solution; the runtime's
  `applyBindings` filters them against the input args.
- **File-backed CSV**: `source(p/2, file('path/to/data.csv'))` — the
  generated handler opens the file at runtime, splits each line on
  commas, and uses the same `ForeignMulti` mechanism. Atoms in the
  CSV must already be in the intern table — declare them via
  `intern_atoms(...)` if no compiled WAM body mentions them.

The two forms are interchangeable for the calling Prolog code:
identical query results, only the underlying delivery differs.

### CLI of the generated program

```
scala -classpath <classes> <package>.GeneratedProgram <pred>/<arity> <arg1> [arg2 ...]
```

Argument syntax: integers (`42`, `-3`), floats (`3.14`), atoms (`a`),
structures (`f(a, b)`), and lists (`[a, b, c]`). The program prints
`true` or `false` and exits.

There is also an inner-loop benchmark mode that runs the same query
many times in one JVM invocation:

```
scala -classpath <classes> <package>.GeneratedProgram --bench <N> <pred>/<arity> <arg1> ...
```

Prints `BENCH n=<N> elapsed=<sec> last=<true|false>`. The first 5%
of iterations (capped at 50) are warmup so the JIT can settle before
timing begins. JVM startup is paid once, so the elapsed time is
dominated by the per-iteration runtime cost — the right shape for
comparing fact-source backends or measuring optimization wins.

## Supported builtins

The runtime currently implements:

| Group | Builtins |
|---|---|
| **Control** | `=/2`, `true/0`, `fail/0`, `!/0`, `\+/1`, `call/1..N`, soft-cut ITE |
| **Arithmetic** | `is/2`, `=:=/2`, `=\=/2`, `</2`, `>/2`, `=</2`, `>=/2` (Int/Float; promotes on mixed operands) |
| **Type checks** | `var/1`, `nonvar/1`, `atom/1`, `number/1`, `atomic/1`, `is_list/1`, `ground/1` |
| **Lists** | `member/2` (multi-solution), `length/2` (deterministic + generative), `append/3` (concat + split) |
| **Strings** | `atom_codes/2`, `atom_length/2` |
| **Aggregates** | `findall/3`, `bagof/3`, `setof/3` |
| **Sorting** | `sort/2`, `msort/2` |
| **Term** | `copy_term/2` |

## Supported instructions

`call`, `execute`, `proceed`, `jump`, `allocate`, `deallocate`, the
`get_*` / `put_*` / `set_*` / `unify_*` family, `try_me_else` /
`retry_me_else` / `trust_me`, `switch_on_constant`, `cut_ite`,
`builtin_call`, `call_foreign`, and `begin_aggregate` /
`end_aggregate` (the bracketing instructions for `findall/3`).

## Testing

Two test suites live in [tests/](../tests/):

- [test_wam_scala_generator.pl](../tests/test_wam_scala_generator.pl)
  — structural tests on the generated source. Always runs.
- [test_wam_scala_runtime_smoke.pl](../tests/test_wam_scala_runtime_smoke.pl)
  — end-to-end tests: generate → `scalac` → `scala` → assert on
  stdout. Gated on `scalac`/`scala` being on PATH (or set
  `SCALA_SMOKE_TESTS=1` to force).
- [test_wam_scala_classic_programs.pl](../tests/test_wam_scala_classic_programs.pl)
  — same gating; runs full Prolog programs (list reverse, naive
  reverse, Ackermann, Fibonacci) end-to-end and verifies known
  answers.
- [test_wam_scala_lowered_emitter.pl](../tests/test_wam_scala_lowered_emitter.pl)
  — structural tests for `emit_mode` resolution, predicate
  partitioning and the generated lowered functions (always run) plus
  gated runtime *parity* tests that compile the same predicates in both
  `interpreter` and `functions` mode and assert identical, correct
  results.

Run them:

```bash
swipl -g 'use_module(library(plunit)),consult("tests/test_wam_scala_generator.pl"),run_tests,halt' -t 'halt(1)'

SCALA_SMOKE_TESTS=1 swipl -g 'use_module(library(plunit)),consult("tests/test_wam_scala_runtime_smoke.pl"),run_tests,halt' -t 'halt(1)'
```

## Benchmark

A synthetic three-way bench compares the WAM-compiled, inline-tuple,
and file-backed fact-source paths on a chain of `c0 → c1 → … → cN`:

```bash
swipl -g main -t halt tests/benchmarks/wam_scala_fact_source_bench.pl -- 50 --inner 2000
```

For each backend × size combination it emits a line of the form:

```
RESULT n=<N> backend=<wam|inline|file>
       gen=<sec>           # write_wam_scala_project/3
       compile=<sec>       # scalac
       run=<sec>           # cold-start single-shot scala invocation
       inner_total=<sec>   # `--bench <I>` invocation; JVM startup paid once
       per_iter=<sec>      # inner_total / I
```

Cold-start time (`run`) is dominated by `scalac` and JVM startup;
the inner-loop columns (`inner_total`, `per_iter`) amortise that
fixed cost over `I` iterations and reveal the real per-query
difference between backends — typically: WAM-compiled is ~3-4×
faster than inline tuples, file-backed is the slowest because the
generated handler re-reads the CSV on every call.

## Limitations

- No free-variable grouping (`^`) for `bagof`/`setof`.
- `setof`'s atom ordering is by interned-string and is stable within
  a single program but may differ from SWI-Prolog if interning order
  matters.
- Inverse-mode `atom_codes` requires the resulting atom to already
  be in the intern table (immutable `WamProgram.internTable`).
- No `assert/retract`, `format/2`, `write/1`, `read/1`, or other
  side-effecting builtins.
- LMDB-backed sidecar fact sources are not yet implemented (Phase S8).
- Float arithmetic is `Double` only; rationals and bigints aren't
  supported.

## Contributing

The target lives in:

- [src/unifyweaver/targets/wam_scala_target.pl](../src/unifyweaver/targets/wam_scala_target.pl)
  — codegen
- [templates/targets/scala_wam/](../templates/targets/scala_wam/)
  — runtime + program + build.sbt mustache templates

Every new feature should ship with a smoke test in
[tests/test_wam_scala_runtime_smoke.pl](../tests/test_wam_scala_runtime_smoke.pl)
that asserts on actual `scala` stdout — generator-level structural
tests have repeatedly missed real runtime bugs in this target.
