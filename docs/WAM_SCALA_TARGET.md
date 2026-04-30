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

Run them:

```bash
swipl -g 'use_module(library(plunit)),consult("tests/test_wam_scala_generator.pl"),run_tests,halt' -t 'halt(1)'

SCALA_SMOKE_TESTS=1 swipl -g 'use_module(library(plunit)),consult("tests/test_wam_scala_runtime_smoke.pl"),run_tests,halt' -t 'halt(1)'
```

## Benchmark

A synthetic three-way bench compares the WAM-compiled, inline-tuple,
and file-backed fact-source paths on a chain of `c0 → c1 → … → cN`:

```bash
swipl -g main -t halt tests/benchmarks/wam_scala_fact_source_bench.pl -- 50
```

The bench prints `RESULT n=<N> backend=<wam|inline|file> gen=<sec> compile=<sec> run=<sec>`
lines for each backend×size combination. Cold-start time is
dominated by `scalac` and JVM startup; the three backends are within
noise of each other for cold queries because the fact-source choice
mostly affects inner-loop performance, which a single-shot CLI query
doesn't expose.

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
