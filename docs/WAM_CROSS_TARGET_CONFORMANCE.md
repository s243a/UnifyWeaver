# Cross-target WAM conformance harness

**Files:** `tests/wam_conformance_fixtures.pl` (the shared spec),
`tests/test_wam_cross_target_conformance.pl` (the harness).

## What it is

A single set of classic Prolog programs (`member`, `append`, `reverse`,
`fib`, `ack`, and a `builtins` arithmetic/comparison/unification pack)
with one shared table of expected query results. The harness compiles
that *same* spec to every WAM backend whose toolchain is on `PATH`, runs
each backend, and asserts the answers match.

Per-target classic-program tests already exist, but each re-declares its
own fixtures, so a backend can silently diverge without any test
noticing. This harness is the safety net for exactly that — the kind of
divergence the Haskell `Proceed` and WAT `allocate` first-argument
indexing bugs produced (`member/2` wrongly succeeding).

## How a query is run

Backends disagree on how to feed list/compound args into a predicate's
entry point, so invocation is per-target:

- **scala** passes the query args straight to its `GeneratedProgram`
  driver (`<predkey> <args...>` → `true`/`false`).
- **elixir / wat** synthesise a ground 0-arity wrapper per query
  (`ctw_N :- pred(args).`), compile it with the program, and ask whether
  `ctw_N` succeeds. This is the shape their own runtime tests use.

Each adapter is self-contained (`ct_build/4`, `ct_run/5`,
`ct_teardown/2`); adding a backend is one adapter plus a
`conformance_target/1` entry. Missing toolchains **skip**, they don't
fail.

## Coverage vs CI speed

The dominant cost is process startup (scalac/JVM/BEAM/node), not compute,
so the harness is built to stay cheap:

- builds are **per program** (small projects), and the recursive samples
  (`fib`, `ack`) use small inputs;
- `member` — a set operation — is the preferred everyday case: high-level
  but far cheaper than a generic recursive algorithm;
- the query set can be **random-sampled** so any single CI run is bounded
  while coverage accumulates across runs.

Environment knobs:

| Variable | Effect |
|---|---|
| `CONFORMANCE_TARGETS=scala,elixir` | limit which backends run |
| `CONFORMANCE_PROGRAMS=member,fib`  | limit which programs run |
| `CONFORMANCE_SAMPLE=N`             | random N queries per program |
| `CONFORMANCE_SEED=N`               | seed the sampler (reproducible) |

Suggested CI tiers:

- **fast / every push:** `CONFORMANCE_PROGRAMS=member,builtins
  CONFORMANCE_SAMPLE=2` — broad builtin coverage, set operation, minimal
  compute. Seed varies per run so sampling rotates coverage.
- **full / nightly or pre-merge:** no sampling — every program, every
  query, every available backend.

## Known divergences (tracked as `ct_xfail/2`)

The harness is green today; the divergences below are tolerated and
logged (an unexpected pass is logged as `XPASS` so the entry can be
retired). Each is a real backend gap the harness surfaced, not a fixture
artifact — **Scala passes the whole spec**, which is the reference.

| Backend | Program(s) | Kind | Cause |
|---|---|---|---|
| wat | member | xfail | Read-mode structure/list argument unification is unimplemented (the read-mode branches of `unify_variable`/`unify_value`/`unify_constant` are nops; no S-register), so `get_structure`/`get_list` match only the functor. See `WAM_SWITCH_INDEXING_CROSS_TARGET.md`. |
| wat | append, reverse | **skip** | A *second*, separate WAT bug: the generator loops re-emitting millions of "unrecognized instruction" warnings on recursive list-**building** predicates, so the project is impractical to write. Skipped (not built) rather than xfail'd. |
| elixir | append, reverse | xfail | The lowered Elixir backend fails to unify a freshly-constructed list against an already-**ground** compound head argument: `capp([a],[b],[a,b])` returns false, while `capp([a],[b],X), X=[a,b]` succeeds. `member` passes (it only matches an input list). |

`ct_xfail/2` = build and run, tolerate a wrong answer (and log `XPASS` if
it unexpectedly matches). `ct_skip/2` = do not even build, because
*generation itself* is unusable.

### Other backend issue surfaced (not xfail)

- **Scala loops compiling 0-arity predicates with comparison-only
  bodies** (e.g. `p :- 3 > 2.`). The harness sidesteps this by (a) giving
  Scala direct args rather than 0-arity wrappers, and (b) phrasing the
  `builtins` comparison fixture as a 1-arity predicate. Worth fixing in
  the Scala backend separately.

## Adding more backends

Targets with real toolchains that are not yet wired up (haskell, rust,
go, cpp, c, fsharp, lua, r, clojure, llvm) each already have a
`write_wam_<target>_project/3` and a per-target runtime test
demonstrating how to compile+run. Wiring one in = a `ct_build/ct_run/
ct_teardown` adapter mirroring the Scala (direct-arg) or Elixir/WAT
(0-arity wrapper) shape, plus a `conformance_target/1` line and a
toolchain probe.
