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
| `UW_SMOKE_TMPDIR`, `TMPDIR`, `$PREFIX/tmp` | writable temp-root selection via `tests/helpers/smoke_paths.pl` |
| `SCALA_MAVEN_ROOT` | optional root for Scala runtime jars when running generated Scala classes via `java -cp` |

Scala builds still use `scalac`, but the harness runs generated Scala
classes via `java -cp` when Scala 3 runtime jars are discoverable. This
avoids Scala-CLI/JNA launcher failures on native Termux while preserving
the legacy `scala -classpath ...` fallback for environments without a
local Scala maven cache.

Suggested CI tiers:

- **fast / every push:** `CONFORMANCE_PROGRAMS=member,builtins
  CONFORMANCE_SAMPLE=2` — broad builtin coverage, set operation, minimal
  compute. Seed varies per run so sampling rotates coverage.
- **full / nightly or pre-merge:** no sampling — every program, every
  query, every available backend.

Because random sampling is nondeterministic, a real divergence can pass
one push and fail the next, which is confusing for whoever's bisecting.
CI should **log the `CONFORMANCE_SEED` it used** (and set one explicitly)
so any failure is reproducible from the recorded seed.

## Known divergences (tracked as `ct_xfail/2`)

The harness is green today; the divergences below are tolerated and
logged (an unexpected pass is logged as `XPASS` so the entry can be
retired). Each is a real backend gap the harness surfaced, not a fixture
artifact. The oracle is the hand-specified expected-results table in
`wam_conformance_fixtures.pl` (standard Prolog semantics), not any
backend's output; among the backends, **Scala is the reference
implementation** — it passes the whole spec. The table below is the full
set of tracked divergences (it matches the `ct_xfail/2` / `ct_skip/2`
facts in the harness); WAT conforms only on `ack`, and `elixir` and
`scala` pass the whole spec.

| Backend | Program(s) | Kind | Cause |
|---|---|---|---|
| wat | member | xfail | Read-mode structure/list argument unification is unimplemented (the read-mode branches of `unify_variable`/`unify_value`/`unify_constant` are nops; no S-register), so `get_structure`/`get_list` match only the functor. See `WAM_SWITCH_INDEXING_CROSS_TARGET.md`. |
| wat | fib | xfail | `is/2` with an already-bound LHS doesn't verify the computed value — `cfib(10,54)` returns true though `fib(10)=55` (the result is stored over the bound arg instead of being unified/checked). |
| wat | builtins | xfail | `cbi_arith` uses `//` (integer div) and `mod`, which the WAT backend does not evaluate correctly (returns false). The comparison (`cbi_cmp`) and unification (`cbi_eq`) families are fine. |
| wat | append, reverse | **skip** | A *separate* WAT codegen bug: the generator loops re-emitting millions of "unrecognized instruction" warnings on recursive list-**building** predicates, so the project is impractical to write. Skipped (not built) rather than xfail'd. |
| ~~elixir~~ | ~~append, reverse~~ | **fixed** | Was: a freshly-constructed list (`./2`, from `put_list`) would not unify against an already-**ground** list compound (`[|]/2`, from `put_structure`) in a clause head, so `capp([a],[b],[a,b])` returned false while `capp([a],[b],X), X=[a,b]` succeeded. Root cause: `unify/3`'s compound clause demanded *identical* functor names and never applied the `./2`↔`[|]/2` cons-cell aliasing that the `get_structure` match path (`step_get_structure_matches?/2`) already used. Now conformant; xfails removed. |

`ct_xfail/2` = build and run, tolerate a wrong answer (and log `XPASS` if
it unexpectedly matches). `ct_skip/2` = do not even build, because
*generation itself* is unusable. (`append`/`reverse` on WAT are `ct_skip`
only — they are never built; the formerly-shadowing `ct_xfail` facts have
been removed.)

### Other backend issue surfaced (not xfail)

- **Scala loops compiling 0-arity predicates with comparison-only
  bodies** (e.g. `p :- 3 > 2.`). The harness sidesteps this by (a) giving
  Scala direct args rather than 0-arity wrappers, and (b) phrasing the
  `builtins` comparison fixture as a 1-arity predicate. Worth fixing in
  the Scala backend separately.

## Adding more backends

Each remaining backend already has a `write_wam_<target>_project/3` and a
per-target runtime test showing how to compile+run. Wiring one in is a
`ct_build/4` + `ct_run/5` + `ct_teardown/2` adapter, a
`conformance_target/1` (+ `ct_default_target/1`) entry, and a
`ct_toolchain/2` probe.

### The invocation style is the hard part

Scala and Elixir were cheap to add because each ships a **query driver**
that takes `<predkey> <args>` (scala) or runs a named predicate
(elixir's `run_classic.exs`). The remaining native backends do **not**:
`write_wam_<target>_project/3` emits a *library* exporting one boolean
function per predicate (e.g. Rust/Go/C/C++ `fn pred(args) -> bool`), or a
runtime you drive by label (Haskell's
`run :: WamContext -> WamState -> Maybe WamState`). So each needs a
**hand-written driver** that turns "does `pred(args)` hold?" into
true/false — and that driver is itself a piece of code with its own
correctness pitfalls. Budget per-backend *investigation*, not a quick
loop.

Project-gen + run contract per backend (toolchains present in the audit
env: ghc/cabal, cargo, go, g++/clang, gcc):

| Backend | Project-gen | Toolchain | What it produces | Driver you must supply |
|---|---|---|---|---|
| haskell | `write_wam_haskell_project/3` (opts `module_name`, `use_hashmap(false)`) | cabal | `Main.hs`/`WamRuntime.hs`/`Predicates.hs` (exports `allCode`, `allLabels`, `compileTimeAtomTable`, `mkContext`); `run` is by-label | Replace `Main.hs` with a driver: `mkContext allCode allLabels` (+ `wcInternTable=compileTimeAtomTable`, `wcLoweredPredicates=Lowered.loweredPredicates`), `emptyState{wsPC=lookup key allLabels}`, `run ctx s0` → Just/Nothing. `cabal run -v0 <exe> -- <key>`. |
| rust | `write_wam_rust_project/3` | cargo | `lib.rs` exporting `pub fn <pred>(..) -> bool` | a `main.rs`/bin that calls the wrapper fn and prints |
| go | `write_wam_go_project/3` (`package_name`, `parallel`) | go | lib exporting `func Pred(..) bool` (or `main.go` if `package_name=main`) | a `main` that calls the wrapper fn |
| cpp | `write_wam_cpp_project/3` | g++/clang++ | `generated.cpp/.hpp` with per-pred methods | a `main.cpp` calling the method |
| c | `write_wam_c_project/3` | gcc/clang | `wam_runtime.c`/`lib.c` + `wam_run_predicate(state,"pred/arity",..)` | a `main.c` calling `wam_run_predicate` |

For all of them the **0-arity wrapper** shape (as used by Elixir/WAT)
generalises: synthesise `ctw_N :- pred(args).`, compile it with the
program, and have the driver ask whether `ctw_N` succeeds.

### Prototype finding: Haskell (not landed)

A throwaway Haskell adapter was prototyped. The driver above **compiles
and runs**, and discriminates correctly on arithmetic (`fib(10,55)`→true,
`fib(10,54)`→false, `ack` true/false both right). But via realistic
wrapper invocation it reports many wrong answers — including
**`cbi_eq(foo)`→false**, i.e. `foo = foo` is false. That is trivially
impossible for a correct backend, so the minimal driver almost certainly
has an **atom-interning mismatch** (atoms built in the wrapper don't
intern to the same ids as atoms in the clause bodies, so unification
spuriously fails). The Haskell adapter was therefore **not committed** —
a backend adapter on top of an untrustworthy driver would produce
meaningless xfails.

Open questions a future Haskell adapter must resolve first:
- Wire the driver's intern table exactly as the generated `Main.hs` does
  (it extends `compileTimeAtomTable` with runtime atoms via
  `internAtom`); `mkContext`'s default table is likely not what the
  compiled code expects.
- Re-confirm `member/2` on the **heap-cons** list path. The PR #2708
  Haskell indexing fix was validated by injecting the list as a `VList`
  straight into a register; realistic programs build the list as a heap
  cons cell. Whether `member(z,[a,b,c])` is correct on that path was not
  established here (the interning bug masks it). This is worth checking
  independently of the harness.

### Cross-cutting observation (investigated)

The divergences the harness has found cluster around **matching /
unifying heap-built compound terms** — WAT (read-mode structure unify),
Elixir (constructed list vs ground compound head arg), the suspected
Haskell path. The open question was whether this is *one* shared lowering
weakness or *N* independent backend bugs.

**It is not a shared lowering bug.** The WAM lowering
(`wam_target.pl:compile_head_arguments/5` + `compile_unify_arguments/5`)
emits `get_structure`/`get_list`/`unify_*` identically for every backend,
and **Scala consumes that same stream and passes the whole spec** — so
the instruction stream is sound. The failures live in each backend's
*runtime* implementation of the read-mode unify protocol, and they are
distinct:

- **Elixir** — a self-contained `unify/3` bug (cons-functor `./2`↔`[|]/2`
  aliasing missing on the structural-unify path). Fixed in this pass; one
  clause, ~6 lines. `append`/`reverse` now conformant.
- **WAT** — a genuine runtime *gap*: the read-mode `unify_*` branches are
  nops and there is no S-register / argument queue. This is a real
  runtime feature to build, not a one-line fix.
- **Haskell** — still unverified: the prototype driver's atom-interning
  bug (above) masks it. Worth re-auditing `readNextArg`/`unifyValues`
  against the now-fixed Elixir/Scala runtimes for the same cons-aliasing
  and heap-cons `member/2` concerns once the driver is sound.

So the leverage is per-backend after all, but the harness did its job:
it pinned each divergence to a specific runtime and made the Elixir one a
quick, verified fix.
