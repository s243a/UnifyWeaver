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
- **elixir / wat / haskell / python / go** synthesise a ground 0-arity
  wrapper per query (`ctw_N :- pred(args).`), build it with the program,
  and ask whether `ctw_N` succeeds. This is the shape their own runtime
  tests use. (Python is interpreted — no build step; its generated
  `main.py` prints `A_i = ...` register dumps on success and `false.` on
  failure, so the adapter reads success as the absence of `false.`. Go is
  compiled via `prefer_wam(true)` — its default strategy is the
  dataflow/stream backend, not the shared WAM pipeline — and `ct_build`
  gates compilation with `go build`, while `ct_run` executes `go run`,
  which launches the program from go's build cache rather than exec'ing a
  binary under `$TMPDIR`.)

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

- **fast / PR + main smoke:** one backend per matrix job with
  `CONFORMANCE_TARGETS=<target> CONFORMANCE_PROGRAMS=member,builtins
  CONFORMANCE_SAMPLE=2 CONFORMANCE_SEED=<run number>` — broad builtin
  coverage, set operation, minimal compute, and failures attributed to a
  single backend. Missing toolchains still skip through the harness's
  normal `ct_available/1` guards.
- **full / nightly or pre-merge:** no sampling — every program, every
  query, every available backend.

Because random sampling is nondeterministic, a real divergence can pass
one push and fail the next, which is confusing for whoever's bisecting.
CI sets `CONFORMANCE_SEED` explicitly, and the harness logs
`CONFORMANCE_SEED=<n>` once per run so any failure is reproducible from
the recorded seed.

## Known divergences (tracked as `ct_xfail/2`)

The harness is green today and **every registered backend — scala,
elixir, wat, haskell, python, go, rust, c, and c++ — passes the whole
spec** (there are no live `ct_xfail/2` or `ct_skip/2` entries; both are
declared `dynamic` so they may have zero clauses). The oracle is the hand-specified
expected-results table in `wam_conformance_fixtures.pl` (standard Prolog
semantics), not any backend's output; Scala was the original reference.
A tolerated divergence would be logged via a new `ct_xfail/2`, with an
unexpected full match logged as `XPASS` so the entry can be retired once
the gap is fixed. The table below records every divergence the harness has
surfaced — each a real backend gap, not a fixture artifact — all now
fixed.

| Backend | Program(s) | Kind | Cause |
|---|---|---|---|
| ~~wat~~ | ~~member~~ | **fixed** | Read-mode argument unification was unimplemented (the read-mode branches of `unify_variable`/`unify_value`/`unify_constant` were nops; no S register), so `get_structure`/`get_list` matched only the functor and element mismatches went undetected. Added an **S register** (heap arg pointer at addr 65568); `get_structure`/`get_list` set it and the `unify_*` read-mode branches consume successive arg cells (binding/checking and failing on mismatch). `get_list` also matches a tag-3 `[|]/2` compound as a list cell (the compiler emits the outer list via `put_list`/tag-4 but nested cons via `put_structure [|]/2`/tag-3 — the same `./2`-vs-`[|]/2` split as the other backends) using a generated `$cons_op1` global. |
| ~~wat~~ | ~~builtins~~ | **fixed** | Two bugs: `eval_arith` lacked `//` and `mod`; and `functor_arity_of('///2')` split on `/` expecting exactly two parts and fell back to **arity 0**, so `//` structure cells were skipped by `eval_arith`'s arity-2 dispatch (same `/`-in-operator class as the Haskell `//` bug). Fixed `functor_arity_of` to take the last `/`-component, and added `//`/`mod` (with zero-divisor guards and floored `mod`). `cbi_cmp`/`cbi_eq` already worked. |
| ~~wat~~ | ~~fib~~ | **fixed** | `cfib(10,54)` wrongly succeeded. `peephole_fused_arith` rewrites `F is F1+F2` into a single `fused_is_add(Dest,Src1,Src2)`, and the WAT handlers for the fused `is/*` forms (`add`/`sub`/`mul` + `_const` variants) called `$bind_reg_deref` to **store** the result into `Dest` unconditionally — never checking an already-bound `Dest`. So a recursive predicate's result check (`F` bound to the queried value) always succeeded by clobbering `F`. Added `$is_unify_int` (bind if unbound, else integer-equality check, mirroring `builtin_is`) and routed all five fused forms through it. The non-fused `is/2` path was already correct (hence `cbi_arith` passed) — fib was the only program exercising the fused result-check. |
| ~~wat~~ | ~~append, reverse~~ | **fixed** | Two bugs. **(1) Generation:** the `switch_on_term` first-arg index that list-recursive predicates emit had no working parser clause (`parse_term_entries` expected an old operand format), so it fell to the `unrecognized instruction → allocate` fallback — looping on warnings / failing to generate. Fixed by emitting an empty (unindexed) `switch_on_term_hdr` on register 0, mirroring `switch_on_term_a2`; the `try_me_else` chain alone is correct. **(2) Unification:** with generation working, `$unify_regs` did only **shallow** (tag+payload) equality, so a constructed cons (tag-3 `[|]/2`) would not match an already-ground list cell (tag-4) and append/reverse returned false on correct answers — the same `./2`-vs-`[|]/2` split as the other backends, plus it never recursed into elements. Replaced with recursive, cons-aware `$unify_addrs`. Both conformant; skips removed. |
| ~~elixir~~ | ~~append, reverse~~ | **fixed** | Was: a freshly-constructed list (`./2`, from `put_list`) would not unify against an already-**ground** list compound (`[|]/2`, from `put_structure`) in a clause head, so `capp([a],[b],[a,b])` returned false while `capp([a],[b],X), X=[a,b]` succeeded. Root cause: `unify/3`'s compound clause demanded *identical* functor names and never applied the `./2`↔`[|]/2` cons-cell aliasing that the `get_structure` match path (`step_get_structure_matches?/2`) already used. The runtime also aliases native Elixir `[]` with WAM `"[]"` for direct list inputs. Now conformant; xfails removed. |
| ~~haskell~~ | ~~member, append, reverse~~ | **fixed** | All three failed on heap-built cons lists. Four root causes, now fixed in `wam_haskell_target.pl`: (1) `unifyVal`/`unifyValues` did **no** structural unification — added a shared `unifyTerms`; (2) the cons functor `"[|]/2"` interned to its own id, distinct from `atomDot` — `intern_struct_functor/2` folds every cons spelling onto `atomDot` (the `./2`-vs-`[|]/2` class, same as the Elixir bug); (3) `GetValue` had its own inline unify bypassing the above — now routed through `unifyVal`; (4) **the multi-element bug** — building `[a\|X]` with `X` a `set_variable` tail placeholder emitted `VList [a, X]` (`X` as a 2nd *element*) and `put_structure` filling `X` did not bind the embedded var, so `addToBuilder` now emits a `Str atomDot [hd, tl]` cons cell for a partial tail and binds the placeholder var on finalize. Fixes lists of any length; xfails removed. |
| ~~haskell~~ | ~~builtins~~ | **fixed** | `=/2` of two identical atoms returned false (no `BuiltinCall "=/2"` handler — added one routing through `unifyVal`), and `//`/`mod` evaluated incorrectly: the arity-stripper `takeWhile (/= '/')` truncated any operator containing `/` (so `//` → `""`). Replaced with `bareArithOp` (strips only a trailing `/<digits>`). `fib`/`ack` already passed. Now conformant; xfail removed. |
| ~~python~~ | ~~builtins~~ | **fixed** | `cbi_arith(28)` → false: `wam_line_to_python_literal` computed the `put_structure`/`get_structure` arity with `split_string(Fn,"/","",[_,ArStr])`, which expects exactly two parts and fell back to **arity 0** for `///2` (integer division `//`). A 0-arity `//` compound carried no argument cells, so `eval_arith` could not apply it and `X is 17 // 5` failed — the same `/`-in-operator class as the Haskell/WAT `//` bugs. Added `python_functor_arity/2` (last `/`-component). The Python adapter was added to the harness in the same change; every other program already passed. |
| ~~go~~ | ~~fib, ack~~ | **fixed** | `is/2` wrapped every result in `&Float`, but Go's `Unify` is type-strict (Integer never unifies with Float), so `R is N + 1` failed whenever R was already bound to a ground Integer — `cack(0,5,6)`, `cfib(10,55)`. Integral results now wrap as `&Integer` (mirrors the Python int-vs-float heuristic), in `templates/targets/go_wam/state.go.mustache`. Made fib and ack pass; the Go adapter was added in the same change. |
| ~~go~~ | ~~builtins~~ | **fixed** | Nested-arithmetic gap. A depth-1 `is` was correct (`//`/`mod` too) but `cbi_arith`'s `A+B+C+D+E` — `+(+(+(+(A,B),C),D),E)` — mis-evaluated. The compiler builds nested terms outer-first: `set_variable` drops an unbound placeholder into the outer arg, then `put_structure` builds the inner term into that *same register* — but `PutStructure` overwrote the register without binding the embedded placeholder, so the outer arg stayed unbound and `evalArithmetic` gave up at depth ≥ 2. `PutStructure` now binds an unbound placeholder it overwrites (`wam_go_target.pl`), which also fixes list **tail** cells built the same way. |
| ~~go~~ | ~~member, reverse~~ | **fixed** | Cons-cell traversal. Lists build as an outer `put_list` `*List` cell whose tails are `put_structure "[|]/2"` `*Structure`s (the `./2`-vs-`[|]/2` class). Two further `GetList` bugs, both fixed: (a) it recognised only `*List`, not the inner `[|]/2` `*Structure` tail cells — added `consHeadTail` (`state.go.mustache`); (b) it tested `isUnbound` **before** `deref`, so a *bound* variable tail (after the placeholder fix above) was mistaken for unbound and sent into write mode, wrongly succeeding `cmem(z,…)` and looping `clist_reverse` — now derefs first. With all three fixes member, reverse (and append's negative cases) pass. |

(Haskell is opt-in — `CONFORMANCE_TARGETS=haskell` — because each program is a cabal compile. `fib` and `ack` pass; the driver, `tests/fixtures/haskell_conformance_driver.hs`, runs a 0-arity wrapper whose atoms are baked into the compiled stream, which is what removed the earlier interning mismatch — see below.)

(Go is opt-in too — `CONFORMANCE_TARGETS=go` — and now passes the whole spec. The driver is a small generated `main.go` that looks up the wrapper label in `SharedWamLabels` and runs `vm.Run()`.)

(Rust and C are opt-in — `CONFORMANCE_TARGETS=rust` / `=c` — and pass the whole spec; both needed the same convention fixes the other native backends did (cons-cell aliasing, placeholder binding, integer `is/2`, and — for the indexing instructions a backend doesn't translate — degrading to a real no-op rather than dropping the instruction and shifting label PCs). See `WAM_BACKEND_CONVENTIONS.md`.)

(C++ is opt-in — `CONFORMANCE_TARGETS=cpp` — and was **conformant on first onboarding**: the only WAM-runtime gaps the conventions warn about were already handled in `wam_cpp_target.pl`, so no backend fix was needed. `write_wam_cpp_project/3` with `emit_main(true)` generates the CLI driver itself (`cpp/main.cpp` runs `vm.query(key, args)` and exits 0/1), so the adapter just builds the three `.cpp` files with `g++` and reads the exit status.)

`ct_xfail/2` = build and run, tolerate a wrong answer (and log `XPASS` if
it unexpectedly matches). `ct_skip/2` = do not even build, because
*generation itself* is unusable. Both registries are **empty today** — no
backend is xfailed or skipped. (WAT `append`/`reverse` were the last
`ct_skip` entries; once WAT's `switch_on_term` generation and cons-aware
unification were fixed they build and pass like everything else, and the
skips were removed.)

### Other backend issue surfaced

- **Scala 0-arity predicates with comparison-only bodies** (e.g.
  `p :- 3 > 2.`) — **fixed.** Two independent gaps: (1)
  `emit_scala_wrapper/4` called `numlist(1, 0, _)`, which *fails* when
  `Low > High`, silently failing the whole project write for any 0-arity
  predicate; and (2) the CLI driver's single-query / `--queries` dispatch
  required at least one argument (`rest.nonEmpty`), so even a compiled
  0-arity predicate could not be invoked. Now the wrapper guards the
  empty-arg case (and emits a typed `Array[WamTerm]()`), and dispatch
  keys on `key.contains("/")` so a lone `pred/0` runs with no args.
  Covered by `tests/test_wam_scala_classic_programs.pl`
  (`zero_arity_comparison`). The harness still gives Scala direct args
  rather than 0-arity wrappers (for list-arg passing), and the `builtins`
  comparison fixture stays 1-arity — that workaround is no longer
  *required*, just retained.

## Adding more backends

Each remaining backend already has a `write_wam_<target>_project/3` and a
per-target runtime test showing how to compile+run. Wiring one in is a
`ct_build/4` + `ct_run/5` + `ct_teardown/2` adapter, a
`conformance_target/1` (+ `ct_default_target/1`) entry, and a
`ct_toolchain/2` probe.

> **Before debugging answers, read
> [`WAM_BACKEND_CONVENTIONS.md`](WAM_BACKEND_CONVENTIONS.md).** Every
> divergence in the table above belongs to one of five recurring classes
> (cons-cell spelling, `/`-in-operator functor arity, outer-first
> placeholder binding, deref-before-type-test, `is/2` result typing) that
> have bitten *every* WAM backend in turn. That doc is the checklist; this
> section is about the orthogonal problem of *invoking* the generated
> program.

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

### Haskell adapter — landed, with the interning bug solved

The earlier throwaway Haskell driver reported impossible answers —
notably **`cbi_eq(foo)`→false** (`foo = foo` false) — an **atom-interning
mismatch**: a driver that parsed query atoms at *runtime* interned them to
different ids than the same atoms compiled into the clause bodies, so
unification spuriously failed. The committed driver
(`tests/fixtures/haskell_conformance_driver.hs`) sidesteps this entirely:
it runs a **0-arity wrapper** (`ctw_N :- pred(args).`) whose atoms are all
baked into the compiled instruction stream, so there is **no runtime atom
parsing** — and it wires `wcInternTable = compileTimeAtomTable`. Proof it
discriminates correctly: `fib` and `ack` (true and false cases) pass.

That unmasked the **real** backend bugs. `member`, `append`, and
`reverse` are **now fixed** (see the table) — the chain was: no structural
unification, the `"[|]/2"`-vs-`atomDot` cons split, a `GetValue` inline
unify that bypassed both, and the multi-element construction bug (a
`set_variable` tail placeholder emitted as a `VList` *element* and never
bound when `put_structure` filled it). The PR #2708 first-arg indexing fix
had only been validated with the list injected as a `VList` in a register;
the harness exercised the realistic **heap-cons** path and surfaced all of
this. Validated with no regressions across the Haskell suites (target
codegen, lowered phases 1–4, dispatch ghc smoke, st_regs e2e, iso, csr).

Still open (own PR): the `builtins` `=/2` / `//` / `mod` gaps — isolated
builtin bugs, unrelated to the list path.

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
  clause, ~6 lines. The runtime also aliases native `[]` with WAM `"[]"`
  so direct Elixir-list inputs and heap-built WAM lists agree.
  `append`/`reverse` now conformant.
- **WAT** — a genuine runtime *gap*: the read-mode `unify_*` branches are
  nops and there is no S-register / argument queue. This is a real
  runtime feature to build, not a one-line fix.
- **Haskell** — **fixed** (`member`/`append`/`reverse`). It was the same
  `./2`-vs-`[|]/2` cons-aliasing class as Elixir, but with three more
  layers: no structural unification at all, a `GetValue` inline-unify
  bypass, and a multi-element construction bug (`set_variable` tail
  placeholders emitted as `VList` elements, never bound on `put_structure`
  finalize). So Haskell needed real compound unification *plus* a
  cons-cell representation fix for partial tails — more than the Elixir
  one-liner, but the same root family. `=/2`/`//`/`mod` remain (separate
  builtin bugs).

So the leverage is per-backend after all, but the harness did its job:
it pinned each divergence to a specific runtime — made the Elixir one a
quick, verified fix, and turned the Haskell "interning bug" from a
guess into a precise, now-unmasked list of runtime gaps.
