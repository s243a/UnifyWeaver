# WAM R Target -- Usage Guide

The WAM R target compiles Prolog predicates to a self-contained R
package that runs under any standard `Rscript` (>= 3.6). It implements
a hybrid WAM: predicates and queries are interpreted by an R-level
stepping engine, but the instruction array, label map, intern table,
and dispatch tables are all compiled at codegen time and shared across
queries.

The target lives in:

- [src/unifyweaver/targets/wam_r_target.pl](../src/unifyweaver/targets/wam_r_target.pl)
  -- codegen + project writer
- [src/unifyweaver/targets/wam_r_lowered_emitter.pl](../src/unifyweaver/targets/wam_r_lowered_emitter.pl)
  -- Phase-3 native R emitter (function-mode lowering)
- [templates/targets/r_wam/](../templates/targets/r_wam/)
  -- runtime + program + DESCRIPTION mustache templates

## Quick start

```prolog
:- use_module('src/unifyweaver/targets/wam_r_target').

:- dynamic user:greet/1.
user:greet(world).
user:greet(r).

:- initialization((
    write_wam_r_project(
        [user:greet/1],
        [ module_name('greet.demo'),
          intern_atoms([world, r])
        ],
        '/tmp/greet_demo'),
    halt
)).
```

After running this script:

```bash
cd /tmp/greet_demo
Rscript R/generated_program.R 'greet/1' world
# -> true
Rscript R/generated_program.R 'greet/1' mars
# -> false
```

The runtime is self-sourced from the generated program: `Rscript
R/generated_program.R` finds `R/wam_runtime.R` next to it and loads
it automatically. No `R CMD INSTALL` step is required.

## API

### `write_wam_r_project(+Predicates, +Options, +ProjectDir)`

Generates a complete R project for a list of Prolog predicates.

**Predicates** -- list of `Module:Pred/Arity` or `Pred/Arity`
indicators. Predicates that aren't declared as dynamic must be
defined in the calling file.

**ProjectDir** -- output directory (created if needed). Layout:

```
<ProjectDir>/
├── DESCRIPTION
└── R/
    ├── wam_runtime.R         <- stepping engine, helpers, builtins
    └── generated_program.R   <- intern table, instructions, dispatch, main
```

**Options**:

| Option | Meaning |
|---|---|
| `module_name(Name)` | DESCRIPTION `Package:` field (default `wam.r.generated`). |
| `emit_mode(Mode)` | Lowering mode: `interpreter` (default), `functions`, or `mixed([P/A, ...])`. |
| `foreign_predicates([P/A, ...])` | These predicates' WAM bodies are replaced by a `CallForeign` stub. |
| `r_foreign_handlers([handler(P/A, "<R-expr>"), ...])` | Inline R source for each foreign handler. |
| `intern_atoms([atom1, atom2, ...])` | Pre-intern atoms whose runtime identity matters but which don't appear in any compiled WAM body. |

`emit_mode/1` resolution order: explicit option, then the multifile
fact `user:wam_r_emit_mode/1`, then `interpreter`. See "Emit modes"
below.

### CLI of the generated program

```
Rscript R/generated_program.R <pred>/<arity> <arg1> [arg2 ...]
```

Each CLI arg is parsed via the runtime's operator-precedence
parser, so atoms (`alice`), integers (`-3`), floats (`3.14`),
lists (`[1, 2, 3]`), compound terms (`f(a, b)`), nested compounds
(`g(h(1), [2, 3])`), and operator expressions (`1+2*3`) all reach
the predicate as proper WAM terms. Bare uppercase names parse as
fresh unbound variables; the same name in multiple args refers to
the same logical variable (the parser shares state across args).
The program prints `true` or `false` and exits with status 0 / 1.

For richer drivers, source the generated program from another R
script and call `WamRuntime$run_predicate(shared_program, start_pc,
args)` directly. Each predicate also gets a `pred_<name>(...)`
wrapper at top level (e.g. a Prolog `ancestor/2` exposes
`pred_ancestor(x, y)`); the `pred_` prefix avoids clashes with
base R functions when the user's predicate name is `c`, `t`, `q`,
`cat`, etc.

## Architecture

### Tagged-list values

Every Prolog term is an R `list(tag = <string>, ...)`:

| Tag | Shape | Meaning |
|---|---|---|
| `"atom"` | `list(tag, id)` | interned atom; `id` is integer index into intern table |
| `"int"` | `list(tag, val)` | integer (R's `integer` type) |
| `"float"` | `list(tag, val)` | double-precision float |
| `"unbound"` | `list(tag, name)` | logic variable; `name` is the trail key |
| `"struct"` | `list(tag, fid, args)` | compound `f(a1,...,an)`; `args` is an R list of values |

Lists are encoded as cons cells: `[a, b, c]` is `'.'(a, '.'(b, '.'(c,
[])))`. Atoms `'[]'` (empty list) and `'.'` (cons) are pre-interned
at ids 2 and 3.

### Mutable WamState

State is held in an R environment (pass-by-reference), so deeply
nested helpers can mutate registers, the trail, the choice-point
stack, etc. without explicit threading:

- `state$regs2` -- environment mapping `"r1"`, `"x101"`, `"y201"` to
  values. Register encoding: A1->1, X1->101, Y1->201.
- `state$bindings` -- environment mapping unbound-var names to
  their bound values.
- `state$trail` -- character vector of var-names to undo on
  backtrack.
- `state$cps` -- list of choice-point records; backtrack pops the
  last one.
- `state$pc`, `state$cp`, `state$halt` -- program counter,
  continuation, halt flag.
- `state$mode`, `state$build_stack` -- compound-term construction
  state (read mode while unifying nested args, write mode while
  building).

### Choice-point machinery

A choice point captures snapshots needed to roll back on failure:
trail length, register environment (deep-copied), `cp`,
`var_counter`, mode, build stack. The standard kind triggers the
`next_pc` jump; specialised kinds carry extra fields:

- `"iter"` -- pushed by builtin enumerators (`member/2`,
  `between/3`). Carries a `retry(state)` closure that performs the
  next iteration's unification and may push a successor iter-CP.
- `"dynamic"` -- pushed when dispatching a runtime-asserted
  predicate with multiple clauses; carries the clause list, current
  index, and `resume_pc` so backtracking re-tries the post-call
  body with the next clause's bindings.
- `"aggregate"` -- pushed by `BeginAggregate`; on backtrack it
  resumes at `next_pc` (after `EndAggregate`) and binds the bag
  register to the collected list.

### Dispatch order

When the WAM emits `Call <pred>/<n>` (or `Execute`), the runtime
tries handlers in order and stops at the first hit:

1. **Compiled label** -- `program$labels[["pred/n"]]` jumps to the
   compiled instruction array.
2. **Foreign handler** -- `program$foreign_handlers[["pred/n"]]`,
   set via `r_foreign_handlers/1` option. Receives `(state, args,
   intern_table)` and returns `list(ok = TRUE/FALSE, ...)`.
3. **Dynamic store** -- `program$dynamic[["pred/n"]]`, populated by
   `assertz/1` / `asserta/1` at runtime.
4. **Library** -- `WamRuntime$call_library` (atom_codes, format,
   findall, etc.). Returns `TRUE`, `FALSE`, or `NULL` (= not
   handled).
5. **Builtin** -- `WamRuntime$call_builtin` (`is/2`, `=/2`, type
   checks, `member/2`, ...). Returns `TRUE` / `FALSE`.

If none match, the call fails and triggers backtracking.

### Recursive-kernel detection

Predicates that match a registered graph-traversal pattern are
classified at codegen time and emitted as a native R fast path
that bypasses the WAM stepping engine entirely. The classifier
lives in
[`src/unifyweaver/core/recursive_kernel_detection.pl`](../src/unifyweaver/core/recursive_kernel_detection.pl)
and is shared with the Haskell / Rust / Elixir targets; this
target wires up all seven registered kernel patterns:
`transitive_closure2`, `transitive_distance3`,
`weighted_shortest_path3`, `transitive_parent_distance4`,
`transitive_step_parent_distance5`, `category_ancestor`, and
`astar_shortest_path4`. Canonical shapes:

```prolog
% transitive_closure2 -- streams reachable nodes
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).

% transitive_distance3 -- streams (target, distance-from-source)
tdist(X, Y, 1) :- edge(X, Y).
tdist(X, Y, D) :- edge(X, Z), tdist(Z, Y, D1), D is D1 + 1.

% weighted_shortest_path3 -- streams (target, shortest-weight)
wsp(X, Y, W) :- edge(X, Y, W).
wsp(X, Y, W) :- edge(X, Z, W1), wsp(Z, Y, RestW), W is RestW + W1.

% transitive_parent_distance4 -- streams (target, parent, distance)
pd(X, Y, X, 1) :- edge(X, Y).
pd(X, Y, P, D) :- edge(X, Z), pd(Z, Y, P, D1), D is D1 + 1.

% transitive_step_parent_distance5 -- streams (target, step, parent, distance)
% step is the FIRST hop neighbour of source on the path X -> ... -> Y.
tspd(X, Y, Y, X, 1) :- edge(X, Y).
tspd(X, Y, M, P, D) :- edge(X, M), tspd(M, Y, _, P, D1), D is D1 + 1.

% category_ancestor -- streams reachable ancestors with depth cap.
% max_depth/1 must be asserted before write_wam_r_project/3 runs;
% the detector reads it at codegen time and the cap is embedded
% in the lowered function. Visited and Hops are inputs / discarded
% outputs that the native impl doesn't touch.
:- assertz(user:max_depth(3)).
ca(Cat, Anc, Visited, 0) :- \+ member(Cat, Visited), edge(Cat, Anc).
ca(Cat, Anc, Visited, Hops) :- \+ member(Cat, Visited),
    edge(Cat, Mid), ca(Mid, Anc, [Cat | Visited], Hops0),
    Hops is Hops0 + 1.

% astar_shortest_path4 -- goal-directed shortest-path search.
% direct_dist_pred/1 names the heuristic predicate (arity 3:
% h(N, Goal, H)); the detector falls back to the edge predicate
% when not set. Dim is a passthrough arg the native impl ignores.
% Result is the single shortest distance source->target.
% Admissibility of the heuristic is the user's responsibility;
% non-admissible heuristics may produce non-optimal results.
:- assertz(user:direct_dist_pred(h_dist/3)).
astar(X, Y, _, W) :- edge(X, Y, W).
astar(X, Y, D, W) :- edge(X, Z, W1), astar(Z, Y, D, RestW),
    W is W1 + RestW.
```

The runtime helpers BFS (`transitive_closure2`,
`transitive_distance3`) or run Dijkstra (`weighted_shortest_path3`)
from a ground source over the underlying edge predicate (invoked
via `iterate_goal`, so the edges can be a fact-table, a dynamic
store, a WAM-compiled predicate, or any other registered dispatch
path) and stream results via iter-CPs. Source must be ground;
target may be ground (check) or unground (enumerate); for the
distance / weight variants, those args are computed (always
unground at the call). Note that the Prolog source clauses
enumerate *all* paths, but the native fast paths return the
*shortest* per reachable node -- that's the labelling implied by
the kernel name and matches the Haskell / Rust semantics.

The lowered function is registered in `program$lowered_dispatch`,
so `Call` and `Execute` instructions for kernel-detected
predicates pick up the fast path before the WAM array tier --
not just direct R-API calls through the per-pred wrapper.

Disable per-call with `kernel_layout(off)` in Options or
globally via `user:wam_r_kernel_layout(off)`.

### Fact-table lowering

Predicates whose every clause is just `get_constant + proceed` are
classified as pure fact tables and lowered to a flat R list of arg
tuples plus a one-line dispatch function -- the WAM stepping
engine isn't entered at all. **Per-arg hash indexes** (one R env
per arg position, each keyed by `"a<atom-id>"` / `"i<int-val>"`)
route any ground arg to a bucket lookup; the dispatcher picks the
smallest matching bucket among bound atom/int args (most
selective) and iterates just that bucket, with per-tuple unify
filtering the rest of the conditions. A query with all args
unbound (or all bound to floats / structs) falls back to a linear
scan over the full tuple list.

Memory: O(N · F) for F facts and arity N -- per-arg, not per-arg-
*combination*, so no `2^N` composite-key blowup. Each fact
contributes at most one entry per arg position. The codegen emits
`<pred>_index_arg<K>` envs (K = 1..N) bundled into
`<pred>_indexes <- list(...)`; the runtime equivalent for
externally-loaded facts is `WamRuntime$build_fact_indexes(facts,
arity)`.

**Range queries** on integer-valued arg positions use a separate
sorted-by-value index. For every numeric (Int/FloatTerm) position
the codegen emits `<pred>_sorted_arg<K> <- list(vals = c(...),
idxs = c(...))` -- two parallel vectors sorted ascending by value
-- and bundles them into `<pred>_range_indexes <- list(arg1 =
<sorted-or-NULL>, ...)`. The bundle is registered in
`program$fact_range_indexes` at program-init time. The
runtime-exposed builtin

```prolog
fact_in_range(+PredArity, +ArgPos, +Lo, +Hi, ?ArgList)
```

binary-searches the `vals` vector for the index range covering
`[Lo, Hi]` (inclusive), extracts the corresponding `idxs` subset,
and iterates matching tuples via the existing
`fact_table_iter_subset` enumeration path (so multi-solution
backtracking just works). `ArgList` is a Prolog list of length
`Arity` that gets unified against each matching fact's args. The
builtin fails if the predicate isn't fact-tabled, the position
doesn't have a sorted index (atom-only column), `Lo`/`Hi` aren't
numeric, or `ArgList` isn't a proper list of the predicate's
arity. Memory: O(F) per numeric-valued arg position, on top of
the hash-index O(N · F).

The bench (`tests/benchmarks/wam_r_fact_source_bench.pl`) shows
modest per-query wins (~10% at N=100 chains, querying the deepest
element) over the WAM `switch_on_constant` path. The bigger wins
are structural: codegen-time savings, simpler emitted R, and now
selective dispatch on any ground arg (not just the first).

Disable per-call with `fact_table_layout(off)` in Options, or
globally via the multifile `user:wam_r_fact_layout(off)` fact.

For the design rationale -- per-arg vs. composite indexes,
smallest-bucket selection, alternatives ruled out (composite,
successive hashing, bitmap intersection), and how this aligns
with the parameterized C# query runtime -- see
[`design/WAM_R_FACT_INDEXING.md`](design/WAM_R_FACT_INDEXING.md).

### External fact sources

Mirrors the Scala target's `scala_fact_sources` option. Users can
declare a predicate's facts as a separate file. Three backends are
supported today:

- `file('data.csv')` -- comma-separated rows, one fact per row.
  Any arity.
- `grouped_by_first('data.tsv')` -- tab-separated rows shaped as
  `key<TAB>v1<TAB>v2<TAB>...`; the loader explodes each row into
  separate `(key, vK)` tuples. Arity-2 only.
- `lmdb('path.lmdb')` -- on-disk LMDB env containing one fact per
  key, value encoded as TAB-separated `tag:payload` fields. Any
  arity. Requires an R LMDB binding (see *LMDB install* below);
  loader returns an empty fact table with a warning if the binding
  is absent. Step-1 semantics: load-everything (treats LMDB as a
  serialization format); step-2 probe-on-demand is a follow-up.

```prolog
:- write_wam_r_project(
       [user:cp/2, user:test/0],
       [intern_atoms([alice, bob, carol]),
        r_fact_sources([source(cp/2, file('data.csv'))])],
       '/tmp/proj').

:- write_wam_r_project(
       [user:parent/2, user:test/0],
       [intern_atoms([alice, bob, carol]),
        r_fact_sources([source(parent/2,
                                grouped_by_first('parents.tsv'))])],
       '/tmp/proj').

:- write_wam_r_project(
       [user:edge/2, user:test/0],
       [intern_atoms([alice, bob, carol]),
        r_fact_sources([source(edge/2, lmdb('edges.lmdb'))])],
       '/tmp/proj').
```

#### LMDB encoding

Values stored in the LMDB env are TAB-separated strings. Each field
has the shape `tag:payload` where:

- `a:<string>` -- atom (re-interned via the program's intern table
  at load time)
- `i:<integer>` -- IntTerm (decimal text, parsed via `as.integer`)
- `f:<double>` -- FloatTerm (decimal or scientific, parsed via
  `as.numeric`)

Keys are opaque to the loader (it iterates all keys via
`env$list()`); ordering doesn't affect correctness since the result
feeds the same `build_fact_indexes` pipeline as the inline / CSV /
grouped-TSV backends. The runtime writer
(`WamRuntime$lmdb_write_facts`) emits keys as
`sprintf("%010d", i)` so a binary cursor walk returns tuples in
insertion order, which is convenient for inspection with `mdb_dump`
but not required.

To produce an LMDB env at setup time, either pre-bake one with any
LMDB-aware tool (`mdb_load`, the Scala backend's writer, etc.) or
call the runtime helper from R:

```r
source("R/generated_program.R")  # or sourcing just runtime.R
WamRuntime$lmdb_write_facts("edges.lmdb",
                            list(list(Atom(1), Atom(2)),  # alice, bob
                                 list(Atom(2), Atom(3))), # bob, carol
                            intern_table)
```

#### LMDB install

The runtime auto-detects an R LMDB binding; in priority order:

1. [`thor`](https://cran.r-project.org/package=thor) (Mozilla wrapper,
   recommended)
2. [`lmdbr`](https://cran.r-project.org/package=lmdbr) (lighter
   alternative)

System LMDB is required first:

```sh
apt install liblmdb-dev   # Debian / Ubuntu
brew install lmdb         # macOS
```

Then install the R binding of your choice:

```sh
R -e 'install.packages("thor")'   # or "lmdbr"
```

If no binding is available at runtime, `read_facts_lmdb` logs a
message and returns an empty fact table -- downstream queries
fail (no solutions) rather than erroring out. Tests that depend on
LMDB auto-skip when no binding is present.

The predicate has **no Prolog clauses** -- the codegen emits a
runtime loader that reads the file at program-init time and
populates the standard fact-table data structures
(`<pred>_facts` / `<pred>_indexes`). The predicate then dispatches
via the same `fact_table_dispatch` path used by inline-clause
fact tables (PR #1921), so per-arg hash indexing, multi-
solution backtracking, and `iterate_goal` integration all work
the same way regardless of backend.

For both formats: lines starting with `#` and blank lines are
skipped. Fields that parse as a finite numeric become `IntTerm` /
`FloatTerm`; everything else interns as an atom. CSV trims each
field; grouped-by-first additionally drops empty fields and
silently skips rows with no values (just a key).

For atoms that don't appear in any compiled WAM body but are
needed by the loaded facts, declare them via `intern_atoms(...)`
so the intern table includes them at startup -- otherwise the
loader interns them lazily, which is fine but means some atoms
get IDs outside the codegen-known range.

The CLI path works the same as inline fact tables: the
predicate's body is a single `Execute("P", A)` instruction that
falls through to the lowered_dispatch tier. To add a new backend,
add a `fact_source_loader_call/4` clause for the Spec shape and a
`WamRuntime$read_facts_<shape>(...)` runtime helper.

### Emit modes

The Phase-3 lowered emitter can replace the instruction-array body
of a predicate with hand-emitted R that calls runtime helpers
directly. This skips the per-instruction `step` dispatch and is a
useful hot-path optimisation.

- `interpreter` (default) -- everything goes through the instruction
  array.
- `functions` -- every predicate the lowered emitter recognises is
  emitted as native R; non-lowerable ones fall back to the array
  path transparently.
- `mixed([Pred/Arity, ...])` -- only the listed predicates are tried
  for lowering.

A lowered predicate is dispatched the same way as a compiled label:
the wrapper calls the lowered function instead of stepping the
instruction array. Failure semantics are identical.

### Mode analysis (`mode_comments`, `mode_specialise`)

WAM-R mode-analysis integration. The shared
`core/binding_state_analysis.pl` analyser is already wired into the
WAM compiler for `=../2` / `functor/3` / `arg/3` / `not_member_set`
specialisations. The WAM-R lowered emitter wires the same analyser
output into two consumers:

**1. Visibility (`mode_comments(on)`)** -- prepend a
`# Mode analysis:` comment block above each lowered function so
developers can inspect what the analyser inferred:

```prolog
:- assertz(user:mode(p(+, -))).

:- write_wam_r_project(
       [user:p/2, ...],
       [emit_mode(functions), mode_comments(on)],
       '/tmp/proj').
```

```r
# Mode analysis (phase 1, visibility-only):
#   clause 1 head: [Arg1-bound, Arg2-unbound]   (mode_decl=[input, output])
#
# Lowered: p/2  (deterministic single-clause)
lowered_p_2 <- function(program, state) { ... }
```

Head-arg state legend: `bound` (mode +), `unbound` (mode -),
`unknown` (mode ?, no declaration, or analyser can't prove either
way). Non-variable head patterns (e.g. `p(alice)`) report `bound`
because head unification by definition binds them, but this is
visibility-only -- the specialisation below uses the mode
declaration directly, not the visibility output.

**2. Specialised inline emission (default ON)** -- when the mode
declaration says `+` for a head arg position, the lowered emitter
inlines `get_constant` head matches as:

```r
{ val_ <- WamRuntime$deref(state, WamRuntime$get_reg(state, AIdx))
  if (is.null(val_) || !identical(val_, CTerm)) return(FALSE) }
```

instead of delegating to `WamRuntime$step`. Saves one `step()` call
(list construction + function call + switch dispatch) per
get_constant. Opt out via `mode_specialise(off)` if you need the
unspecialised codegen (testing / regression bisection).

**Soundness:** the specialisation skips the "rebind unbound to the
constant" branch of `step`'s GetConstant handler, since mode `+`
promises A_k is bound at clause entry. If a caller passes an
unbound term where the mode says `+`, the inline form returns
`FALSE` (treating unbound as a tag mismatch) where the step path
would have bound it. Document; user is responsible for honest mode
declarations.

The mode declaration uses `user:mode/1` with `+` / `-` / `?`
shorthand (input / output / any), the same convention
`demand_analysis:read_mode_declaration/3` reads. See
[`design/WAM_R_MODE_ANALYSIS_PLAN.md`](design/WAM_R_MODE_ANALYSIS_PLAN.md)
for the phase roadmap and measured impact.

### `is/2` specialisation (phase 3)

Two complementary specialisations targeting the `is/2` arithmetic
builtin (dominant cost on arith-heavy recursive predicates):

**Runtime fast-path** (in `WamRuntime$call_builtin`'s is/2 branch).
When the expression is a 2-arg arithmetic struct (`+`, `-`, `*`,
`//`, `mod`) with both operands derefing to ints, bypasses the
recursive `eval_arith` walk + `arith_to_term` dispatch and
fast-binds when the target is unbound. Falls through to the
original slow path for floats, nested expressions, unbound vars in
the RHS, etc. Always active, no codegen flag needed.

**Lowered-emitter inline.** For `builtin_call is/2 2` lines in a
lowered function body, the emitter emits inline R that bypasses
`WamRuntime$step` → `WamRuntime$call_builtin` (saving 2 function
calls + 2 switch lookups per is/2 hit). Always fires when the
lowered emitter handles the clause; the runtime fast-path then
applies inside the inline body.

Note: both specialisations only affect `is/2` calls that reach the
lowered emitter (phase-3 b) or the array path / step (phase-3 a). A
multi-clause predicate where clause 2+ contains the is/2 will go
through the array path for those clauses, so phase-3 (a) handles
it. The lowered-emitter inline (b) only helps for the lowered
clauses (today: deterministic or multi_clause_1's first clause).
See `WAM_R_MODE_ANALYSIS_PLAN.md` phase 4 for the multi_clause_n
extension that would unlock more of the win.

## Supported features

### Control

`true/0`, `fail/0`, `!/0`, `\+/1` (also `not/1`), `=/2`, `\=/2`,
`==/2`, `\==/2`, `@</2`, `@>/2`, `@=</2`, `@>=/2`, `=../2`,
`functor/3`, `arg/3`, `copy_term/2`, `compare/3`, `call/1`.
`call/2..N` flow through the WAM compiler's auto-extension of
goals.

### Arithmetic

`is/2`, `=:=/2`, `=\=/2`, `</2`, `>/2`, `=</2`, `>=/2`. Supported
operators inside `is`-expressions:

- Unary: `-`, `+`, `abs`, `sign`, `sqrt`, `exp`, `log`, `log2`,
  `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`,
  `tanh`, `floor`, `ceiling`, `truncate`, `round`, `\` (bitwise
  not).
- Binary: `+`, `-`, `*`, `/`, `//`, `mod`, `rem`, `min`, `max`,
  `^`, `**`, `atan2`, `gcd`, `/\` (bitand), `\/` (bitor), `xor`,
  `<<`, `>>`.
- Constants: `pi`, `e`, `inf` / `infinite`, `nan`, `epsilon`,
  `max_tagged_integer`, `min_tagged_integer`.

Integer/float promotion follows R's normal type widening; bigints
and rationals are not supported.

### Type checks

`var/1`, `nonvar/1`, `atom/1`, `integer/1`, `float/1`, `number/1`,
`atomic/1`, `compound/1`, `is_list/1`, `ground/1`.

### Lists

`member/2` (multi-solution via iter-CP), `memberchk/2`, `length/2`
((+, -), (-, +), and (-, -) modes), `append/3` ((+, +, -) mode
and finite split mode over a ground third list), `reverse/2`
(deterministic if either side bound), `last/2`, `nth0/3`, `nth1/3`,
`select/3` (multi-solution over list positions), `delete/3` (==/2 semantics),
`permutation/2` ((+, +) check + (+, -) identity), `numlist/3`,
`sort/2` (sort + dedup, standard order of terms), `msort/2` (sort,
keep duplicates), `maplist/2..3`.

### Higher-order / meta

`call/1..N`, `\+/1`, `once/1`, `forall/2`, `findall/3`, `bagof/3`,
`setof/3`, `phrase/2`, `phrase/3`. Aggregators back-end via two
paths:

- Compiled goals push an `aggregate` CP at `BeginAggregate`,
  collect on every backtrack, and finalise at `EndAggregate`.
- Library-level aggregators (`bagof`, `setof`, runtime-only
  predicates) call `WamRuntime$collect_solutions` or
  `WamRuntime$collect_bag_groups`, which snapshot state, drive the
  goal via `iterate_goal`, and restore state fully before returning.
  `bagof/3` and `setof/3` honor `^/2` existential scope and group
  non-existential free variables; additional witness groups are
  exposed through normal backtracking.

`iterate_goal` has R-level fast paths for `member/2`, `between/3`,
`,/2` conjunctions over the above, and dynamic predicates -- so
`findall(X, member(X, [1,2,3]), L)` works without ever entering the
WAM stepping loop.

### Strings / atoms

`atom_codes/2` (also `string_codes/2`), `atom_chars/2` (also
`string_chars/2`), `atom_length/2` (also `string_length/2`),
`atom_concat/3` (also `string_concat/3`), `atom_number/2`,
`atom_string/2`, `string_to_atom/2`, `number_codes/2`,
`number_chars/2`, `string_upper/2`, `string_lower/2`,
`string_code/3`, `split_string/4`, `sub_atom/5` (forward mode:
Before and Length given), `char_type/2` (forward mode: char given;
types `alpha`, `alnum`, `digit`, `space`, `upper`, `lower`,
`punct`, `ascii`, `csym`, `csymf`, `newline`, `white`),
`term_to_atom/2`, `read_term_from_atom/2`,
`read_term_from_atom/3` (the options arg is accepted but ignored
in this scaffold), `tab/1`.

The `atom`/`string` distinction is collapsed at the WAM-text level
(SWI's WAM emitter doesn't preserve it), so `string_*` predicates
alias onto their `atom_*` counterparts.

`term_to_atom/2` reverse mode parses canonical structural Prolog
(`f(a, b)`, `[1, 2, 3]`, atoms, integers, floats) **and operator
notation**: arithmetic (`+`, `-`, `*`, `/`, `**`, `^`, ...),
comparison (`=`, `\=`, `==`, `\==`, `=:=`, `=\=`, `<`, `>`, `=<`,
`>=`, `@<`, `@>`, `@=<`, `@>=`, `=..`), control (`:-`, `-->`, `;`,
`->`, `,`), and prefix (`\+`, `:-`, `?-`, `-`, `+`, `\`).
Precedence and associativity follow the standard Prolog operator
table.

User-defined operators are supported via the runtime `op/3` builtin
and the codegen `r_op_decls(...)` option. `op(+Priority, +Type,
+Name)` mutates the parser's operator table at runtime; subsequent
`read_term_from_atom/2,3` and CLI arg parses see the new operator.
Type is one of `xfx`, `xfy`, `yfx`, `fx`, `fy`, `xf`, `yf` (binary
infix, prefix, and postfix). Priority `0` removes the operator.
Name may be an atom or a proper list of atoms. `current_op/3`
enumerates the table with backtracking. When a name is registered
as both infix and postfix, the parser tries infix first and only
falls through to postfix when no infix entry fits the current
precedence ceiling -- matches SWI behaviour. The parser does not
distinguish `xf` from `yf` at chained-postfix sites: both accept
`5!!` rather than the ISO-strict `xf` rejecting it. Real code rarely
relies on that distinction; the existing prefix path makes the same
simplification (`fx` and `fy` are treated identically).

For compile-time declarations (so the operator is known before any
runtime call), pass `r_op_decls([op(P, T, N), ...])` to
`write_wam_r_project/3`. Each declaration emits a
`WamRuntime$op_set("<N>", <P>L, "<T>")` line at program init,
ahead of CLI parsing.

### I/O

Stdout family: `write/1`, `writeln/1`, `print/1`, `nl/0`,
`format/1`, `format/2`. Stream-aware variants: `write/2`,
`writeln/2`, `nl/1`, `format/3`. Format control sequences
supported: `~w`, `~a`, `~d`, `~p`, `~s`, `~n`, `~~`.

### Streams (file I/O)

`open/3`, `close/1`, `read/2`. Streams are opaque `stream(<id>)`
structs whose integer id keys into `program$streams`, an R env
that maps id -> R connection. `open/3` accepts modes `read`,
`write`, `append` and returns a `stream(<id>)` term; `close/1`
closes the connection and removes the entry. `read/2` accumulates
lines into a buffer until the trimmed buffer ends with `.` AND the
operator-precedence parser accepts the buffer minus the trailing
`.`. This handles single-line clauses (the common case) and terms
whose source spans multiple lines (open paren on one line, args on
the next, closing `).` on a third). When the trimmed buffer ends
with `.` but the parser fails, the `.` is treated as being inside a
quoted atom / string / unclosed compound and reading continues. EOF
on an empty buffer binds the term to the atom `end_of_file` (matches
SWI semantics); EOF on a non-empty buffer makes one final parse
attempt and fails if that doesn't yield a term.

### DCG (`-->`)

DCG rules are translated at SWI's term-expansion time (or at
runtime via `dcg_translate_rule/2`) into ordinary clauses with
two extra difference-list args, so by the time the WAM compiler
reads them they look like normal predicates and need no special
handling. The runtime supports `phrase/2` and `phrase/3` to
bridge the user-level call into the translated `<head>/N+2` form.

```prolog
:- use_module(library(dcg/basics)).
dcg_translate_rule((seq(0) --> []), C0), assertz(user:C0),
dcg_translate_rule(
    (seq(N) --> {N > 0}, [N], {N1 is N - 1}, seq(N1)), Cn),
assertz(user:Cn),
% phrase(seq(3), [3, 2, 1]) succeeds via seq/3.
```

### Dynamic predicates

`assertz/1`, `asserta/1`, `retract/1` (multi-solution: iter-CP
walks the snapshot taken at the call, removing each match in turn
on backtracking), `abolish/1` (takes `Name/Arity`), `clause/2`
(multi-solution introspection: same iter-CP shape as
`retract/1`, without the removal side-effect; facts are surfaced
with body = `true`). Clauses are stored in `program$dynamic` (an
R env, so mutations propagate across queries). Multi-clause
dynamic predicate calls are dispatched through a `dynamic` CP
that walks the clause list on backtracking.

### Exception handling

`throw/1`, `catch/3`. Implemented on R's `tryCatch`: `throw/1`
raises a `prolog_throw` condition carrying a `copy_term`'d snapshot
of the thrown term; `catch/3` unwinds the trail and CP stack to its
entry snapshot before unifying with the catcher and running the
recovery. Uncaught throws at the top level become query failure.

## Supported WAM instructions

`call`, `execute`, `proceed`, `jump`, `allocate`, `deallocate`, the
`get_*` / `put_*` / `set_*` / `unify_*` family (atom, integer,
float, variable, value, list, structure), `try_me_else` /
`retry_me_else` / `trust_me`, `switch_on_constant` (plus `default`
fallthrough), `cut_ite`, `builtin_call`, `call_foreign`, and
`begin_aggregate` / `end_aggregate` (the bracketing instructions
for `findall/3`-style goals when the surrounding predicate is
compiled).

A runnable variant of this example -- with an R foreign handler
for `age_of/2` and a `findall/3` aggregator -- lives in
[examples/wam_r_demo/](../examples/wam_r_demo/).

## End-to-end example: ancestor with arithmetic guard

```prolog
:- use_module('src/unifyweaver/targets/wam_r_target').

:- dynamic user:parent/2.
user:parent(alice, bob).
user:parent(bob,   carol).
user:parent(carol, dave).

user:ancestor(X, Y) :- user:parent(X, Y).
user:ancestor(X, Y) :- user:parent(X, Z), user:ancestor(Z, Y).

% Find every ancestor pair X ~> Y where the names differ in length.
user:long_path(X, Y) :-
    user:ancestor(X, Y),
    atom_length(X, LX),
    atom_length(Y, LY),
    LX =\= LY.

:- initialization((
    write_wam_r_project(
        [user:parent/2, user:ancestor/2, user:long_path/2],
        [ module_name('ancestor.demo'),
          intern_atoms([alice, bob, carol, dave])
        ],
        '/tmp/ancestor_demo'),
    halt
)).
```

```bash
cd /tmp/ancestor_demo
Rscript R/generated_program.R 'long_path/2' alice carol
# -> true
Rscript R/generated_program.R 'long_path/2' alice bob
# -> false
```

For multi-solution use, source the program from R and drive it
manually:

```r
source("R/generated_program.R")
state <- WamRuntime$new_state()
WamRuntime$promote_regs(state)
WamRuntime$put_reg(state, 1L, Atom(WamRuntime$intern(intern_table, "alice")))
WamRuntime$put_reg(state, 2L, Unbound("Y0"))
state$pc <- shared_labels[["ancestor/2"]]
state$cp <- 0L
WamRuntime$run(shared_program, state)
# Inspect bindings via WamRuntime$deref(state, ...)
```

## Limitations

- **WAM-text quoting collision**. The atom `'42'` and the integer
  `42` both serialise as `set_constant 42` in SWI's WAM emitter,
  so the codegen can't distinguish them. The runtime's `atom_*`
  family is intentionally lenient (accepts ints/floats with their
  decimal name) to compensate. To round-trip an atom-of-digits
  reliably, build it via `atom_codes/2`.
- **`retract/1` immediate-update view**. `retract/1` is multi-solution
  via an iter-CP, and the iteration reads the live clause list afresh
  on every retry. Clauses asserted *during* the iteration are visible
  to subsequent solutions; clauses already retracted (or retracted by
  another call mid-iteration) are simply skipped. This deliberately
  diverges from SWI's logical-update view (which would hide mid-
  iteration asserts); see `streams_multiline_read_e2e_rscript`-adjacent
  `multi_solution_retract_e2e_rscript` for the locked-in test case.
- **Full ISO/SWI compatibility for every `bagof/3` / `setof/3` edge**.
  The runtime now groups non-existential free variables and enumerates
  additional witness groups on backtracking, but unusual attributed-var
  or cyclic-term cases still need broader compatibility coverage.
- **`length/2` (-, -)** is an unbounded generator. It enumerates
  `([], 0)`, then one-element lists, two-element lists, and so on via
  an iter-CP; callers should add guards when collecting solutions.
- **`between/3` (+, +, -)** in compiled-goal context produces an
  iter-CP and works as a generator. In runtime-aggregator context
  the R-level fast path enumerates directly. Mixed contexts that
  need iter-CP-driven generation from inside library dispatch are
  not exhaustively covered.
- **Stream-aware predicate set is partial**. `open/3`, `close/1`,
  `read/2`, `write/2`, `writeln/2`, `nl/1`, and `format/3` are
  supported, but multi-line term reads, `current_input/1` /
  `current_output/1`, `set_input/1` / `set_output/1`, character
  I/O (`get_char/1,2`, `put_char/1,2`), and binary streams are not.
- **Float arithmetic is `double` only**; bigints and rationals are
  not supported. Integer overflow follows R's normal `integer`
  semantics.

## Testing

The full test suite lives in
[tests/test_wam_r_generator.pl](../tests/test_wam_r_generator.pl)
and contains 57 tests covering both structural assertions on the
generated source and end-to-end execution via `Rscript`. The
`*_e2e_rscript` tests auto-skip when `Rscript` is not on `PATH`.

Run it:

```bash
swipl -g 'use_module(library(plunit)),consult("tests/test_wam_r_generator.pl"),run_tests,halt' -t 'halt(1)'
```

Coverage map (e2e tests, by feature group):

| Test | Group |
|---|---|
| `foreign_handler_e2e_rscript` | foreign-handler dispatch |
| `builtin_arith_e2e_rscript` | `is/2`, comparisons |
| `extended_builtins_e2e_rscript` | type checks, `=/2`, `\=/2`, `=../2` |
| `term_inspection_e2e_rscript` | `functor/3`, `arg/3`, `copy_term/2` |
| `list_atom_builtins_e2e_rscript` | `member/2`, `length/2`, `atom_codes/2`, ... |
| `extended_arith_e2e_rscript` | trig, log, bitwise |
| `stdlib_round4_e2e_rscript` | `compare/3`, `nth*/3`, `select/3`, `delete/3`, `succ/2` |
| `negation_meta_call_e2e_rscript` | `\+/1`, `call/1..N` |
| `higherorder_listutil_e2e_rscript` | `maplist/2..3`, `reverse/2`, `last/2` |
| `io_between_e2e_rscript` | `write/1`, `format/2`, `nl/0`, `between/3` |
| `string_ops_e2e_rscript` | full string family |
| `dynamic_preds_e2e_rscript` | `assertz/1`, `asserta/1`, `retract/1`, `abolish/1` |
| `findall_e2e_rscript` | `findall/3` (compiled + runtime aggregator) |
| `bagof_setof_once_forall_e2e_rscript` | `bagof/3`, `setof/3`, `once/1`, `forall/2` |
| `enumerable_builtins_e2e_rscript` | enumerable `member/2` / `between/3` inside aggregators |
| `catch_throw_dyn_aggregator_e2e_rscript` | `catch/3`, `throw/1`, dynamic-pred aggregation |
| `stdlib_polish_e2e_rscript` | `numlist/3`, `tab/1`, `sub_atom/5`, `char_type/2`, `term_to_atom/2` |
| `operator_parser_e2e_rscript` | operator-precedence parsing (`+`, `*`, `^`, `=:=`, `\+`, `,`, ...) |
| `op_3_runtime_e2e_rscript` | runtime `op/3` builtin: add infix / prefix custom ops, xfy right-associativity, `op(0, ...)` removal, `current_op/3` enumeration |
| `op_3_decl_e2e_rscript` | codegen `r_op_decls([op(P, T, N), ...])` option seeds the operator table at program init; covers atom-name and list-of-names forms |
| `multi_solution_retract_e2e_rscript` | multi-solution `retract/1` via iter-CP |
| `nested_ite_e2e_rscript` | nested `(C -> T ; E)` / `(A ; B)` inside aggregate bodies, disjunction arms, and ite-branches (compile_inner_call_goals recursion) |
| `bagof_setof_existential_e2e_rscript` | `^/2` existential scope, witness grouping/backtracking in `bagof`/`setof`, and compiled `findall` over runtime grouped aggregators |
| `cli_arg_parser_e2e_rscript` | structured CLI args (lists, structs, expressions) parse via the runtime parser |
| `base_name_clash_e2e_rscript` | predicates named after base R functions (`c`, `t`, `q`, `cat`) don't shadow them |
| `read_term_clause_e2e_rscript` | `read_term_from_atom/2,3` and multi-solution `clause/2` |
| `dcg_e2e_rscript` | DCG `-->` rules + `phrase/2,3` (recursive grammars, prefix-with-rest) |
| `streams_e2e_rscript` | `open/3`, `close/1`, `read/2`, `write/2`, `writeln/2`, `format/3` round-trip |
| `streams_multiline_read_e2e_rscript` | `read/2` across multi-line clauses: buffer accumulates lines until trimmed buffer ends with `.` and parser accepts |
| `fact_table_e2e_rscript` | fact-table lowering: hash-indexed dispatch, multi-solution backtracking, atoms + integers |
| `fact_in_range_e2e_rscript` | `fact_in_range/5` range query on fact tables (sorted-arg index + binary search + iter-CP), inclusive bounds, atom-only column = fail, out-of-range ArgPos = fail |
| `findall_template_in_struct_arg_e2e_rscript` | findall template var initialised via `put_variable Y, Y` self-init so a later `put_structure` on A1 (when the inner goal's first arg is a compound) doesn't auto-bind a shared ref into the template |
| `fact_table_multi_arg_index_e2e_rscript` | per-arg fact-table indexes: dispatch picks smallest matching bucket among bound atom/int args (arg2-only, arg3-only, multi-arg-bound queries) |
| `kernel_tc2_e2e_rscript` | recursive-kernel detection: `transitive_closure2` BFS over a fact-table edge predicate |
| `kernel_td3_e2e_rscript` | recursive-kernel detection: `transitive_distance3` BFS-with-depth over a fact-table edge predicate |
| `kernel_wsp3_e2e_rscript` | recursive-kernel detection: `weighted_shortest_path3` Dijkstra over a weighted fact-table edge predicate |
| `kernel_tpd4_e2e_rscript` | recursive-kernel detection: `transitive_parent_distance4` BFS with parent tracking |
| `kernel_tspd5_e2e_rscript` | recursive-kernel detection: `transitive_step_parent_distance5` BFS with step + parent + distance |
| `kernel_ca_e2e_rscript` | recursive-kernel detection: `category_ancestor` BFS with depth-cap + visited-set cycle detection |
| `kernel_astar4_e2e_rscript` | recursive-kernel detection: `astar_shortest_path4` goal-directed search with user heuristic |
| `external_fact_source_e2e_rscript` | external CSV fact sources via `r_fact_sources([source(P/A, file(...))])` |
| `external_fact_source_grouped_tsv_e2e_rscript` | external grouped-by-first TSV fact sources (arity-2): `r_fact_sources([source(P/2, grouped_by_first(...))])` |
| `external_fact_source_lmdb_e2e_rscript` | external LMDB fact sources: `r_fact_sources([source(P/A, lmdb(...))])`, load-everything semantics, tab-encoded `tag:payload` values; auto-skips when no R LMDB binding (`thor` / `lmdbr`) is installed |
| `op_3_postfix_e2e_rscript` | runtime postfix-operator support: `op(P, xf|yf, N)`, parser wraps primary as postfix struct, mixed infix+postfix, yf chaining, `current_op/3` enumeration, `op(0, ...)` removal |
| `cut_barrier_after_helper_e2e_rscript` | `!/0` truncates `state$cps` to the predicate's call-site depth, dropping leftover CPs from multi-clause helpers and the predicate's own try-chain CP in one shot |
| `stack_frame_cleanup_on_backtrack_e2e_rscript` | failed clauses' env frames get popped on backtrack via the CP's `stack_len` snapshot, so the outer predicate's `Deallocate` doesn't pop a stale frame and re-enter post-call code |
| `cut_ite_barrier_after_helper_e2e_rscript` | `( A -> B ; C )` soft-cut commits past CPs that A's evaluation left alive (the codegen marks the if-then-else's `try_me_else` as `try_me_else_ite`, the runtime tags the CP `kind="ite"`, and `CutIte` truncates `state$cps` back to that CP's pre-push depth) |
| `nested_if_then_else_e2e_rscript` | chained `( A -> B ; C -> D ; E )` compiles as nested cut_ite/try_me_else_ite pairs (each `->` in Else position is recognised recursively rather than emitted as a stub `Call("->", 2)`) |
| `deeply_nested_ite_compiles` | clause_body_analysis no longer stack-overflows on triple-nested if-then-else bodies (the `nonvar` guard on `disjunction_alternatives/2` keeps it from infinitely-expanding an unbound goal that passes through the analyser) |
| `strict_xf_chain_fails_e2e_rscript` | runtime parser enforces strict (xf / fx) vs non-strict (yf / fy) lhs-precedence rules: `5!!` parses with `op(100, yf, '!')` but fails with `op(100, xf, '!')`; `neg neg foo` parses with `op(900, fy, neg)` but fails with `op(900, fx, neg)` |
| `phase3_multi_clause_e2e_rscript` | Phase-3 lowered emitter (multi-clause) |
| `lowered_emitter_e2e_rscript` | Phase-3 lowered emitter (single-clause) |
| `mode_analysis_phase1_comments` | Mode-analysis visibility: `mode_comments(on)` option prepends `# Mode analysis:` block to each lowered function with per-clause head-binding states; covers `+`, `-`, `?`, undeclared mode shapes |
| `mode_analysis_phase2_get_constant_inlined` | Mode-analysis phase 2: structural assertion that `get_constant` head match is emitted as inline `WamRuntime$deref + identical()` when the target A-register's declared mode is `+`; falls back to `WamRuntime$step` when no mode declaration or `mode_specialise(off)` |
| `mode_analysis_phase2_get_constant_e2e_rscript` | Mode-analysis phase 2: e2e correctness -- a predicate with `:- mode(p(+))` and three clauses compiles + runs via Rscript, queries return correct true/false matching across the multi-clause backtracking path |
| `mode_analysis_phase3_is_inlined` | Mode-analysis phase 3: structural assertion that `builtin_call is/2 2` is emitted as inline `WamRuntime$eval_arith + bind/unify` in the lowered function (instead of `WamRuntime$step(... BuiltinCall("is/2", 2))`) |
| `mode_analysis_phase3_is_e2e_rscript` | Mode-analysis phase 3: e2e correctness -- a predicate using `is/2` with simple binary int op (runtime fast-path), nested arith (slow path), and negative-result arith compiles + runs via Rscript with correct values |

## Contributing

Every new feature should ship with an `*_e2e_rscript` smoke test in
[tests/test_wam_r_generator.pl](../tests/test_wam_r_generator.pl)
that asserts on actual `Rscript` stdout. Generator-level structural
tests have repeatedly missed real runtime bugs in this target;
hand-checking the actual runtime output is the only reliable
signal.

## Benchmark

A small fact-source bench compares the WAM-compiled and
foreign-handler paths on a chain of `c0 -> c1 -> ... -> cN`:

```bash
swipl -g main -t halt tests/benchmarks/wam_r_fact_source_bench.pl
swipl -g main -t halt tests/benchmarks/wam_r_fact_source_bench.pl -- 50 --inner 1000
```

For each backend x size combination it emits:

```
RESULT n=<N> backend=<wam|foreign>
       gen=<sec>           # write_wam_r_project/3
       run=<sec>           # cold-start single-shot Rscript invocation
       inner_total=<sec>   # `--bench M` invocation; R startup paid once
       per_iter=<sec>      # inner_total / M
```

Cold-start time (`run`) is dominated by `Rscript` startup; the
`inner_total`/`per_iter` columns amortise that fixed cost over
`M` iterations and reveal the real per-query difference between
backends. Auto-skips when `Rscript` is not on PATH.

The generated program also accepts `--bench N <pred>/<arity> <args>...`
directly when invoked via `Rscript`; it warmups for 5% of the
iterations (capped at 50) so the R-level data structures settle
before timing, then prints `BENCH n=<N> elapsed=<sec> last=<true|false>`.

### Parser benchmark (inline vs compiled-from-Prolog)

`tests/benchmarks/wam_r_parser_bench.pl` measures the cost of the
inline R parser (`WamRuntime$wam_parse_expr` / `parse_term`, called
by `read/2` / `term_to_atom/2` / `read_term_from_atom/2,3` and the
CLI arg parser) versus the cross-target Prolog parser at
`src/unifyweaver/core/prolog_term_parser.pl` compiled to WAM-R.
Both produce identical terms (verified by
`tests/test_prolog_term_parser_wam_r_compile.pl`); this is purely
a "should we swap?" benchmark.

```bash
swipl -g main -t halt tests/benchmarks/wam_r_parser_bench.pl
swipl -g main -t halt tests/benchmarks/wam_r_parser_bench.pl -- --inner 2000
```

Representative numbers (200 iters each, single-threaded R 4.4):

```
     input  inline (s)  compiled (s)   per-iter (us)       ratio
      atom    0.042000      4.081000         20405.0       97.2x
   integer    0.043000      3.756000         18780.0       87.3x
  compound    0.058000     15.028000         75140.0      259.1x
     arith    0.067000     13.869000         69345.0      207.0x
      list    0.072000     20.261000        101305.0      281.4x
    nested    0.093000     25.770000        128850.0      277.1x
```

The compiled parser is **80-300x slower** than the inline one
across the parser surface (atoms / integers / compounds / operator
expressions / lists / nested compounds). The slowdown comes from
running the Pratt parser through the WAM stepping engine instead
of native R control flow plus inline tagged-list construction.
Per the project directive *"only swap if equivalent or better"*,
the runtime keeps the inline parser as the hot path. The
cross-target parser source remains valuable as the canonical spec
and as the parser-of-record for any future target that doesn't
have a hand-written inline path.

### Rprof profile of the WAM stepping engine

The fact-source bench has a `--profile` mode that drives the
generated program via a small `profile_runner.R` (auto-written
into the project on first use). The runner sources
`generated_program.R` (which is a no-op for its main block since
`sys.nframe() == 0L` is false), warms up, then runs the predicate
N times under `Rprof()` and prints the top 25 hotspots from
`summaryRprof()$by.self`.

```bash
swipl -g main -t halt tests/benchmarks/wam_r_fact_source_bench.pl \
    -- --profile 100 --inner 100000
```

Representative hotspots for the WAM backend, single-row `cp(c0, X)`
query against a 100-row chain, 100000 iterations (R 4.4), with X / A
registers stored as an integer-indexed R list, the `state$bindings`
env accessed via `e[[name]]` (single lookup) and trail rollback via
`e[[name]] <- NULL` (env-delete), and a single-slot state pool that
recycles state envs across `run_predicate` calls:

```
  function                                       self.s  %self  total.s   %tot
  WamRuntime$step                                 1.500  29.0%    2.180  42.2%
  WamRuntime$run                                  0.760  14.7%    2.940  56.9%
  WamRuntime$run_predicate                        0.690  13.3%    5.090  98.5%
  WamRuntime$deref                                0.380   7.3%    0.610  11.8%
  WamRuntime$put_reg                              0.260   5.0%    0.260   5.0%
  WamRuntime$reset_state                          0.240   4.6%    0.360   7.0%
  WamRuntime$get_reg                              0.160   3.1%    0.180   3.5%
```

`step` dispatch and `run` lead. `deref` was overhauled to a single
env subscript -- `state$bindings[[v$name]]` returns the bound value or
NULL in one operation, replacing the prior `exists()` + `get()` pair.
The trail rollback sites (`undo_trail_to` and ~16 backtrack/CP-restore
sites) likewise use `state$bindings[[name]] <- NULL` to delete
bindings, replacing the prior `if (exists) rm()` pattern.

`new_state` no longer appears in the top hotspots: a single-slot pool
(`WamRuntime$state_pool_idle`) parks the state env on
`run_predicate` exit and the next call recycles it via `reset_state`.
That swaps a fresh `new.env()` for the state record (the bindings env
is still freshly allocated each call -- cheaper than rm-ing entries
when bindings is non-empty, and correctness-trivial). Recursive
`run_predicate` calls -- if they ever happen -- see a NULL pool,
allocate fresh, and only re-park if the slot is still NULL on exit.

A prior profile of the same workload before these refactors showed
elapsed = 7.7s (env-keyed regs2). After the regs2 -> integer-indexed
list refactor: 5.7s (~26%). After the bindings + state-pool refactor:
~5.2s (~33% total, ~9% incremental). `deref` self.s 0.520 -> 0.380
(~27%); the `exists` / `rm` calls dropped off the by-self table for
the X / A path entirely; `new_state` dropped from 8.8% to 0%
(replaced by `reset_state` at 4.6%).

Remaining levers, in priority order:
- `step` (29% self): the big `switch(op_name, ...)`. A per-instruction
  closure-based dispatch could shave this.
- `run_predicate` (10-13% self): on.exit + tryCatch + the pool dance
  itself adds overhead. Trimming this would also help, though it's
  inherent to the recycling pattern.
- `run` (14.7% self): the main interpreter loop body.

When adding a builtin:

1. Implement it in `WamRuntime$call_builtin` (if the WAM compiler
   emits it as `BuiltinCall`) or `WamRuntime$call_library` (if it's
   emitted as a plain `Call` / `Execute`).
2. Match the SWI semantics for the modes you support; document
   unsupported modes in the leading comment.
3. Add an end-to-end test that compiles a Prolog program using the
   builtin, runs it via `Rscript`, and asserts on stdout.
