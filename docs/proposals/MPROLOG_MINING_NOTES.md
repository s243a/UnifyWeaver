<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Mining notes: M-Prolog (`mprolog`/N-Prolog) as a design-idea source

## Attribution

- **Project:** M-Prolog, distributed under the name `mprolog` on GitHub and as
  "N-Prolog" in its own in-tree documentation and REPL banner
  (`document/COMPILER.md:8`: `N-Prolog Ver 3.91`).
- **Author:** Kenichi Sasagawa.
- **Source:** <https://github.com/sasagawa888/mprolog>, read from a read-only
  clone at `/home/user/sasagawa888/mprolog` (not vendored into this repo, not
  modified).
- **License:** Modified BSD (`license.txt` in the clone).
- **What this document is:** a recon pass that mines mprolog for **portable
  design ideas**, credited by finding. UnifyWeaver adopts *ideas*, never
  *code*, from mprolog. If any future UnifyWeaver change adapts logic that is
  recognizably derived from mprolog source (rather than an independently
  reimplemented idea), that change must carry the Modified BSD notice and
  attribution alongside it, per mprolog's license terms. Nothing in this
  document is a copy of mprolog source; C snippets quoted below are quoted
  for citation/discussion only, at the length needed to support the claim
  next to them.
- **Honesty note:** every claim below cites `file:line` (or a function name
  plus nearby line range) in the cloned tree, or is explicitly marked
  `(inference)`. Nothing is from memory of Prolog implementations outside
  what was read this session.

## Scope and method

Read: `library/jump.pl` (2,552 lines, the "experimental compiler," despite
the file being named after `library/compiler.pl`'s sibling — see note below),
`library/compiler.pl` (2,551 lines, a separate compiler-adjacent library),
`document/COMPILER.md`, `document/CLPFD.md`, `document/PARA1.md`,
`document/PARA2.md`, `document/CLUSTER.md`, `bench/Bench.md`, `bench/cpbench.pl`,
`mpl.h` (main header, cell/tag layout), `data.c` (`deref`/`unify`/`unbind`,
2,716 lines — read in full via targeted sections), `cell.c` (heap init,
`bindsym`/`findvar`, 663 lines), `gbc.c` (mark-sweep GC, 165 lines), `main.c`
(`prove`/`prove_all`, the interpreter's core resolution loop), `clp.c`
(1,292 lines, read via `CLPFD.md`'s own architecture memo plus the top of the
file). **Not read in depth** (flagged rather than silently skipped):
`parallel.c` (1,245 lines — read only enough to confirm the single mutex
call site cited in F9; the thread-pool/signal-queue mechanics described in
`PARA2.md` were not cross-checked against the C); `bignum.c`, `edlog.c`,
`function.c`, `builtin.c`'s non-indexing-related bulk, `superset.c`,
`extension.c` (the actual `mt_and`/`mt_create` C implementation `PARA2.md`
points at, `extension.c:2139`, was not opened). `clp.c`'s propagation loop
was read only at the level `CLPFD.md`'s own memo documents (see F10) — the
C internals of `fd_generate`/`fd_satisfiable` were skimmed, not traced.

**Naming correction worth recording:** `document/COMPILER.md` documents
`use_module(compiler)` and `compile_file/1`, but the compiler's actual
implementation — `pass1`/`pass2`/`pass3`, the type-analysis predicates, and
every `gen_*` code-generation predicate cited below — lives in
`library/jump.pl`, not `library/compiler.pl` (which is a much smaller,
separate library; its contents were not needed for this recon and are not
cited below). All `jump.pl:NNNN` citations below are to the real compiler.

## Headline

- **The compiler's answer to "how does a compiled predicate yield its second
  solution" is real, working, and directly on-point for our #1 gap**
  (D55's missing resumable protocol): mprolog compiles every nondeterministic
  predicate in a file into ONE C function using GCC's labels-as-values
  (`goto *label`), where a choice point is a saved label plus an indexed
  slot in a per-predicate frame array, and backtracking is "jump to the
  saved label," not "call a function again." See F3 — the single most
  valuable finding in this pass.
- **mprolog's `unify` already does what our Rust fix sketch recommends**:
  dispatch on the raw (undereferenced) shape first, deref only the operand
  side that turns out to be a variable, and only one level at a time. See
  F5 — a working, independent precedent for `WAM_RUST_STATUS.md`'s proposed
  fix to `WamState::unify`'s quadratic deref-both-eagerly bug.
- **mprolog's interpreter has zero clause indexing of any kind** — plain
  linear clause-list scan, confirmed by reading `prove()`'s main loop. This
  independently corroborates D54's finding that first-argument indexing is
  not where the fleet's cost lives; a real, non-toy engine shipped for years
  making the same bet. See F7.
- **A whole-predicate static determinism classifier, with a "tail" class
  that compiles straight to a native C loop with zero choice-point
  machinery**, is a working, mechanical precedent for the "sound
  intermediate" (dispatch only deterministic/clause_chain predicates) our
  own `WAM_RUST_STATUS.md` Path Forward already names. See F2 and F11.
- **The compiler's own headline performance claim does not survive
  scrutiny**: the only numbers in the repo are raw LIPS counts for compiled
  runs, with no paired interpreter baseline and no comparison to any other
  system. There is nothing to cite as "M-Prolog proves compiled Prolog is
  Nx faster." See F4 — a plain negative finding, stated honestly.
- **The one place mprolog's multi-thread story touches a lock is a genuine
  anti-pattern worth avoiding, not adopting**: clause-variable renaming
  temporarily mutates the *shared* clause template in place, which forces a
  single global mutex around every clause try on every thread. See F9 —
  useful as a "don't do this" for T7, alongside its otherwise-reasonable
  explicit fork/join API shape.

## Findings

Each finding is scored against the bottleneck list in the task brief:
(1) per-instruction interpretation overhead / lowered-tier + resumable
protocol gap (D54/D55/D59), (2) `WamState::unify`'s eager-deep-deref
quadratic (D59/`WAM_RUST_STATUS.md`), (3) first-argument indexing bought
little (D54), (4) cut-as-barrier correctness (`WAM_BACKEND_CONVENTIONS.md`
§9).

### F1 — The "compiler" is a real Prolog→C transpiler that shells out to GCC, gated by a whole-program two-pass static analysis

**Verdict: adopt** (idea only — the GCC/dlopen delivery mechanism itself has
no analogue in Rust/JS/Go/CLJS targets).

`compile_file/1` runs `pass1` (retract all `type/3` facts, reconsult the
source under the `compiler` module, run `analize` — `jump.pl:10-14, 54-60`),
`pass2` (`analize` again to a fixpoint, then `retype_halt` reclassifies any
predicate still tagged with the placeholder `halt` type as `nondet`,
`jump.pl:62-72`), `pass3` (emit one `.c` file: prototypes, per-predicate
functions, the big nondet dispatch function, an `init_tpredicate`/
`init_declare` pair, `jump.pl:75-87`), then `invoke_gcc/1` shells out to
`gcc -O3 -w -flto -shared -fPIC ... -o <file>.o <file>.c` and deletes the
intermediate `.c` (`jump.pl:94-104`), producing a `.o` the interpreter
`dlopen`s. `document/COMPILER.md:1-16` confirms this from the user's side
(`compile_file('./tests/fact.pl')` → `phase pass1` / `phase pass2` /
`compiling fact` / `invoke GCC`).

Why this matters for us: the STRUCTURE (analyze the whole program first to
classify every predicate, then emit different code shapes per class, then
compile with a real backend compiler) is exactly the shape of our own
multi-tier lowering (deterministic / clause_chain / multi_clause_1 /
multi_clause_n / ite_lowered, per `docs/WAM_RUST_STATUS.md`'s D55 section).
mprolog is independent evidence that "classify first, specialize codegen
per class" is a viable design for a from-scratch Prolog compiler, not just
an artifact of how we happened to grow ours.

### F2 — Whole-predicate static determinism classification (dyn > tail > det > halt-placeholder > nondet), with a documented cycle-breaking trick

**Verdict: adopt** (idea).

`analize_pred1/3` (`jump.pl:2148-2179`) classifies each predicate, in this
priority order, by structural analysis of its full clause set:
1. `dyn` if declared dynamic (`n_dynamic_predicate/1`).
2. `tail` if `tail_recursive/6` succeeds (`jump.pl:2154-2160`).
3. `det` if `deterministic/5` succeeds (`jump.pl:2161-2167`).
4. `halt` — a placeholder — if `halt_check/4` succeeds (`jump.pl:2168-2174`;
   `halt_check`'s own definition was not read, only its call site and the
   fact that it is a fallback used before giving up to `nondet`).
5. `nondet` otherwise (`jump.pl:2175-2179`).

`deterministic/5` (`jump.pl:2182-2208`) accepts: at most one clause whose
body isn't provably deterministic PLUS exactly one cut-guard clause
(`P =< 1, H == 1`), or exactly one plain deterministic clause and nothing
else (`P == 0, D == 1`) — i.e., "single matching clause, or a cut
immediately eliminates the rest." `det_body/3` (`jump.pl:1981-2019`) walks
a clause body checking every goal is itself det/tail-typed, a builtin, or
an arithmetic `is/2`, EXCEPT for one specific escape hatch:
`jump.pl:1997-2006` — when a body goal is a **direct self-call** (same
functor/arity as the head) and that callee's own type is still the `halt`
placeholder, it retroactively **reclassifies the callee to `det`** on the
spot (`retract`+`assertz`) before continuing. This is the whole-program
analysis's cycle-breaker for mutually/self-recursive determinism: assume
`halt` (a temporary "don't know yet, but plausible" tag) optimistically,
then upgrade it to `det` the moment a determinism check actually needs that
fact to go through, and (per F above) `retype_halt` sweeps any `halt` that
never got upgraded back to `nondet` at the start of pass2.

Why this matters for us: this is a concrete, working algorithm for exactly
the classification question the D55 "sound intermediate" already names
(dispatch only `deterministic`/`clause_chain` predicates, leave the rest
interpreted) — not a novel idea we lack, but a second, independently-built
implementation of the same idea, with a specific technique (an optimistic
placeholder type, resolved by first-use, swept at pass2) for the
self-recursion cycle-breaking problem that our own classifier will
eventually have to solve too if it doesn't already.

### F3 — Backtracking in compiled code: one function per file, computed-goto choice points, cut as a saved retry-pointer — the concrete answer to our resumable-protocol gap

**Verdict: adopt** (idea; the GCC-specific delivery mechanism does not
transfer, but the underlying *shape* — a resumable state machine driven by a
trampoline, with choice-point state in an explicit frame array — does).

**All of a compiled file's nondeterministic predicates are emitted into ONE
C function**, `static int user_scbm(int pred, int arity, int clause, int
arglist, int rest, int th)` (`jump.pl:497-512`), using GCC's labels-as-values
extension (`&&label`, `goto *next`). Key structural facts, each cited:

- **Inter-predicate calls between two nondet predicates are a plain `goto`,
  not a C function call.** `gen_nondet_body1` for a call to another nondet
  predicate (`jump.pl:840-860`, the "recur predicate" clause) emits
  `goto Pred_Arity;` directly (`jump.pl:857`) rather than any `call`/`return`
  sequence. Because every nondet predicate lives in the same function, this
  costs one jump, zero stack-frame setup — the closest thing to "zero
  overhead" a call between two genuinely nondeterministic predicates can be
  in C.
- **The call's own "return address" is an explicit continuation-stack push,
  not the C call stack.** Immediately before the `goto`, the compiler emits
  `Spush_next(&&SomeLabel, th); clause = Sget_choice(th);` (`jump.pl:769-771`)
  — pushing a label onto a per-thread "next" stack (`np[th]` is its depth)
  that `success:` later pops to resume the caller.
- **A choice point is: a saved label + the current `arglist` + the callee's
  numeric id**, pushed via `Spush_back(&&RetryLabel, arglist, Pred_Arid, th)`
  (`jump.pl:973-975` for the top-level case, `jump.pl:977-988` for nested
  disjunction). `allfail:` (the shared failure label, `jump.pl:635-644`)
  pops this "back" stack (`rp[th]` is its depth, `back_goto[]` the label
  table) and does `goto *next;` — **directly resuming execution at the saved
  label**, not re-invoking any function. Getting a predicate's second
  solution is therefore "restore `arglist`/`clause` from the popped frame
  and jump into the middle of the same predicate's code that's already
  sitting in the binary," not "call the predicate again from the top."
- **Per-activation local variables live in a static array indexed by
  predicate id and a `vp[]` depth counter**, not on a general call stack:
  `env_stack[Pred_Arid][vp[Pred_Arid][th]][N][th]` (`gen_pack_pointer`/
  `gen_unpack_pointer`, `jump.pl:933-949`), with `vp[Pred_Arid][th]++`/`--`
  around a nested self-call (`gen_push_varp`/`gen_pop_varp`,
  `jump.pl:990-995`) — i.e., recursion depth for a given predicate is
  bounded by that predicate's own array dimension, addressed by simple
  increment/decrement rather than a generic environment-stack push/pop.
  **(inference: the array's actual size bound was not located in this
  pass — see Open Question OQ-mprolog-1 below.)**
- **Cut is a saved retry-pointer, restored on `!`** — structurally identical
  in spirit to `WAM_BACKEND_CONVENTIONS.md` §9's `cut_barrier`/`B0`. On
  entry to a nondet predicate's dispatch block, `P_Arp = rp[th];` snapshots
  the current choice-point-stack depth (`jump.pl:578`); a `!` at the end of
  a clause body, or mid-body, emits `rp[th] = P_Arp;` (`jump.pl:799-802` and
  `jump.pl:885-887` respectively) — i.e., truncate the retry-pointer stack
  back to its value when the predicate was entered, exactly the "cut prunes
  choice points back to a barrier, never to zero" rule our own convention
  states. This is independent convergence on the same idea from a
  differently-shaped runtime, worth citing as validation of §9's design,
  not a new mechanism to adopt.
- **Calling a `det`/`tail` predicate FROM nondet code goes through the
  SAME generic, name-indexed dispatcher used for builtins**
  (`subr_number = Jmakecomp("Pred"); goto builtin_call;` →
  `Jcallsubr(subr_number, Jderef(arglist,th), NIL, th)`,
  `jump.pl:911-913` and `656-659`) — i.e., mprolog solved the nondet-to-nondet
  call path with zero-overhead `goto` (above) but did **not** extend that to
  the nondet-to-det/tail boundary; that crossing still pays a name lookup
  plus an indirect call, the same cost as calling an arbitrary builtin.
  **This is worth citing as a negative example, not a pattern to copy**: if
  we build a similar computed-goto or trampoline-based lowered tier, the
  cross-tier call (lowered nondet calling a lowered det predicate) should be
  a direct call, and mprolog's own compiler is a concrete instance of NOT
  doing that — its own tail/det tier is faster in isolation (F11) but pays
  full dynamic-dispatch cost every time nondet code calls into it.

**Why this is the standout finding**: it directly answers the exact question
`docs/WAM_RUST_STATUS.md`'s Path Forward §0 poses (a resumable protocol: "the
lowered function pushes a choice point carrying its clause index and is
re-entered on backtracking"). mprolog's answer — a saved *label* plus a
per-predicate frame-array slot, not a re-invoked function — is not directly
portable to safe Rust (no labels-as-values) or to JS (no `goto`), but it is
exactly the shape a CPS/state-machine trampoline encoding would produce:
represent each choice point as `(state_enum, saved_locals)`, drive execution
with `loop { match state { ... } }` instead of `goto *label`, and resume by
restoring `saved_locals` and jumping to the right `match` arm instead of the
right label. This is the same transform async/await and Scheme-to-C CPS
compilers use for the same underlying problem (resume mid-function without a
native call stack frame to return into). mprolog is external, working
validation that the underlying idea is sound and shippable, not merely
theoretically possible — worth citing as evidence when scoping D55's item 2
(the resumable protocol), even though the concrete encoding has to be
reinvented per target language.

### F4 — The compiler's own performance claim is unsubstantiated in the sources: raw LIPS only, no interpreter baseline, no comparison to any other system

**Verdict: out of scope** (nothing to adopt or reject as a design idea — this
is a claims-honesty finding, stated as required).

`bench/Bench.md` (the only benchmark write-up found) lists, for five
COMPILED programs on one machine (`Intel i5 2.9GHz, 8GB, WSL/Ubuntu`): derive
1,744,395 LIPS, devide10 1,688,558 LIPS, nreverse 3,068,856 LIPS, qsort
3,022,593 LIPS, queens16 6,835,936 LIPS. **No interpreted-mode run of the same
programs appears anywhere in the file**, and no other Prolog system's numbers
appear either. `grep`-ing `document/HISTORY.md` and `document/MANUAL.md` for
`LIPS`/speedup-adjacent language returned nothing. `bench/cpbench.pl` is a
compile-only driver (compiles eight `.pl` files, no timing code) — the
comparative half of "how much does compiling help" was never captured in
the repository as read.

**Conclusion, stated plainly:** there is no substantiated claim in these
sources that mprolog's experimental compiler achieves any particular
speedup over its own interpreter, or over any other system. The compiler's
*design* (F1-F3, F11) is worth mining regardless — a good architectural
idea doesn't need a benchmark to be worth reading — but nothing here
licenses citing mprolog as proof that compiling Prolog buys a specific
multiplier. This is a fine, honest negative outcome per the task's own
instruction to say so plainly when a claim doesn't hold up.

### F5 — `unify` dispatches on the raw (undereferenced) shape first, then shallow-derefs only the side that needs it — a working precedent for our Rust fix sketch

**Verdict: adopt** (idea, and a strong one — directly on bottleneck #2).

`unify(int x, int y, int th)` (`data.c:1796-1869`) opens with `nullp`/
`anonymousp` checks on the RAW arguments, then:

```
} else if (variablep(x) && !variablep(y)) {
    x1 = deref1(x, th);
    if (x1 == x) { bindsym(x, y, th); return (YES); }
    else return (unify(x1, y, th));
} else if (!variablep(x) && variablep(y)) { ... symmetric ... }
```

(`data.c:1804-1817`). `variablep/1` is a shape test on the cell's tag —
cheap, and crucially performed on `x`/`y` **before** any deref call at all.
Only once a side is known to be a variable does `deref1` (a **shallow**,
single-binding-chain-following deref, `data.c:1724-1750` — it loops while
the bound-to value is itself a variable, but does not walk into a bound
compound's substructure unless it needs to check its shape) get called, and
only on that one side. Structure-vs-structure unification then recurses
element-wise, `unify(car(x), car(y), th) == YES && unify(cdr(x), cdr(y),
th) == YES` (`data.c:1862-1864`) — each recursive call re-derives shape
information lazily, on demand, one cell at a time; there is no eager
"fully deref both entire arguments before looking at either" pass anywhere
in this function.

This is precisely the pattern `docs/WAM_RUST_STATUS.md`'s own fix sketch
proposes for `WamState::unify` (`templates/targets/rust_wam/state.rs.mustache`,
which currently opens with `deref_var(&deref_heap(v1))` /
`deref_var(&deref_heap(v2))` on BOTH arguments unconditionally, `deref_heap`
being the recursive, allocating, whole-structure-rebuilding walk profiled at
~60% malloc/free in the Θ(N²) regression, `WAM_RUST_STATUS.md:88-162`):
"Dispatch on the undereferenced shapes first and deref only the argument the
chosen arm actually needs" (`WAM_RUST_STATUS.md:157-159`). mprolog is a
real, if old, production Prolog engine independently doing exactly that —
worth citing as existence proof the reordering is correct and sufficient,
not merely a hopeful guess, when someone implements the fix.

**One caveat, in the interest of not over-claiming:** `deref1`'s own
structure-arm (`data.c:1742-1743`, `else if (structurep(res)) return
(deref(res, th));`) calls the OTHER, eager, allocating `deref` — which does
rebuild the entire bound substructure via `wcons` (`data.c:1707-1720`) — so
mprolog is not free of "deep eager copy on deref" as a concept; it just
never pays that cost during unify's *first* dispatch step, and only pays it
once, for the one side that's a bound variable, not for both arguments
unconditionally on every call the way the current Rust code does. That
narrower scoping — shallow-first, full-deref only behind a confirmed
variable binding, only for the side that needs it — is the exact shape the
Rust fix needs, and is what's actually being adopted here, not "mprolog
never deep-copies."

### F6 — Two variable representations (heap-resident "atom" variables vs. array-resident "alpha" variables) — insufficiently understood to score

**Verdict: out of scope** (flagged as inference-only; not enough was read to
respons­ibly score this against our bottlenecks — see OQ-mprolog-2 below if
a deeper pass wants to pick it up).

`bindsym`/`findvar` (`cell.c:97-122`) and `unbind` (`data.c:2157-2177`) each
branch on `alpha_variable_p(x)` vs. `atom_variable_p(x)`: alpha variables
bind into a flat, per-thread array `variant[VARIANTSIZE][THREADSIZE]`
(`mpl.h:238`) at index `x - cell_size`; atom variables bind in place on the
heap cell itself via `SET_CAR(x, val)`. The cell struct (`mpl.h:109-134`) is
a heap-index-addressed (not pointer-addressed) struct with an explicit
`tag` field, `car`/`cdr` as `int` heap indices, and a `name` pointer
carried directly on the cell (i.e., atoms/structures carry their name
inline rather than through a separate symbol-table indirection at
unification time — `mpl.h:109-134`, `GET_NAME`-style macros not
individually cited here). This split looks, on its face, like "compiled
code's many short-lived per-activation locals get a flat array (`alpha`)
while interpreter-resident, potentially-long-lived variables get a heap
cell (`atom`)" — but that reading is an inference, not something the two
files read here state directly, and the actual rule for which kind a given
variable becomes was not traced. **Scoring this against our bottlenecks
(a possible dual-representation idea for our own `Value` types) would be
irresponsible without reading `makevariant`/the allocation-site logic that
decides alpha vs. atom, which was not part of this pass's budget.**

### F7 — The interpreter has zero clause indexing of any kind — corroborates D54, not a new finding to act on

**Verdict: already-covered** (cite D54's own conclusion in
`docs/proposals/JS_TARGETS_PARITY_PUNCHLIST.md`; nothing new to adopt).

`prove()`'s clause-resolution loop (`main.c:775-802`) is `while
(!nullp(clauses))` over the FULL clause list for the called predicate/arity
— `car(clauses)`/`cdr(clauses)` linear walk, unify-and-try each in turn, no
first-argument type check, hash, or any other discriminator anywhere in the
loop. A repo-wide search for indexing-adjacent identifiers
(`first.?arg|index|switch_on|hash_first`, case-insensitive, across every
`.c` file) turned up only atom-interning hash tables (`cell.c:127-177`) and
the `assert`/`recorda` record database's hash buckets (`builtin.c:5787-6268`)
— neither related to ordinary clause selection. The compiled path's
`switch(clause)` (`jump.pl:326-339`) is a **resumption** dispatch (which
retry-index to jump back into) driven by the choice-point stack, not a
first-argument discrimination index; it does not reduce the set of clauses
tried, only where execution resumes among clauses already committed to a
linear order.

**Why this doesn't need a new predicate or mechanism from us:** a real,
shipped, non-toy Prolog implementation went years with literally no clause
indexing and answered nondeterminism cost with the compiler (F1-F3, F11)
instead. This is independent, external corroboration — not proof, but a
second data point — for D54's own conclusion that per-instruction/lowered-
tier cost dominates over indexing sophistication on the workloads that
matter to us. Cite this finding alongside D54 in future prioritization
discussions rather than re-deriving it.

### F8 — Term model is structure-copying with a flat-array mark-sweep GC over a fixed-size heap — no structural sharing; useful only as a negative example

**Verdict: out of scope** (the negative example corroborates our own D52
fix; nothing to adopt from the GC itself).

`gbc()` (`gbc.c:10-30`) is classic non-generational, non-incremental
mark-and-sweep: `gbcmark()` then `gbcsweep()` then a linear rescan of the
whole (fixed-size, `HEAPSIZE`) heap array to recompute the free count.
`markcell` (`gbc.c:33-70`) recursively marks `car`/`cdr`/`GET_ARITY`/
`GET_VAR`/`GET_RECORD` per structured cell — ordinary term-graph marking,
nothing novel. Combined with `deref()`'s eager `wcons`-based reconstruction
of an entire bound compound term on every dereference that crosses a
structure boundary (`data.c:1707-1720`, discussed in F5's caveat) and
`freshcell()`'s single globally-mutex-protected bump/free-list pointer
(`cell.c:67-78`, the same `mutex1` discussed in F9), the overall picture is
a **structure-copying** term representation, not structure-sharing: bound
compound terms get rebuilt into fresh cells rather than referencing a
shared spine. This is the opposite of the fix UnifyWeaver's Rust lane
already landed (D52: `Value::Str`/`Value::List` carrying a shared `Arc`
spine, turning `Value::clone` from O(term size) to O(1)). Nothing here is
worth adopting — it's a plain, older design — but it is useful as
corroboration: our own move to structural sharing is not a redundant or
speculative idea revisited for no reason, it is a genuine improvement over
what at least one comparable engine actually shipped with.

### F9 — Parallel model: a reasonable explicit fork/join API, but its only lock protects an in-place mutation of a *shared* clause template — a scalability trap for T7 to avoid

**Verdict: adapt** (adopt the shape of the user-facing API; explicitly avoid
the locking pattern underneath it).

Two parallel extensions are documented: distributed (`document/PARA1.md`,
TCP/IP, one child-Prolog process per machine, up to 100 children, explicit
`dp_create`/`dp_transfer`/`dp_compile`/`dp_consult`/`dp_prove`/`dp_close`
lifecycle) and multi-thread (`document/PARA2.md`, `mt_create(N)` fixed
worker pool sized `core_count - 1`, a signal-based enqueue/dequeue of
worker-id integers). Both expose the same two combinators: **`dp_and`/
`mt_and`** — run N goals in parallel, succeed only if all succeed — and
**`dp_or`/`mt_or`** — run N goals in parallel, succeed (and cancel the
others) on the first success. This is explicit, programmer-annotated
fork/join with no automatic or-parallel/and-parallel search and no implicit
granularity control — a small, honestly-scoped API, not a general parallel
resolution engine.

**The scalability trap:** `prove()`'s clause-instantiation step —
`assign_variant`/`walpha_conversion`/`release_variant`
(`main.c:775-786`) — is wrapped in `pthread_mutex_lock(&mutex1)` /
`pthread_mutex_unlock(&mutex1)` (`main.c:781, 785`), the **same** global
mutex `freshcell()` uses to protect the shared heap free-list pointer
(`cell.c:71-76`). The reason `assign_variant`/`release_variant`
(`data.c:2182-2198`) need a lock at all, despite each thread having its own
`variant[][th]`/`localstack[][th]` binding arrays, is that they **temporarily
mutate the shared clause template itself**: `SET_CDR(car(x),
makevariant(th))` (`data.c:2185`) writes a per-call variant number directly
onto the read-only clause's variable-occurrence cells (a structure-sharing
renaming trick — attach the fresh variable inline, copy, then
`release_variant` unwrites it, `data.c:2192-2198`, `SET_CDR(car(x), NIL)`).
Because that scratch write lands on a structure every thread can see (the
one shared clause database), **every thread's every clause attempt, for
any predicate, must serialize on one global lock**, regardless of which
predicate or which thread is involved — a single choke point that would
erase most of the benefit of running clause search on multiple cores.

**What to adopt vs. avoid, explicitly:**
- **Adopt the shape:** `dp_or`'s "first success cancels siblings" and
  `dp_and`'s "all must succeed" as named, minimal, explicit parallel-goal
  primitives are a reasonable, easy-to-reason-about API if T7's
  parallel-aggregate work doesn't already have an equivalent explicit
  fork/join primitive at the Prolog-source level.
- **Avoid the locking pattern:** never rename/instantiate a clause by
  mutating a shared/interned representation in place, even transiently,
  even if it looks cheap because it avoids an allocation — it forces a
  global lock the moment more than one thread touches clause search
  concurrently. Any UnifyWeaver clause-renaming or clause-copy step that
  feeds T7's parallel path should be checked for this specific shape (see
  Open Question OQ-mprolog-4).

### F10 — CLP(FD) is deliberately light and explicitly documented as *not* benefiting from compilation — a caution against over-generalizing the lowered-tier win

**Verdict: out of scope** (we have no CLP(FD) today; recorded as a caution
for if/when we ever do).

`document/CLPFD.md:1-8` states the design intent directly: "N-Prolog's
CLPFD deliberately does not include heuristic optimizations... valuable and
preferable for educational purposes," and, most relevantly, **"Since most
execution cost lies in constraint propagation and labeling, native
compilation does not currently provide significant performance benefits,
and some cases remain interpreter-only by design."** The data structures
backing it are small, fixed-size C arrays (`fd_domain[256]`,
`fd_min/max/len[256]`, `fd_removed[256][256]`, per `CLPFD.md`'s own memo
section) rather than general Prolog terms, and `clp.c` is a modest
1,292 lines.

Why this is worth one line rather than zero: it is a documented case, from
people who actually tried, where the "lowering removes interpretation
overhead" lever we are currently betting on (our #1 fleet-wide cost) turned
out NOT to be the dominant cost for a different workload shape (constraint
propagation/labeling). This is a caution, not a contradiction of our
current work — uw-resolve-shaped code is not constraint-solving-shaped code
— but it is a reason to measure rather than assume if UnifyWeaver ever
scopes CLP(FD)-style work: don't assume the lowered-tier investment
automatically pays off there without checking.

### F11 — Tail-recursive predicates compile straight to a native C loop with zero choice-point machinery — a lower-risk, ship-first special case of the lowered tier

**Verdict: adopt** (idea; directly actionable ahead of the resumable
protocol, and ties to the same bottleneck as F3).

Predicates classified `tail` by `tail_recursive/6` (F2; `jump.pl:2210-2258`,
which itself depends on `independ_head/1`, `jump.pl:2261-2270` — no head
argument variable may repeat across the clause head, a decidable, purely
syntactic check) compile via `gen_tail_pred`/`gen_tail_clause`/
`gen_tail_body` (`jump.pl:1684-1760`) to: assign arguments once, emit a
`loopN:` label, test base cases in order (each `return`s directly), and for
the recursive case, reassign the argument registers via `gen_tail_args`
(`jump.pl:1754-1759`, direct C variable writes, each wrapped in
`Jcopy_work(Jderef(...))` to materialize the next iteration's arguments)
then `goto loopN;` (`jump.pl:1745-1752`). **There is no choice point, no
environment-stack frame, no C recursion, and no `Spush_back`/`Spush_next`
anywhere in this code path** — the entire predicate becomes one flat,
loop-shaped C function, because `tail_recursive/6` only accepts a clause
set that is provably single-path (independent-head base cases plus exactly
one genuinely-tail-recursive clause, `jump.pl:2211-2245`).

Why this is actionable now, ahead of F3's harder problem: a
tail-recursive, single-solution-path predicate has, by construction, no
choice point ever to resume — so lowering it is **safe under our EXISTING
first-solution-only lowered tier**, with none of D55's item-2 risk ("wiring
dispatch without fixing [the resumable-protocol gap] would silently drop
solutions," `WAM_RUST_STATUS.md:419-421`). This maps closely onto the
`deterministic`/`clause_chain` classes our own Path Forward §0 already
names as the sound intermediate to land first. mprolog's
`tail_recursive/6` + `independ_head/1` (`jump.pl:2210-2270`) is a ready-made,
mechanical checklist for exactly that class of predicate — worth reading in
full by whoever implements or extends our own front-end classifier, as a
second working implementation of the same decidable property, rather than
reinventing the independent-head-variable check from scratch. It is
independent of F3/D55's item 2 and can land without it.

## Summary: verdict counts

- **Adopt** (F1, F2, F3, F5, F11) — **5 findings**. Three of these (F2, F3,
  F5) map directly onto named, open items in `WAM_RUST_STATUS.md`'s Path
  Forward and the D54/D55/D59 bottleneck ledger; F11 is a lower-risk,
  ship-first sibling to F3; F1 is the umbrella architectural precedent for
  all four.
- **Adapt** (F9) — **1 finding**: take the explicit fork/join API shape,
  discard the shared-mutable-clause-template locking pattern underneath it.
- **Already-covered** (F7) — **1 finding**: corroborates D54's own
  conclusion about indexing; no new predicate or mechanism needed.
- **Out of scope** (F4, F6, F8, F10) — **4 findings**: F4 is an honesty
  finding about an unsubstantiated claim (nothing to adopt or reject as
  design); F6 is flagged as under-read rather than scored; F8 is a
  negative example that corroborates a fix we already made; F10 is a
  caution for a workload class (CLP(FD)) we don't have yet.

## Open questions for the coordinator

Each is phrased so a deeper-analysis agent can pick it up directly, tagged
with the bottleneck it serves. None of these were answered here because
they need either a prototype/measurement, a read of code outside this
pass's budget, or a judgment call about prioritization that recon
shouldn't make unilaterally.

1. **[Bottleneck: lowered tier / resumable protocol, D55 Path Forward §0]**
   mprolog's compiled-nondet resumption (F3) works via GCC labels-as-values,
   which Rust/JS/Go don't have. Is a trampoline/enum-state-machine encoding
   of "choice point → resume at saved label" (`loop { match state { ... } }`
   with saved locals in a per-choice-point struct, instead of `goto
   *label`) cheap enough in Rust — given the existing profile is already
   65% `step`-dispatch / 21% `backtrack` and malloc/free-dominated
   (`WAM_RUST_STATUS.md`'s post-sharing callgrind numbers) — to be worth
   prototyping ahead of, or alongside, the three-item scoping already
   recorded there (`lowered_call` call-site hook, the resumable protocol
   itself, ITE-per-clause + `(A ; B)` support)? This needs a small
   prototype and a measurement, not more reading of mprolog or of our own
   docs.

2. **[Bottleneck: per-instruction interpretation cost / "fuse straight-line
   runs," D54's CLJS lever]** mprolog's `tail_recursive/6` +
   `independ_head/1` (F11, `jump.pl:2210-2270`) is a mechanical, working
   whole-predicate classifier that identifies predicates safe to compile
   straight to a native loop with zero choice-point machinery, needing none
   of the resumable-protocol work in question 1. Does UnifyWeaver's
   front-end (`wam_target.pl` / the lowered emitters referenced in
   `WAM_RUST_STATUS.md`) already have an equivalent "is this predicate
   safely loop-compilable" classifier, or would porting the *idea* behind
   mprolog's `tail_recursive/6` (not its code) be net-new front-end work?
   If net-new, should it be sequenced before or after question 1's
   resumable-protocol prototype, given it needs none of that machinery and
   could land independently?

3. **[Bottleneck: `WamState::unify` quadratic, D59 / `WAM_RUST_STATUS.md`
   "Quadratic list handling"]** F5 shows mprolog's `unify` checks
   `variablep(x)` on the raw argument before any deref, then shallow-derefs
   only the variable side — exactly the shape `WAM_RUST_STATUS.md`'s own
   fix sketch proposes. Is reordering `WamState::unify`'s opening lines to
   match this shape (shape-check first, single-level deref only where the
   chosen arm needs it) a mechanical, drop-in rewrite given Rust's actual
   `Value`/ownership model post-D52 structural sharing, or does the
   borrow-checker (unlike mprolog's raw `int`-indexed heap, which has no
   ownership rules to satisfy) force a materially different structuring of
   the match arms? This is an implementation question that needs someone
   working directly in `templates/targets/rust_wam/state.rs.mustache`, not
   further recon.

4. **[Bottleneck: T7 parallel-aggregate machinery]** F9 found mprolog's
   only multi-thread lock exists because clause-variable renaming
   temporarily mutates the *shared* clause template in place rather than
   allocating independently. Does any UnifyWeaver clause-instantiation or
   clause-renaming step that T7's parallel-aggregate path would touch (in
   any target's runtime, not just Rust) ever mutate a shared/interned
   clause representation in place, even transiently, during instantiation?
   If so, that is a latent single-lock bottleneck waiting to surface the
   moment T7 adds real thread-level parallelism, worth auditing now rather
   than after T7 lands. This needs a code audit across the relevant
   runtimes, not another read of mprolog.

5. **[Bottleneck: overall prioritization / cross-cutting judgment call]**
   F1/F3/F11 together describe a real, working precedent for
   "native-compile whole predicates, including genuinely nondeterministic
   ones, with a real resumable-solution protocol" — but delivered via a
   mechanism (GCC labels-as-values plus `dlopen` at compile time) with no
   direct analogue in Rust, JS, Go, or ClojureScript. Should this precedent
   move the RELATIVE priority of D55's resumable-protocol work (item 1
   above) upward, on the theory that a proven design lowers execution
   risk — or is the honest read that the precedent is C-specific enough
   that it shouldn't reweight anything, and the only concrete,
   directly-portable takeaways from this whole mining pass are F5 (the
   unify reordering) and F11 (the tail-loop fast path)? This is a
   portfolio-level call about how much weight a C-lineage precedent should
   carry against our own measured multi-target constraints — above what a
   recon pass should decide unilaterally.
