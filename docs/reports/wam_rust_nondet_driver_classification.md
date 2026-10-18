<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Nondet driver classification — the three uw-resolve backtracking drivers

Status: **Design proposal** (2026-09-07). Scope: classify the three genuine
backtracking drivers in the frozen `examples/pkg_resolver/resolver.pl`
(`pick/7`, `blocked_from/4`, `dep_breaks/5`) and, for each that genuinely
exposes alternatives, propose a **semantics-preserving** rewrite to a
deterministic-by-construction shape that **every** target's transpiler can
recognize and lower the cheap way. **`resolver.pl` / `resolver_store.pl` are NOT
modified by this round** — a resolver rewrite is a separate, deliberately
authorized resolver round with full before/after semantics verification and
human approval. This document is a proposal that goes to the human first.

This supersedes the earlier plan to build a per-target resumable trampoline for
these drivers (`WAM_RUST_LOWERED_TIER_THROUGHPUT_PLAN.md` §5/§9 Stage 2 "2b").
The higher-leverage, lower-risk path is to make the drivers deterministic in the
**spec**, so the cheap deterministic lowering (native loop / explicit stack + a
minimal 3-scalar snapshot, no resume-state choice point — exactly regions 1–4)
applies on Rust, C++, Go, wamjs and every other target at once, and the hard
soundness gates (G-4 multiplicity/order, G-5 activation identity /
cut-invalidation) are **vacuous** because no choice point is ever left.

## The three deterministic-lowering classes

A recursion (or driver) whose body performs a nondeterministic sub-search still
yields **exactly one solution and leaves no choice point** — so it lowers the
cheap deterministic way — when the nondeterminism is *contained* before it can
cross the iteration/solution boundary. Three containment shapes:

- **(a) committed-per-iteration recursion.** The per-step nondet goal is
  committed (`once/1`, a cut, or the condition of an `->`) **before** the tail
  self-call, so backtracking can never re-enter an earlier iteration's choice.
  One solution overall. (PLAWK's per-record match-and-print loop is this shape;
  `resolver.pl`'s `build_tree/4` — Stage-2 region 4 — is the arithmetic-test
  special case already lowered deterministically.)
- **(b) aggregate-then-iterate.** The nondet search is bounded inside an
  aggregation (`findall`/`bagof`/`setof`/`aggregate_all`), which is itself
  deterministic (it explores the goal fully and returns **one** list); the
  useful work is a deterministic fold over that list. Backtracking is contained
  at the aggregate boundary. (PLAWK's codegen uses `findall(X, member(X,L), Xs)`
  then folds; `resolver.pl`'s `virtual_provider_ceilings/4` is internally this
  shape.)
- **(c) genuinely alternative-exposing.** The predicate hands multiple solutions
  **upward** to its caller across the recursion — the caller relies on the
  order/multiplicity (via backtracking search or `findall`). Only this class
  needs a resume-state choice point, and only this class cannot be lowered the
  cheap way without either a semantics-preserving rewrite to (a)/(b) or accepting
  the interpreter.

## Summary verdict

| driver | class (as written) | reachable in corpus? | deterministic rewrite? |
|---|---|---|---|
| **`dep_breaks/5`** | **(a)** committed-per-iteration | yes | **none needed** — already deterministic; recognizer needs one generalization |
| **`pick/7`** | **(c)** essential search nondeterminism | **no (dead code)** | **not possible locally** — nondeterminism is the resolver's version-backtracking search; also unreachable |
| **`blocked_from/4`** | **(c)** generator, multiplicity consumed by `findall` | yes | **possible only as a subsystem-level (b) refactor** with real order/multiplicity risk — proposed below, flagged high-risk |

Net: of the three "nondet drivers", **one (`dep_breaks/5`) is already
deterministic-by-construction**; **one (`pick/7`) is dead code whose
nondeterminism is the core search and cannot be localized away**; and **one
(`blocked_from/4`) is a genuine generator whose full solution sequence is
observed by the resolver's own test suite via `findall`**, so it can be made
deterministic only by refactoring the whole *explain* subsystem to an
all-solutions list form (which mostly already exists as `explain_blocked_list`),
not by a local clause edit.

---

## 1. `dep_breaks/5` — class (a), already deterministic

### Evidence (resolver.pl:1008–1014)

```prolog
dep_breaks([depends(HN, HV, D, C)|Rest], N, V, Acc, COut) :-
    (   HN == N,
        HV == V,
        dep_breaks_need(Acc, D, C, CBroken)      % nondet (member/2 inside)
    ->  COut = CBroken                           % commit — no recursion here
    ;   dep_breaks(Rest, N, V, Acc, COut)        % tail self-call in the else only
    ).
```

`dep_breaks_need/4` (resolver.pl:1016) is nondeterministic — its clause 1 is
`member(dep(D,COut),Alts), selected_ver(...), \+ satisfies(...), \+ (...)`. But
it sits **inside the `->` condition**, so the `->` commits its first solution and
discards the alternatives *before* the tail recursion in the else-branch. The
single clause has one recursive call, reached only when the condition fails.
Therefore:

- No alternative of `dep_breaks_need` survives past the iteration boundary.
- No second solution of `dep_breaks/5` exists.

It is additionally **consumed first-solution**: its only caller,
`dep_breaks_moving/5` (resolver.pl:1004) is entered from `first_broken/4`
(resolver.pl:999) as `dep_breaks_moving(Cat,N,V,Acc,C) -> ...`. Both the commit
point and the consumer confirm class (a).

### Rewrite: none needed

`dep_breaks/5` is already deterministic-by-construction. It should lower to a
native walk over the `depends` list with the per-element condition committed —
the same deterministic tier as regions 1–4.

### Recognizable marker

*Single clause (or a base clause + one recursive clause) whose body is an `->`
whose **condition** contains the per-iteration goals (including any nondet
sub-goal) and whose **else-branch is the sole tail self-call**.* Structurally:
`Body = ( Cond -> Then ; Self(...) )` with `Self` the recursive call and `Cond`
containing no recursive call.

### What each target's recognizer needs

The Rust `rust_region4_build_tree_ok` recognizer (wam_rust_target.pl:10073)
already matches a single-clause `( Test -> ... ; ... )` shape via `=@=`, but it
special-cases build_tree's *arithmetic* test (`N =:= 0`) as the committing
condition and requires the recursion inside the `->`-else's *then*. **The
generalization needed** (all targets): accept an arbitrary committing condition —
including one that calls a nondeterministic predicate — provided (i) the only
recursive self-call is the else-branch tail, and (ii) the condition is committed
by `->`/`once`/cut. The det-lattice must treat `Cond` as "at-most-one-solution
after the commit" rather than requiring `Cond` itself to be det. This is a
recognizer widening, not new runtime machinery: the emitted region is still a
plain native loop with a minimal snapshot and no resume-state CP.

---

## 2. `pick/7` — class (c), essential nondeterminism, and dead code

### Evidence (resolver.pl:622–635)

```prolog
pick(classic, Cat, Name, C, _Acc, Ver, from_catalog) :-
    candidates_high_first(Cat, Name, C, Ver).            % bare, uncommitted
pick(layered, Cat, Name, C, _Acc, Ver, Origin) :-
    (   base_ver(Cat, Name, BV)
    ->  satisfies(BV, C), Ver = BV, Origin = from_base
    ;   candidates_high_first(Cat, Name, C, Ver),        % bare, uncommitted
        Origin = from_catalog ).
```

`candidates_high_first/4` (resolver.pl:248) is itself aggregate-then-**expose**:

```prolog
candidates_high_first(Cat, Name, C, Ver) :-
    candidate_versions(Cat, Name, C, Desc),   % deterministic: sorted version list
    member(Ver, Desc).                         % exposes each version upward
```

`candidate_versions/4` builds the full descending-version list deterministically
(ITE + `sort_versions_desc`); the `member/2` then **hands each version to the
caller one at a time, highest first**. `pick(classic,...)`'s body is this bare
enumeration with **no commit**, so `pick/7` genuinely yields multiple `Ver`
solutions upward.

This is not incidental nondeterminism — it **is** the resolver's search: the
main resolution loop tries the highest version, and on a downstream conflict
*backtracks into `pick`* to try the next-highest. The order (descending) and the
ability to backtrack are load-bearing for which resolution is found.

### Reachability: dead code

`pick/7` has **no callers anywhere** in `examples/pkg_resolver/` (only its own
two clause heads and the unrelated `pick_need/8` / `pick_repair/4` match a
`grep`). The header comment (resolver.pl:621) says it is "kept for the
real-package-only path"; the live path uses `pick_need/8`. So `pick/7` never
enters the term/store differential corpus and cannot be measured.

### Rewrite: not possible as a local edit

The nondeterminism is the version-choice backtracking the resolver's whole search
depends on; collapsing it to a fold would require re-expressing the entire
resolution search as an explicit backtracking fold (a full architectural
rewrite, not a driver-local change). It is also moot because `pick/7` is dead.

### Recommendation

Leave `pick/7` interpreted (it is dead and unmeasured). If a future round wants
to lower the *live* search (`pick_need/8`), that is a genuine class-(c) target
for the resume-state trampoline, to be justified by its own measurement — out of
scope here.

---

## 3. `blocked_from/4` — class (c) generator; multiplicity consumed by `findall`

### Evidence (resolver.pl:713–733)

```prolog
blocked_from(Cat, req(alternatives(Alts), _), Seen, Blocked) :-  % clause 1
    !, alt_reasons(Cat, Alts, Seen, Rs), Blocked = blocked(alternatives(Rs)).
blocked_from(Cat, req(Name, C), _Seen, Blocked) :-               % clause 2
    base_ver(Cat, Name, BV), \+ satisfies(BV, C),
    Blocked = blocked(Name, needs(C), base_has(BV)).
blocked_from(Cat, req(Name, C), _Seen, Blocked) :-               % clause 3
    virtual_provider_ceilings(Cat, Name, C, Reasons), Reasons \== [],
    Blocked = blocked(Name, needs(C), providers(Reasons)).
blocked_from(Cat, req(Name, C), Seen, Blocked) :-               % clause 4 (recursive)
    \+ seen_name(Seen, Name),
    walk_pkg_for_blocked(Cat, Name, C, Pkg, Ver),               % nondet (3 clauses)
    collect_deps(Cat, Pkg, Ver, DepReqs),                       % det (returns a list)
    member(Dep, DepReqs),                                       % exposes each dep upward
    blocked_from(Cat, Dep, [Name|Seen], Blocked).              % uncommitted recursion
```

- Clauses **2, 3, 4 share the head `req(Name, C)`** and carry **no cut** — a
  genuine uncommitted clause chain (`TryMeElse`/`RetryMeElse`). They are **not**
  mutually exclusive: a name can have a non-satisfying base version (clause 2),
  *and* virtual-provider ceilings (clause 3), *and* blocking dependencies
  (clause 4), each a distinct solution.
- Clause 4 compounds nondeterminism: `walk_pkg_for_blocked/5` has three
  provider-enumerating clauses (resolver.pl:758), `member(Dep, DepReqs)`
  enumerates the dependency list, and the recursive `blocked_from` is the
  uncommitted tail. Alternatives survive across the recursion.
- The deterministic sub-parts are already contained: `collect_deps/4`
  (resolver.pl:598) returns one list; `virtual_provider_ceilings/4`
  (resolver.pl:767) is internally `findall(...)` — class (b) already.

So the multiplicity is the **clause chain × walk providers × deps × recursion**,
handed upward.

### The multiplicity is observed — this is the decisive constraint

`blocked_from/4` is reached in the term/store differential (via
`explain_blocked_list → blocked_acc → alt_reasons → explain_alt`, where
`explain_alt` at resolver.pl:788 does `blocked_from(...) -> true`, first
solution). If that were the *only* consumer, a first-solution commit would be
observationally safe. **It is not.** The resolver's own test suite consumes the
full sequence:

```
examples/pkg_resolver/test_pruning_probes.pl:418  findall(B, explain_blocked(Cat, a,   B), Bs)
examples/pkg_resolver/test_pruning_probes.pl:449  findall(B, explain_blocked(Cat, app, B), Bs)
examples/pkg_resolver/test_resolver.pl:536,544,702,710  findall(B, explain_blocked(Cat, app, B), Bs0)
```

`explain_blocked/3` (resolver.pl:703) calls `blocked_from/4` directly, and these
tests `findall` over it — so **the order and multiplicity of `blocked_from`'s
solutions are part of the resolver's semantic contract**. Committing to the first
solution (turning it class (a)) would change those results. `blocked_from/4` is
therefore **genuinely class (c)**.

### Proposed rewrite: subsystem-level aggregate-then-iterate (class b) — HIGH RISK

The only semantics-preserving way to make the *explain* work deterministic while
preserving multiplicity is to compute **all** blocked explanations
deterministically, in the exact order the generator backtracks, and hand callers
the list. That deterministic all-solutions form **largely already exists** as
`blocked_acc/5` + `explain_blocked_list/3` (resolver.pl:735, 707), which walks
with an accumulator and `sort`s. The rewrite is:

1. Add a deterministic all-solutions predicate that reproduces
   `findall(B, blocked_from(Cat, Req, Seen, B), Bs)` **in generator order** by an
   explicit depth-first accumulation (the explicit-stack transform used by
   regions 3b/4), threading `Seen` exactly as clause 4 does:

   ```prolog
   % PROPOSED (design only — NOT applied to resolver.pl)
   blocked_all(Cat, Req, Seen, Bs) :-
       findall(B, blocked_step(Cat, Req, Seen, B), Bs).   % aggregate boundary
   %   ^ or an explicit-stack fold that yields the identical list in the identical order
   ```

2. Update the `findall(B, explain_blocked(...), Bs)` call sites (the four test
   sites above, plus `explain_alt`'s `-> true`, which becomes "take the head of
   the list") to consume the list form.

**Why it *can* preserve results:** `findall/3` already fixes the enumeration
order to the generator's left-to-right search order, so wrapping the existing
generator in `findall` is trivially identical. The value is only realized if the
generator *inside* the aggregate is itself lowered deterministically (an explicit
DFS collecting into the list) — otherwise the aggregate is just interpreted
`findall` over an interpreted generator (no win). Reproducing the generator's
exact DFS order (clause 2 before 3 before 4; within clause 4, `walk` providers in
clause order, `member` in list order, recursion depth-first with the growing
`Seen`) in an explicit-stack fold is where the risk lives.

**Risk (high):** any reordering of the explicit DFS relative to the interpreter's
clause/subgoal order changes the list order → the `findall` tests diverge, and
`explain_alt`'s first-solution (`-> true`) could change *which* reason is
reported. The seven-scenario soundness bar (order/multiplicity must match the
interpreter exactly, no reordering for optimization) applies in full. This is a
genuine algorithm reimplementation of a recursive, `Seen`-threaded, three-source
search — **not** a mechanical clause edit — and it changes the predicate's
interface (list-returning) and its callers. It should not be undertaken as a
by-product; it is its own resolver round with the full differential + the
`findall`-based unit tests as the oracle.

**Recommendation for `blocked_from/4`:** **do not rewrite in the near term.** The
honest classification is class (c) with observed multiplicity; the deterministic
form already exists for the aggregated consumer (`explain_blocked_list`), and the
generator form is retained precisely so callers can `findall` it. Leaving the
generator interpreted is correct and cheap relative to its small corpus share
(the `pick`/`blocked_from`/`dep_breaks` group is ~4.9% of B2 per the census, most
of which is not `blocked_from`). If the *explain* subsystem is ever unified on the
list form, that is the moment to lower it as a deterministic all-solutions region
— tracked as future work, gated by its own before/after semantics proof.

### Recognizable marker (for the class-(b) form, if pursued)

At the **call site**: `findall(X, Gen(...), L)` (or `bagof`/`setof`) followed by
a deterministic fold over `L` (`foldl/4`, or a member-driven recursion whose
per-element choice is committed). The transpiler keys on the `findall`/`bagof`/
`setof` wrapper as the aggregate boundary and on the following fold as the
deterministic consumer — the same shape PLAWK's codegen already detects
(`findall(X, member(X,L), Xs)` idioms in
`examples/plawk/codegen/plawk_native_codegen.pl`).

### What each target's recognizer needs

No target currently lowers a `findall`-bounded generator as a native
all-solutions region; `findall/3` stays an interpreted builtin (correctly).
Realizing the win requires two new recognizer/emitter cases, **target-agnostic
because the markers live in the spec**: (i) recognize the `findall(_, Gen, L)`
+ fold pair, and (ii) lower `Gen`'s DFS as an explicit-stack accumulation into
`L` (regions 3b/4 already demonstrate the explicit-stack transform per target).
Until both exist, the marker documents intent but the aggregate is interpreted.

---

## 4. Why this is the right call over the trampoline

- **Correctness:** classes (a)/(b) leave **no** choice point, so the two hardest
  soundness gates — G-4 (at-most-one-solution / order / multiplicity) and G-5
  (activation identity + cut-invalidation of the predicate's own resume-state
  CP) — are **vacuous**, not merely satisfied. The resume-state trampoline would
  have to satisfy them for a *recursive* driver (`blocked_from`) with a cut
  (clause 1) and observed multiplicity — the maximum-risk case.
- **Leverage:** a spec-level marker lowers on every target (Rust, C++, Go,
  wamjs, …) through each one's existing deterministic tier, instead of a
  Rust-only resume-state runtime.
- **Honest scope of the win:** exactly **one** of the three drivers
  (`dep_breaks/5`) is already deterministic and merely needs a recognizer
  widening; `pick/7` is dead; `blocked_from/4` genuinely exposes alternatives
  that are consumed and should stay interpreted unless the explain subsystem is
  deliberately refactored. There is no free deterministic rewrite for the two
  class-(c) drivers.

## 5. Recommendations (for human approval before any resolver round)

1. **`dep_breaks/5`:** widen the deterministic single-clause `->`-commit
   recognizer (region-4 family) to accept a committing condition that contains a
   nondet sub-goal, with the recursion confined to the else-branch tail. No
   `resolver.pl` change; recognizer-only, per target. Lowest risk, clear win on
   its dispatch share, and it generalizes region 4.
2. **`pick/7`:** leave interpreted (dead code; essential search nondeterminism).
   Revisit `pick_need/8` (the live search) separately if a nondet tier is ever
   justified by measurement.
3. **`blocked_from/4`:** keep the generator interpreted now. Record the
   subsystem-level class-(b) refactor (unify explain on a deterministic
   all-solutions list computed in generator order) as future work, to be run as
   its own resolver round with the `findall`-based unit tests + the 2600/503
   differential as the semantics oracle. Do **not** commit to first solution.
4. No target builds a resume-state trampoline for these drivers in this round.

## Attribution

The committed-choice loop shape and the resume-state choice-point idea are
concepts from Kenichi Sasagawa's M-Prolog / N-Prolog
(`https://github.com/sasagawa888/mprolog`, Modified BSD), never code, per
`MPROLOG_MINING_NOTES.md`. The aggregate-then-iterate and committed-choice
transforms are standard deterministic-lowering patterns (also used by this
project's PLAWK native codegen) and are not mprolog-specific.
