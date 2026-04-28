# WAM Mode Analysis: Philosophy

> **Scope:** This doc covers the design-level *why* of the binding-state
> analysis pass that will unlock the deferred `PutStructureDyn`
> lowering for `=../2`. The companion docs are
> `WAM_HASKELL_MODE_ANALYSIS_SPEC.md` (the contract) and
> `WAM_HASKELL_MODE_ANALYSIS_PLAN.md` (the work breakdown).

## What we have today

The Haskell WAM target supports `T =.. [Name | Args]` (Prolog univ) via
a single `BuiltinCall "=../2" 2` instruction whose runtime handler
covers both modes:

- **Compose** (T unbound, Name bound to atom): build a `Str fid args`
  term from the list and bind T.
- **Decompose** (T bound to a structure): walk the term, build the
  list `[Atom name, arg1, ..., argN]`, bind it to A2.

This works correctly as of PR #1657 (the `SetVariable` constructor
fix). It is also slow on the compose path: the compiler emits a
`put_list`/`set_value`/`set_constant` cascade to *materialise* the list
on the heap, then the runtime walks the list to extract the functor
name and argument vector before constructing the `Str` term. For a
predicate that constructs terms in a tight loop, that allocation +
walk dominates.

A faster instruction already exists: `PutStructureDyn nameReg arityReg
targetReg`. It is the runtime-parsed variant of `PutStructure` —
it reads the functor name (an atom) and the arity (an integer) from
registers and pre-allocates a `BuildStruct` builder that subsequent
`SetValue`/`SetConstant` instructions can populate, exactly like
static `PutStructure`. No intermediate list. No runtime list walk.

The instruction has been in the WAM since the original
`PutStructureDyn` work. Nothing emits it yet.

## Why we have not lowered to `PutStructureDyn` yet

`PutStructureDyn` is unsafe in decompose mode. The runtime expects
`Name` to dereference to an `Atom fid`; if it dereferences to an
`Unbound`, the step returns `Nothing` and the goal fails. The same
Prolog source

```prolog
T =.. [Name, X, Y].
```

is used both ways — it composes when called with `T` fresh and `Name`
bound, decomposes when `T` is bound and `Name` is fresh. The compiler
cannot tell from local syntax alone which mode applies at any given
call site. Eagerly emitting `PutStructureDyn` for every `=../2`
would silently break decompose-mode callers.

The fix is to give the compiler enough static knowledge to *prove*
compose mode at the program point of the `=..` goal. That is what a
binding-state analysis pass provides.

## What we are willing to spend

We are explicitly **not** building Mercury-style mode analysis.

- We do not need every-mode-of-every-predicate inference.
- We do not need uniqueness or determinism analysis.
- We do not need the analysis to be sound across module boundaries
  in the absence of declarations.

We need a small, conservative pass that answers a single question for
each variable at each program point in a clause body:

> *Is this variable definitely unbound here? Is it definitely bound to
> a non-variable? Otherwise: don't know.*

Three-valued, monotonic, propagated forwards. No fixpoint. No
backwards inference. No abstract domain richer than
`{ unbound, bound, unknown }`.

That suffices for the `=../2` lowering decision: emit
`PutStructureDyn` when `T` is `unbound` *and* `Name` is `bound` at the
goal's program point; fall back to the existing builtin
otherwise. Anything we cannot prove stays on the existing path —
correctness is preserved by construction.

## Why this is the right shape of analysis for us now

Three reasons.

**It pays off on the hot predicates we already care about.** The
canonical compose pattern is

```prolog
build_term(Functor, Args, T) :- T =.. [Functor | Args].
```

where `T` is a head argument with mode `-` (output). That is the
shape used by code-generation utilities in user programs and by
Prolog-meta interpreters. Those are exactly the predicates that
benefit from the O(1) `PutStructureDyn` path because they construct
many terms.

**It opens the door for further small lowerings.** Once we have
binding state at each program point, several other optimizations
become trivial:

- `functor/3` in compose mode (functor name + arity → fresh term)
  can lower to `PutStructureDyn` plus `SetVariable` ×N.
- `arg/3` on a known-bound term can fuse into a `GetValue` against a
  pre-projected register, skipping the runtime structure traversal.
- `\+ member(X, L)` where `L` is ground at the call site can skip
  the unification machinery and use `IS.notMember` directly.

We are not doing those in this arc, but the analysis we write now is
the substrate they would all build on. The cost is paid once.

**It composes with the analyses we already have.** The Haskell target
already runs `purity_certificate`, `clause_body_analysis`, and
`demand_analysis` per clause. Adding a `binding_state_analysis` pass
slots in next to them — same input shape (head + body), same output
shape (a per-goal record list), same consumption pattern (the WAM
compiler reads it before emitting each goal).

## What we are choosing not to do

- **No backwards propagation.** We do not infer "Y must be bound
  before this goal because the goal binds it" — only forward
  reachability.
- **No interprocedural analysis.** Calls to user predicates are
  treated as opaque: outputs of a call are `unknown` unless the
  callee has a `:- mode` declaration that says otherwise.
- **No groundness lattice.** `bound` means "not unbound" — it does
  not distinguish "fully ground" from "bound to a structure with
  unbound holes". For `=..`, that is all we need: `Name` must be
  bound to an atom (which is automatically ground), and `T` must be
  unbound. The analysis answer for `T` is what the lowering
  predicate keys on.
- **No mode-driven specialisation.** We are not generating multiple
  WAM bodies per call mode. One body, with the lowering decision
  made at compile time based on what the analysis can prove.

## Soundness contract

This is the only safety property the analysis must preserve:

> If the analysis says "variable V is `unbound` at program point P",
> then in every execution that reaches P with V's binding state
> resolved, V is in fact unbound.
>
> If the analysis says "variable V is `bound` at program point P",
> then in every such execution V is bound to a non-variable term.
>
> Otherwise the analysis says `unknown`, and the compiler emits the
> conservative path.

The lowering decision is gated on the analysis returning a
*positive* answer for both `T` (unbound) and `Name` (bound). Any
`unknown` keeps the existing builtin path. There is no scenario in
which a wrong analysis result silently produces incorrect runtime
behaviour — at worst, an over-conservative analysis leaves
performance on the table.

## Relationship to `:- mode` declarations

Mercury-style `:- mode` declarations on user predicates are an
optional input to the analysis, not a requirement. When present,
they let us:

- Initialise the binding state of head arguments at the start of
  body propagation.
- Annotate predicate calls so output positions become `bound` after
  the call and input positions are required to be `bound` going in.

When absent, head arguments start at `unknown` and predicate calls
return `unknown` for all output positions. This is the conservative
default and is what existing un-annotated code in the repo will get.

The `demand_analysis` module already reads `:- mode/1` declarations
(see `core/demand_analysis.pl:133-149`). We will reuse that reader
verbatim — no new directive grammar.

## Non-goals worth naming

- **We are not stabilising a public API for the analysis result.**
  The output records are an internal detail of the WAM compiler.
  Other targets (Rust, Go, ILAsm) may grow their own analysers or
  share this one — that decision is out of scope for this arc.
- **We are not adding mode polymorphism to user code.** A predicate
  declared `:- mode pred(+, -)` cannot also be called as
  `:- mode pred(-, +)` and have both compiled correctly. If you
  need both, write two predicates. Mercury's mode polymorphism is
  an entire compiler architecture and we are not building one.
- **We are not running the analysis on lowered/native predicates.**
  Predicates that bypass the WAM (the lowered Haskell path,
  CallForeign FFI predicates, external_source LMDB-backed
  predicates) are already opaque to the WAM compiler; the
  analysis treats their outputs the same as it would a call to any
  user predicate without a mode declaration: `unknown`.

## Success metrics for this arc

The arc is done when:

1. `T =.. [Name | Args]` in compose mode (T proven unbound, Name
   proven bound) emits `PutStructureDyn` plus `SetValue`/`SetVariable`
   ×N + a `GetValue` to unify with T, instead of the list-build path.
2. The same source in decompose mode (T proven bound) keeps the
   existing builtin path. (Or, when we are not sure, also keeps it.)
3. The full `tests/test_wam_haskell_target.pl` suite remains green,
   including the existing `=../2` tests that exercise the builtin
   path.
4. A new unit test exercises a compose-mode predicate and asserts
   the generated `Predicates.hs` contains `PutStructureDyn`, and
   that the resulting cabal project builds and produces correct
   results when invoked.

We will *not* benchmark in this arc — the optimisation enables
follow-up work, but the immediate measurable win is correctness
of emitted code on the new path. Performance numbers come when a
real compose-heavy workload exists to measure against.
