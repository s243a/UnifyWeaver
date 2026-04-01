# Positive Step Optimization Contracts

## Problem

The C# query engine now has a fast path for weighted `min` accumulation
 when the recursive step is:

- additive
- strictly positive
- depth bounded

Today the runtime can prove this by inspecting the generated arithmetic
 expression and, if needed, sampling the auxiliary relation values at
 runtime.

That is useful, but it is not the ideal long-term contract for native
 lowering across targets.

We want two stronger mechanisms:

1. an explicit Prolog-side guard that proves positivity in the query
2. a declarative metadata mechanism the compiler can consume directly

## Mechanism 1: Explicit Positivity Guards

The immediate mechanism is an ordinary Prolog guard in the recursive
 definition:

```prolog
path(X, Y, Acc) :-
    edge(X, Y),
    weight(X, Cost),
    Cost > 0,
    Acc is Cost.

path(X, Z, Acc) :-
    edge(X, Y),
    weight(X, Cost),
    Cost > 0,
    path(Y, Z, Acc1),
    Acc is Acc1 + Cost.
```

The compiler can recognize this as an optimization contract:

- `Cost > 0`
- `0 < Cost`
- `Cost >= 1`
- `1 =< Cost`

This keeps semantics explicit in the query itself and is easy to
 implement immediately.

### Pros

- clear and local to the predicate definition
- target-independent at the source level
- no new syntax required

### Cons

- mixes optimization-relevant facts with query logic
- not ideal when positivity is already known structurally elsewhere
- still narrower than a full declarative constraint system

## Mechanism 2: Declarative Positive-Step Metadata

The longer-term mechanism should be a declarative compiler-visible
 contract, distinct from the query body.

Possible shapes:

```prolog
:- constraint(path/3, [positive_step(3)]).
```

or

```prolog
:- optimization_contract(path/3, [min_step_positive]).
```

or a target-neutral metadata term attached to the accumulator position.

The exact syntax is still open, but the design goal is:

- the user states positivity once
- the compiler records it in plan metadata
- C#, Rust, and other targets can lower against the same contract

## Recommended Sequencing

1. implement explicit guard recognition now
2. keep runtime inference as a fallback for backwards compatibility
3. design the declarative metadata mechanism next
4. use one of these explicit contracts before extending Rust lowering

## Why Rust Should Wait For This

Rust should not depend purely on runtime inference copied from the C#
 path.

It should lower against an explicit source-level or plan-level contract
 such as:

- positive guard recognized by the compiler
- positive-step metadata declared by the user

That keeps the cross-target story coherent and avoids benchmark-specific
 magic.

## Current Status

The immediate explicit-guard mechanism is the minimum acceptable
 contract before moving to Rust on weighted `min`.

The declarative metadata mechanism is still proposed work, but should be
 designed before we rely on weighted `min` native lowering broadly
 across targets.
