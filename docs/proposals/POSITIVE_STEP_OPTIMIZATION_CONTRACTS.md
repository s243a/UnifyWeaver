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

The compiler-visible metadata mechanism is now implemented through the
 existing constraint system:

```prolog
:- constraint(path/3, [positive_step(3)]).
```

This proves that the accumulator-bearing position is driven by a
 strictly positive recursive step for optimization purposes.

The earlier alternative spellings remain historical design ideas:

Possible shapes:

```prolog
:- optimization_contract(path/3, [min_step_positive]).
```

or a target-neutral metadata term attached to the accumulator position.

The chosen syntax keeps the design goal:

- the user states positivity once
- the compiler records it in plan metadata
- C#, Rust, and other targets can lower against the same contract

## Recommended Sequencing

1. implement explicit guard recognition now
2. keep runtime inference as a fallback for backwards compatibility
3. keep runtime inference as a fallback for backwards compatibility
4. use the declarative `constraint/2` contract before extending Rust lowering

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

The declarative metadata mechanism is now available through
 `constraint/2` with `positive_step(N)`.

The remaining work before broader cross-target adoption is not syntax
 design but deciding how broadly targets should rely on:

- explicit positive guards
- declarative `positive_step/1` metadata
- or both

Current implementation status:

- C# weighted `Min` lowering consumes both explicit guards and
  `positive_step/1` metadata
- Rust native lowering for counted and positive-additive weighted `Min`
  now consumes the same contracts during native emission
- Go native lowering for counted and positive-additive weighted `Min`
  now consumes the same contracts during native emission
