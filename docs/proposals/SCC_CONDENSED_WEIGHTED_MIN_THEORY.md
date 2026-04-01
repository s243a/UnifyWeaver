# SCC-Condensed Weighted Min: Theory

## Problem Statement

Consider a recursive weighted path relation of the form:

```prolog
path(X, Y, Acc) :-
    edge(X, Y),
    weight(X, W),
    Acc is W.

path(X, Z, Acc) :-
    edge(X, Y),
    weight(X, W),
    path(Y, Z, Acc1),
    Acc is Acc1 + W.
```

with per-path uniqueness:

- no node may repeat within a single path

and `min` tabling semantics on the accumulator:

- for each `(source, target)`, return only the minimum accumulated cost

## Why Scalar Best-Known Pruning Fails in General

In a cyclic graph, two distinct simple paths can reach the same
intermediate node `N` with:

- different accumulated costs
- different visited sets

The path with the better current cost is not always the one with the
best continuation, because it may already have used a node that the
other path still needs.

So the implication

```text
best_cost_to(N) is enough state
```

is false in general under simple-path semantics.

That is why exact weighted `min` requires more than a scalar table on the
original graph.

## SCC Condensation

Given a directed graph `G = (V, E)`, compute its strongly connected
components:

```text
SCC(G) = {C1, C2, ..., Ck}
```

and build the condensation graph:

```text
G* = (SCC(G), E*)
```

where there is an edge `Ci -> Cj` in `G*` iff there exists an edge in
`G` from a node in `Ci` to a node in `Cj`, and `Ci != Cj`.

Standard result:

- `G*` is a DAG

This matters because once a path leaves an SCC, it can never return to
that SCC in the condensation graph.

## Why Condensation Helps

The difficult visited-set ambiguity is caused by cyclic revisitation
possibility. SCC condensation limits that ambiguity:

- between components, path order is acyclic
- repeated reconsideration of the same global cyclic region disappears
- dynamic programming becomes possible on the outer graph

The remaining hard part is local:

- inside an SCC, simple-path constraints still matter
- but the scope of exact path-state reasoning is confined to the
  component rather than the whole graph

## Theoretical Decomposition

Weighted `min` on the original graph can be decomposed into:

1. local weighted simple-path transfer inside each SCC
2. acyclic composition between SCCs on the condensation DAG

Informally:

```text
global_min_path = DAG composition of local SCC transfer functions
```

The win comes from the fact that the second phase no longer needs
visited-set-sensitive search over the whole graph.

## Soundness Envelope

Let a path in the original graph correspond to a sequence of SCCs:

```text
C0 -> C1 -> ... -> Cn
```

Because the condensation graph is acyclic, every valid path visits each
component at most once.

So if we summarize the minimal transfer cost from:

- an SCC entry boundary
- to an SCC exit boundary

then the outer composition across SCCs can be treated as a DAG shortest
path problem.

This does not solve the internal SCC problem automatically, but it
changes the asymptotic shape of the global problem:

- expensive exact search is local
- cheap DAG dynamic programming is global

## Expected Runtime Consequence

The current exact frontier algorithm pays path-state costs at every
level of the graph. SCC condensation should reduce this by:

- shrinking the search space seen by the global algorithm
- reducing dominance comparisons across globally unrelated cycles
- making the number of exact visited-state comparisons proportional more
  to SCC size than to total graph size

This is especially attractive when:

- SCCs are small or moderate
- the condensation DAG is much smaller than the raw graph
- weighted `min` is asked over many seeds

## Relationship to Potentials

Potential functions or admissible lower bounds can still help, but they
are best layered on top of the condensed graph:

- SCC condensation removes structural cyclicity
- potentials can then guide search or prune within the condensed model

That is a cleaner path than applying heuristic pruning directly to the
raw graph with full visited-state dependence.

## What Must Be Proven Empirically

The proposal rests on three empirical questions:

1. Are benchmark SCCs small enough that local exact search is cheap?
2. Does condensation materially reduce weighted `min` runtime relative to
   the current frontier approach?
3. Does the benefit hold at `5k` and `10k`, where weighted `min` should
   become clearly faster than `All`?

Those are benchmark questions, not just code questions, so the
implementation plan must include SCC structure reporting as part of
evaluation.
