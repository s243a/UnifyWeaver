# SCC-Condensed Weighted Min: Philosophy

## Why This Exists

The current weighted `min` support for `PathAwareAccumulationNode` is
now split into two practical cases.

That is a different situation from counted shortest path:

- counted `min` on `PathAwareTransitiveClosureNode` already delivers the
  expected performance win
- positive-additive weighted `min` on `PathAwareAccumulationNode` now
  also has a fast runtime path based on layered dynamic programming
- broader weighted `min` shapes with more path-sensitive continuation
  still need a better structural solution than exact frontier tracking

So the remaining next step is no longer "make weighted `min` fast at
all." It is narrower: improve the cases that are not reducible to the
current positive-additive fast path.

## Core Intuition

For the remaining hard cases, the expensive part of weighted `min` is
not arithmetic. It is the fact that correctness depends on more than:

- current node
- current accumulated cost

It also depends on:

- which nodes have already been used in the path

That visited-set sensitivity is what prevents the simple "best cost per
target" rule from being universally sound in cyclic graphs.

The philosophical goal of SCC condensation is to move complexity out of
the search state and into the graph representation:

- collapse strongly connected regions into a single component graph
- make the outer problem closer to dynamic programming on a DAG
- reserve path-specific reasoning for the hard part only

## Why SCC Condensation Fits This Problem

Weighted minimum path is much easier on an acyclic graph than on a
general graph:

- no revisiting a component later through another cycle
- no repeated global reconsideration of the same recursive region
- much stronger pruning and dynamic-programming opportunities

The Wikipedia-style category graph used in the benchmark is not fully
acyclic, but the measured SCC structure is very favorable: only a small
number of cyclic SCCs remain, and the largest cyclic SCC is small. That
makes SCC condensation attractive because:

- it removes global cyclic structure from the outer graph
- it may sharply reduce the number of states that need exact visited-set
  tracking
- it gives us a path toward handling the broader weighted `min` cases
  that are not already solved by the positive-additive fast path

## What This Is Not

This is not primarily a calculus-of-variations problem.

The more relevant analogies are:

- shortest path on weighted graphs
- semiring / dynamic-programming evaluation
- branch-and-bound with graph condensation
- potential-guided search after structural reduction

If we later add admissible lower bounds or node potentials, those should
be treated as refinements on top of the condensed graph, not as the
starting point.

## Design Principle

The runtime should prefer the **cheapest sound state model** available
for a workload.

That suggests the following hierarchy:

1. Counted `min` on simple path-aware closure
   Use scalar best-known pruning.

2. Weighted `min` on a positive-additive recurrence with hop bound
   Use layered dynamic programming over `(node, depth)`.

3. Weighted `min` on an SCC-condensed component DAG
   Use dynamic-programming or component-level best-known pruning where
   possible.

4. Weighted `min` inside cyclic SCCs
   Use exact path-aware state only where condensation does not remove the
   ambiguity.

This keeps the hard machinery where it is truly needed instead of paying
for it everywhere.

## Cross-Target Importance

This matters beyond the C# query engine.

If the right abstraction is "weighted minimum path over an SCC-condensed
graph", then that abstraction is portable:

- C# query engine can specialize it in the runtime
- Rust can lower to the same condensed-graph algorithm natively
- other DFS-style targets can follow later

That is better than teaching every target a fragile, path-frontier-heavy
special case independently.

## Evaluation Standard

This proposal should only be considered successful if it achieves both:

1. exact agreement with the current `All` semantics for the supported
   weighted `min` workloads
2. a clear runtime advantage over `All`, especially at larger scales

That bar is now met for the positive-additive benchmarked case. The
remaining success criterion for this SCC direction is narrower:

1. preserve exact output agreement for the broader weighted `Min`
   workload class
2. beat the exact frontier fallback on those broader cases
