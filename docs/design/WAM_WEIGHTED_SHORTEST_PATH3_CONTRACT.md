# Hybrid WAM — `weighted_shortest_path3` contract (finite nonnegative Dijkstra)

Fleet-wide semantics for the shared recursive-kernel kind
`weighted_shortest_path3` (detector shape
`wsp(X, Y, W) :- edge(X, Y, W).` /
`wsp(X, Y, TotalW) :- edge(X, Mid, W1), wsp(Mid, Y, RestW), TotalW is RestW + W1.`
with ternary `edge(From, To, Weight)`).

This document is the authoritative contract. Target handlers (native /
foreign) and their tests must agree with it. Enumeration **order** is
non-semantic unless a target documents otherwise.

## Relation

For a finite directed weighted graph, define:

```
cost(S, T) = minimum sum of edge weights over directed paths from S to T
```

Emit exactly one distinct pair:

```
(Target, FloatCost)
```

for every reachable Target **other than Source**, where `FloatCost = cost(S, Target)`.

Consequences:

- Source remains excluded even through a self-loop or positive cycle.
  This is a deliberate WSP3 convention — do **not** import TSPD5’s
  positive-cycle Source result.
- Duplicate edges and equal-cost alternative paths do not duplicate a
  result (one minimum per Target).
- A more expensive direct edge loses to a cheaper multi-edge path.
- Cycles terminate (Dijkstra with stale PQ entries discarded).
- Sink and unknown sources produce no solutions.
- Ordering is non-semantic; compare normalized complete pair sets.
- An integral sum must still be emitted as a float (e.g. `FloatTerm(3.0)`),
  never `IntTerm(3)`.

## Algorithmic reference

Ordinary Dijkstra from Source with priority-queue seed `(Source, 0.0)`.
Maintain `dist[N]` for discovered nodes. When expanding `U` with cost `c`:

- discard stale entries where `c > dist[U]`;
- for each outgoing edge `U → V` with weight `w`, candidate `nc = c + w`;
- if `V` is undiscovered or `nc < dist[V]`, update and enqueue.

Emit every `(N, dist[N])` with `N ≠ Source`.

## Edge domain and validation

Integer and floating edge facts convert to IEEE-754 binary64.

Supported edge weights are finite and **nonnegative**. Zero is allowed.

When expanding a node reachable from Source, every returned row must
have:

- an atom destination;
- a numeric weight;
- a finite, nonnegative weight.

A reachable invalid row (negative, NaN, infinite, nonnumeric, or
non-atom destination) must **fail the complete kernel call cleanly**.
Never silently discard such a row and return a potentially incorrect
shortest path.

Rows keyed by an unreachable non-atom source are irrelevant and need
not be scanned.

## Modes

With Source bound, support all four Target/Cost combinations:

| Target | Cost |
|--------|------|
| free   | free |
| bound  | free |
| free   | bound |
| bound  | bound |

Requirements:

- Source must dereference to a bound atom.
- Bound Target must be an atom.
- Bound Cost must be a float under the runtime’s typed unification
  rules — integer `3` does not match float `3.0`.
- Variables in Source and invalid types fail cleanly.
- Preserve ordinary aliases involving Source, Target, and Cost.
- Source=Target fails (Source excluded).
- Target=Cost and Source=Cost aliases fail cleanly.
- Filter complete `(Target, Cost)` pairs jointly.
- If an early candidate conflicts, roll back its partial bindings and
  continue scanning later candidates.
- Exhaustion leaves no residual bindings or owned choice point.

## Continuation / streams

Reuse each target’s established foreign-stream ABI and the transactional
pair-stream fixes from TPD4/TSPD5 (#3830/#3838). Do **not** invent a
WSP-specific retry protocol.

First yield and retry must use identical transactional filtering and
snapshots. Aggregates project from the already jointly filtered pair
stream.

## Capability honesty

- Advertise native WSP3 only when a real handler exists.
- LLVM already has a real Dijkstra handler — preserve it.
- F#: add to the allow-list only after a real Mustache handler and
  weighted-fact materialization exist.
- A\* (`astar_shortest_path4`) is out of scope for this contract.

## History

- 2026-07-17: Contract introduced (`WSP3-CONTRACT-PARITY-FS`), based on
  reviewed TSPD5 PR #3838 final head
  `83bda2d0857e53fe91c49d8100ac3af3218e4939`. Pre-contract fleet:
  Dijkstra existed on Rust/Haskell/LLVM/C/Go/Scala/R/Elixir with
  divergent float/int emission, C fixed-256 + first-only unbound
  results + global weighted bag, F# capability-gated, ordinary WAM
  enumerating all path weights (not a valid multipath/cycle oracle).
