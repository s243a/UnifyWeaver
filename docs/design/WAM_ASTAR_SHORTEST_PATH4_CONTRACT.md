# Hybrid WAM — `astar_shortest_path4` contract (correctness-safe A*)

Fleet-wide semantics for the shared recursive-kernel kind
`astar_shortest_path4` (detector shape

```
astar(X, Y, _Dim, W) :- edge(X, Y, W).
astar(X, Y, Dim, TotalW) :-
    edge(X, Mid, W1),
    astar(Mid, Y, Dim, RestW),
    TotalW is RestW + W1.
```

with ternary weighted `edge/3` and optional ternary
`direct_dist_pred/3` heuristic oracle).

This document is the authoritative contract. Target handlers (native /
foreign) and their tests must agree with it.

## Relation

```
astar(Source, Target, Dim, FloatCost)
```

`FloatCost` is the **minimum finite nonnegative edge-weight sum** over
directed paths from Source to Target — the same optimum as finite
Dijkstra. Dim and heuristic values may affect **scheduling** only; they
must never change the final shortest distance.

## Inputs

| Arg | Mode / type |
|-----|-------------|
| Source | Bound atom |
| Target | Bound atom |
| Dim | Strictly positive bound integer (`Dim > 0`) |
| Cost | Free or bound float |

Invalid or unbound Source/Target/Dim fail the call cleanly. Bound Cost
must be a float under the runtime’s typed unification rules — integer
`3` does not match float `3.0`.

## Results

- Exactly one `FloatCost` equal to the Dijkstra-minimum path weight.
- Unreachable Target fails (no solution).
- **Source = Target** succeeds once with `FloatCost = 0.0`.
- Integral totals still emit as float terms (`FloatTerm(3.0)`, never
  `IntTerm(3)`).
- Duplicate edges, zero-cost edges, equal-cost routes, and cycles
  terminate correctly (stale PQ entries discarded).

## Heuristic safety

Treat direct-distance data as a **scheduling hint**, not an unchecked
correctness assumption.

- Missing `(Node, Target)` heuristic means **0.0**.
- Relevant heuristic values must be finite, nonnegative numbers.
- For duplicate `(Node, Target)` entries, use the **minimum** valid value.
- A malformed heuristic row consulted for Source or a discovered node
  before Target settles fails the **complete** call.
- Unreachable heuristic rows are irrelevant.
- An overestimating heuristic must not produce a wrong route: every
  target must still return the Dijkstra-minimum result.
- Targets may degrade safely to Dijkstra when admissibility is unknown.
- If `g^Dim + h^Dim` (or `g + h`) remains the scheduling score,
  establish optimality using **g-cost bounds** (or an equivalent safe
  method) before terminating — never “first time Target is popped under
  an unchecked `f`”.

Reference safe algorithm:

1. If Source = Target, return `0.0`.
2. Run a finite Dijkstra (PQ ordered by `g`, stale entries discarded)
   from Source until Target is settled or the open set is empty.
3. Optionally use `h` / Dim only as a secondary key for exploration
   order; never early-stop on Target under an unchecked `f`.
4. On settle, emit `g[Target]` as a float.

## Edge validation

Validate every row returned while expanding a settled node before Target
settles:

- atom destination;
- finite nonnegative numeric weight;
- zero allowed;
- any malformed expanded row fails the complete call;
- no partial result escapes;
- rows in unreachable components, or in branches not expanded before the
  goal-directed search settles Target, are irrelevant.

## Modes and unification

With Source, Target, and Dim bound as above:

| Cost |
|------|
| free |
| bound float |

Preserve ordinary aliases involving Cost. Exhaustion leaves no residual
bindings or owned choice point. Surrounding older choice points,
`call`, and `cut` reuse each target’s established foreign / stream ABI
(single-output float binding).

## Capability honesty

- Advertise native A\* only when a real handler and weighted /
  direct-distance materialization exist.
- LLVM already has native A\* lowering — preserve and align it.
- F#: add to the allow-list only after a real Mustache handler and
  dual relation-keyed `WcFfiWeightedFacts` materialization exist.
- Preserve WSP3, TSPD5, TPD4, TD3, TC2, and unrelated kernels.

## History

- 2026-07-18: Contract introduced (`ASTAR4-CONTRACT-PARITY-FS`), stacked
  on reviewed WSP3 draft head
  `b7995b5ab93a18bb63e840dbf6491bf2a537b1c5`. Pre-contract fleet:
  missing-h defaults of 0.0 vs 1.0; unsafe first-Target termination under
  overestimating heuristics; divergent Dim handling; C fixed-256 + global
  bags; Haskell IntMap-as-function bug; F# correctly gated.
