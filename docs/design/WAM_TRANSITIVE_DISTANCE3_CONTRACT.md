# Hybrid WAM — `transitive_distance3` contract (dist+)

Fleet-wide semantics for the shared recursive-kernel kind
`transitive_distance3` (detector shape
`td(X,Y,1) :- edge(X,Y).` /
`td(X,Y,D) :- edge(X,Z), td(Z,Y,D1), D is D1 + 1.`).

This document is the authoritative contract. Target handlers (native /
foreign) and their tests must agree with it. Enumeration **order** is
non-semantic unless a target documents otherwise.

## Relation

Define **shortest positive distance**:

```
dist+(S, T) = minimum number of edges in any path from S to T
              containing at least one edge
```

Consequences:

- Emit each reachable `Target` **exactly once**, paired with
  `dist+(Source, Target)`.
- Duplicate edges and multiple paths never duplicate a target.
- Unequal alternate paths select the **shorter** distance.
- `Source` is emitted only when reached through a self-loop or a
  nonempty cycle:
  - self-loop gives `(Source, 1)`;
  - otherwise use the shortest positive cycle length back to Source;
  - **never** emit distance zero (that would be reflexive R*).
- A sink or unknown source produces no results.
- Algorithmic reference: finite BFS from Source where `visited` tracks
  nodes discovered **via an outgoing edge** (do not seed with Source).
  First discovery distance is minimal.

## Modes

| Call | Contract |
|---|---|
| `td(+S, -T, -D)` | Stream every `(Target, Distance)` pair exactly once; fail if empty. |
| `td(+S, +T, -D)` | Succeed once with `D = dist+(S,T)`; fail if unreachable. |
| `td(+S, -T, +D)` | Stream targets whose minimum distance equals `D`. |
| `td(+S, +T, +D)` | Succeed once iff `D` is exactly `dist+(S,T)`. |

- Source must be a bound atom; unbound / non-atom Source fails cleanly.
- Zero, negative, mismatched, or non-integer bound distances fail cleanly.
- `td/3` remains a **stream-valued relation** at the Prolog interface for
  unbound outputs. Native kernels may collect a deterministic set
  internally, then stream through the target’s foreign multi-solution /
  retry mechanism (`FFIStreamRetry`, `finish_foreign_results`,
  `ForeignMulti`, foreign choice points, …).

## Continuations

Bound calls and stream retries must preserve registers, bindings, trail,
continuation PC, cut barrier, B0 stack, and choice points per the
target’s existing foreign-stream ABI. Do not invent a parallel retry
protocol for TD3. `(Target, Distance)` pairing must survive every retry.

## Ordering policy

- Cross-target comparisons use **normalized sorted tuple sets**.
- Each target should be deterministic for a fixed fact snapshot when
  practical (stable BFS discovery). Do **not** add a sort solely for
  cross-target string equality without documenting the cost.

## What this is not

| Mechanism | Why out of scope |
|---|---|
| Generic WAM recursive `td/3` (`no_kernels(true)`) | Path proofs; on cyclic graphs may duplicate or not terminate. Use only on **acyclic** fixtures after set-normalization. |
| Per-path simple-path enumeration | Emits the same target at several distances — rejected. |
| `reachableToRoot*` (F# LMDB/CSR) | Demand-pruning BFS over reverse/`category_child`; includes root; no distances. |
| `transitive_closure2` | Reachability only (no distance column). |

## Implementation style (per target)

Inline versus Mustache is a target-local engineering choice (same policy
as the TC2 contract). Prefer Mustache for substantial kernel bodies
(Haskell/F#); keep small adapters inline.

## Reference pattern

Discover neighbors via outgoing edges; insert into `visited` only on
first discovery; record `(neighbor, depth+1)`. Matches Go/Scala/R BFS
shape **after** removing Source seeding, and rejects Rust/Elixir
per-path enumeration.

## Fixtures / oracle

See `tests/fixtures/td3_contract_oracle.pl` (finite BFS oracle + fixture
matrix with **literal** expected tuples) and
`tests/test_wam_td3_contract_parity.pl`.

## History

- 2026-07-16: Contract introduced (`TD3-CONTRACT-PARITY-FS`). Discovery
  found Haskell/Scala/Go/R/LLVM generally BFS-min but excluding Source;
  Rust/Elixir enumerated simple paths; C returned first match only;
  LLVM bound `Source==Target` returned distance 0. Aligned to dist+.
  F# gained `nativeKernel_transitive_distance` (Mustache) on the
  existing multi-output `FFIStreamRetry` binder.
