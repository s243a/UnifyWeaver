# Hybrid WAM — `transitive_parent_distance4` contract (shortest-positive parents)

Fleet-wide semantics for the shared recursive-kernel kind
`transitive_parent_distance4` (detector shape
`pd(X, Y, X, 1) :- edge(X, Y).` /
`pd(X, Y, P, D) :- edge(X, Z), pd(Z, Y, P, D1), D is D1 + 1.`).

This document is the authoritative contract. Target handlers (native /
foreign) and their tests must agree with it. Enumeration **order** is
non-semantic unless a target documents otherwise.

## Relation

Define **shortest positive parent-distance triples**:

```
dist+(S, T) = minimum number of edges in any path from S to T
              containing at least one edge

parents+(S, T) = { P | exists a path of length dist+(S,T) from S to T
                       whose immediate predecessor of T is P }

tpd(S, T, P, D) ⇔  D = dist+(S, T) ∧ P ∈ parents+(S, T)
```

Consequences:

- Emit every distinct `(Target, Parent, Distance)` triple.
- When several immediate predecessors occur on equally short paths,
  emit **each** valid parent once. Never keep only the first-discovered
  parent.
- Duplicate edges and alternate proofs never duplicate a triple.
- Unequal alternate paths select the shorter distance (and parents on
  those shortest paths only).
- A direct edge `S→T` produces `P=S` and `D=1`.
- `Source` is emitted only when reached through a self-loop or a
  nonempty cycle:
  - self-loop gives `(Source, Source, 1)`;
  - otherwise use the shortest positive cycle length and every
    immediate predecessor of Source on such a cycle;
  - **never** emit distance zero.
- A sink or unknown source produces no results.
- Cycles terminate (finite BFS with distance/parent-set maps).
- Algorithmic reference: BFS from Source with queue seed
  `(Source, 0)` where `dist` / `parents` track nodes discovered
  **via an outgoing edge** (do not seed Source into `dist`). On first
  discovery of `N` at depth `d`, record `dist[N]=d` and
  `parents[N]={predecessor}`; on rediscovery at the **same** depth,
  add the predecessor to `parents[N]` without re-enqueueing; ignore
  longer paths.

## Modes

All eight bound/free combinations of `T`, `P`, and `D` (with `S`
bound) are supported. Bound calls succeed once per matching triple
under the target’s foreign multi-solution / retry ABI, or fail cleanly
when none match.

| Call | Contract |
|---|---|
| `tpd(+S, -T, -P, -D)` | Stream every triple exactly once; fail if empty. |
| `tpd(+S, +T, -P, -D)` | Stream parents of `T` at `dist+(S,T)`. |
| `tpd(+S, -T, +P, -D)` | Stream targets for which `P` is a shortest parent. |
| `tpd(+S, -T, -P, +D)` | Stream triples whose minimum distance equals `D`. |
| `tpd(+S, +T, +P, -D)` | Succeed once with `D=dist+(S,T)` iff `P∈parents+(S,T)`. |
| `tpd(+S, +T, -P, +D)` | Stream parents of `T` iff `D=dist+(S,T)`. |
| `tpd(+S, -T, +P, +D)` | Stream targets with that parent at distance `D`. |
| `tpd(+S, +T, +P, +D)` | Succeed once iff the triple matches. |

- Source must be a bound atom; unbound / non-atom Source fails cleanly.
- Bound `T` and `P` must be atoms; bound `D` must be a positive integer.
- Zero, negative, mismatched, or non-integer bound distances fail cleanly.
- Argument aliases obey ordinary WAM unification. Aliasing an atom
  output with the integer distance normally fails unless unification
  genuinely permits it (same rule as TD3 after `1e2a888`).
- Compare **complete triples**, never independently projected columns,
  when asserting parity. Aggregate slicing that projects a column
  (Elixir findall of `T` alone, etc.) is a consumer choice and must
  still be fed from jointly filtered triples.

## Continuations

Bound calls and stream retries must preserve registers, bindings, trail,
continuation PC, cut barrier, B0 stack, and choice points per the
target’s existing foreign-stream ABI (`FFIStreamRetry`,
`finish_foreign_results`, `ForeignMulti`, ordinary function-backed
choice points, `kernel_triple_iter`, …). Do **not** invent a parallel
retry protocol for TPD4. `(Target, Parent, Distance)` pairing must
survive every retry; first yield and retries use identical unification
and snapshot rules.

## Ordering policy

- Cross-target comparisons use **normalized sorted triple sets**.
- Each target should be deterministic for a fixed fact snapshot when
  practical (stable BFS discovery). Do **not** add a sort solely for
  cross-target string equality without documenting the cost.

## What this is not

| Mechanism | Why out of scope |
|---|---|
| Generic WAM recursive `tpd/4` (`no_kernels(true)`) | Path proofs; on cyclic graphs may duplicate or not terminate. Use only on **acyclic** fixtures after set-normalization. |
| Per-path simple-path enumeration | Emits the same target at several distances / duplicate parents — rejected. |
| First-parent-only BFS | Loses equal-shortest parents — rejected. |
| Seeding Source at distance 0 into `visited`/`dist` | Suppresses self-loop / cycle-to-Source — rejected (same lesson as TD3). |
| `reachableToRoot*` (F# LMDB/CSR) | Demand-pruning BFS over reverse/`category_child`; includes root; no parent/distance columns. |
| `transitive_distance3` | Distance only (no parent column). |
| `transitive_step_parent_distance5` | Extra first-hop column; separate kind. |

## Implementation style (per target)

Inline versus Mustache is a target-local engineering choice (same policy
as the TC2/TD3 contracts). Prefer Mustache for substantial kernel bodies
(Haskell/F#); keep small adapters inline.

## Capability honesty

- Advertise native TPD4 only when a real handler exists and is registered.
- LLVM: no native TPD4 handler in this fleet pass — remain capability-gated
  / undetected; do not register a nonexistent foreign kind.
- F#: allow-list `transitive_parent_distance4` only after
  `nativeKernel_transitive_parent_distance` ships.

## Fixtures / oracle

See `tests/fixtures/tpd4_contract_oracle.pl` (literal expected triples +
finite BFS cross-check) and `tests/test_wam_tpd4_contract_parity.pl`.

## History / resolved ambiguity

- 2026-07-17: Contract introduced (`TPD4-CONTRACT-PARITY-FS`), depending
  on TD3 (`#3821`) and follow-up `1e2a888` (joint bound/alias filtering
  on multi-output streams).
- Pre-change audit found two camps: BFS first-parent + Source-seeded
  (Haskell/Go/Scala/R/C) vs path-enumerating DFS (Rust/Elixir). F# and
  LLVM had no native TPD. No prior authoritative TPD4 contract existed;
  detector comments said “shortest path” / one parent per target, which
  **conflicts** with equal-shortest diamond graphs and with TD3’s
  Source-via-cycle rule. Fleet rule chosen: **all** equal-shortest
  parents + Source only via positive cycle/self-loop (never distance 0),
  matching the task statement rather than accidental first-parent
  iteration order.
