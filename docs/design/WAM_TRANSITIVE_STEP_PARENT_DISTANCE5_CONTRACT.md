# Hybrid WAM — `transitive_step_parent_distance5` contract (shortest-positive correlated step/parent)

Fleet-wide semantics for the shared recursive-kernel kind
`transitive_step_parent_distance5` (detector shape
`tspd(X, Y, Y, X, 1) :- edge(X, Y).` /
`tspd(X, Y, S, P, D) :- edge(X, M), tspd(M, Y, _, P, D1), D is D1 + 1.`).

This document is the authoritative contract. Target handlers (native /
foreign) and their tests must agree with it. Enumeration **order** is
non-semantic unless a target documents otherwise.

## Relation

For a positive-length directed path

```
v0 = S, v1 = Step, …, v(D-1) = Parent, vD = Target
```

define:

```
dist+(S, T) = minimum D ≥ 1 for which such a path exists

tspd+(S, T, Step, Parent, D) ⇔
    D = dist+(S, T) ∧
    some length-D path has exactly that first hop and final predecessor
```

Emit every distinct correlated quadruple:

```
(Target, Step, Parent, Distance)
```

Consequences:

- Preserve every `(Step, Parent)` pair realized by a shortest path.
- **Never** collect Step and Parent sets independently and cross-product them.
- Ignore longer paths after a shorter positive distance is known.
- Suppress duplicate quadruples from duplicate edges or alternate proofs.
- Direct `S → T` emits `(T, T, S, 1)`.
- Self-loop `S → S` emits `(S, S, S, 1)`.
- A shortest nonempty cycle back to Source emits its actual first hop and
  final predecessor (never distance 0).
- Cycles terminate (finite BFS).
- Sink and unknown sources return no solutions.
- Ordering is non-semantic; compare normalized complete quadruple sets.

## Algorithmic reference

Level-synchronous (FIFO) BFS from Source with queue seed `(Source, 0)`.
For each edge-discovered node `N` maintain:

- `dist[N]` — minimum positive distance;
- `pairs[N]` — set of correlated `(Step, Parent)` pairs on shortest paths.

When expanding `U → V` at `nd = dist_U + 1`:

- if `U` is Source (`dist_U = 0`): candidate pair is `(V, Source)`;
- otherwise: for each distinct `Step` appearing in `pairs[U]`, candidate
  pair is `(Step, U)`.

On first discovery of `V`, record `dist` and `pairs` and enqueue.
On rediscovery at the **same** distance, union new pairs (do not
re-enqueue). Ignore longer paths.

FIFO level order ensures equal-distance first-hop information is
complete before a node’s descendants are expanded.

**Adversarial diamond (correlated pairs):**

```
a→b, a→c, b→p, c→q, p→t, q→t
```

From `a`, target `t` emits exactly `(t,b,p,3)` and `(t,c,q,3)` — never
`(t,b,q,3)` or `(t,c,p,3)`.

## Modes

With Source bound, support all **16** bound/free combinations of
`Target`, `Step`, `Parent`, and `Distance`.

- Source must dereference to a bound atom.
- Bound Target, Step, and Parent must be atoms.
- Bound Distance must be a positive integer.
- Variables in Source, compounds, numeric atom slots, and invalid
  distances fail cleanly.
- Preserve aliases among Source and all atom outputs.
- Atom-output ↔ Distance aliases fail cleanly.
- Filter **complete** quadruples jointly.
- If a later component conflicts, roll back all bindings from that
  candidate and continue scanning.
- Exhaustion must leave no residual bindings or owned choice point.

## Continuations

Reuse each target’s established foreign-stream ABI (`FFIStreamRetry`,
generic tuple streams, `ForeignMulti` + transactional `applyBindings`,
`kernel_quad_iter`, ordinary Elixir choice points, C foreign stream).
Do **not** invent a TSPD5-specific retry protocol. First yield and retry
use identical transactional filtering and snapshots.

## What this is not

| Mechanism | Why out of scope |
|---|---|
| Generic WAM recursive `tspd/5` (`no_kernels(true)`) | Path proofs; unsafe oracle on cyclic/multipath graphs. Acyclic fixtures only after set-normalization. |
| Legacy all-path DFS (pre-#3830 Rust/Elixir quarantine) | Emits non-shortest distances; nonterminating on cycles — rejected. |
| First-route BFS (one step/parent per target) | Loses equal-shortest correlated pairs — rejected. |
| Independent Step × Parent cross-product | Invents impossible correlations — rejected. |
| `transitive_parent_distance4` | No first-hop column. |
| Weighted / A* kernels | Separate kinds; do not alter. |
| LLVM without a real handler | Remain capability-gated. |

## Capability honesty

- Advertise native TSPD5 only when a real handler exists.
- F#: allow-list only after Mustache handler ships.
- LLVM: remain gated unless a real handler is implemented.

## C edge isolation

C graph handlers must not conflate distinct edge relations through a
single global edge bag. Edge storage is keyed by predicate/relation
identity; an executable two-predicate regression proves isolation.

## Fixtures / oracle

See `tests/fixtures/tspd5_contract_oracle.pl` and
`tests/test_wam_tspd5_contract_parity.pl`.

## History

- 2026-07-17: Contract introduced (`TSPD5-CONTRACT-PARITY-FS`), based on
  reviewed #3830 head `c809ff650f066009768bea435c7f348e24f1a62e`.
  Pre-change: Rust/Elixir quarantined all-path DFS; Go/Scala/R/Haskell
  first-route BFS with Source seeding; C first-match over global
  `category_edges`; F#/LLVM ungated or absent. Aligned to correlated
  shortest-positive quadruples; removed #3830 temporary TSPD5 isolation
  helpers once executable coverage passed.
