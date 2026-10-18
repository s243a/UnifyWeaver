# Hybrid WAM — `transitive_closure2` contract (strict R+)

Fleet-wide semantics for the shared recursive-kernel kind
`transitive_closure2` (detector shape `tc(X,Y) :- edge(X,Y).` /
`tc(X,Y) :- edge(X,Z), tc(Z,Y).`).

This document is the authoritative contract. Target handlers (native /
foreign) and their tests must agree with it. Enumeration **order** is
non-semantic unless a target documents otherwise.

## Relation

Use **strict transitive closure R+** of the configured binary edge
predicate:

- A `Target` is a solution for `Source` iff there exists a path of
  **one or more** edges from `Source` to `Target`.
- `Source` is a solution when a **self-loop** or a **nonempty cycle**
  returns to `Source`. It is **not** a solution merely because
  traversal starts there (that would be reflexive R*).
- Duplicate edges and multiple paths do not create duplicate solutions:
  each reachable `Target` is emitted **exactly once** (set semantics).

## Modes

| Call | Contract |
|---|---|
| `tc(+Source, -Target)` | Stream every reachable Target exactly once; fail if the set is empty. |
| `tc(+Source, +Target)` | Succeed **once** iff a ≥1-edge path exists; otherwise fail cleanly. |
| Unbound / non-atom Source | Fail cleanly (no enumeration of all graph sources in this contract). |

`tc/2` remains a **stream-valued relation** at the Prolog interface.
Native kernels may collect a deterministic set internally, then stream
through the target’s foreign multi-solution / retry mechanism
(`FFIStreamRetry`, `finish_foreign_results`, `ForeignMulti`, …).

## Continuations

Bound calls and stream retries must preserve registers, bindings, trail,
continuation PC, cut barrier, B0 stack, and choice points per the
target’s existing foreign-stream ABI. Do not invent a parallel retry
protocol for TC2.

## Ordering policy

- Cross-target comparisons use **normalized sorted sets**.
- Each target should be deterministic for a fixed fact snapshot when
  practical (stable BFS/DFS discovery). Do **not** add a sort solely for
  cross-target string equality without documenting the cost.

## What this is not

| Mechanism | Why it is out of scope for this contract |
|---|---|
| Generic WAM recursive `tc/2` (`no_kernels(true)`) | Correctness path via clause recursion; on cyclic graphs it represents **path proofs** (may duplicate or not terminate). Use as an oracle only on **acyclic** fixtures after set-normalization. |
| `reachableToRoot*` (F# LMDB/CSR) | Demand-pruning BFS over reverse/category_child edges; includes root. |
| `category_ancestor` / `bidirectional_ancestor` | Depth/budget/heuristic-bound ancestor search, not R+ closure. |

## Implementation style (per target)

Inline versus Mustache is a target-local engineering choice:

- Prefer Mustache for large handler bodies or substantial generated
  fragments (Haskell/F# already do this for TC2).
- Small composable fragments may stay inline (C/Rust/Scala/Go style).
- Shared Mustache `{{match}}`/`{{case}}` libraries are appropriate when
  multiple kernels reuse meaningful generation logic — not for trivial
  one-liners.
- Follow the surrounding target’s organization unless a move materially
  improves readability; document reasons when introducing or
  substantially reorganizing a kernel template.

## Reference implementations (already R+)

Rust (`collect_native_transitive_closure_nodes`), Scala
(`emit_scala_kernel_handler` for `transitive_closure2`), Go, and Elixir
discover nodes **only via outgoing edges** (visited/seen is not seeded
with Source). Align other handlers to that pattern.

## Fixtures / oracle

See `tests/fixtures/tc2_contract_oracle.pl` (finite BFS oracle + fixture
matrix) and `tests/test_wam_tc2_contract_parity.pl`.

## History

- 2026-07-16: Contract introduced (`TC2-CONTRACT-PARITY`). Discovery found
  Rust/Scala/Go/Elixir already R+; Haskell/F#/C/R (and LLVM stream)
  excluded Source unconditionally; LLVM bound distance fast-path treated
  `Source==Target` as success with zero edges (R*-like) — out of TC2
  stream scope but noted for LLVM follow-up.
