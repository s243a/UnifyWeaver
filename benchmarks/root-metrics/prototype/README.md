# Root-anchored metrics — materialiser/query prototype

**Status: prototype harness.** These are hand-coded Python tools that
materialise and serve the root-anchored metrics described in
`docs/design/ROOT_ANCHORED_METRICS_*`. They proved the semantics and the
algorithmic collapse (path-enumeration → node-DP) end-to-end, but they are
**not** the intended production path — see "Why this is a prototype" below.

## What's here

| File | Role | Nature |
|------|------|--------|
| `build_max_distance.py` | Materialise `max_dist_to_root` (max,+) via a length-bucketed node-DP over `category_parent`. | **Numeric work — generation candidate** |
| `build_effective_distance.py` | Materialise `effective_distance` (sum,×decay) via the same node-DP, accumulating decayed path counts. | **Numeric work — generation candidate** |
| `query_root_metric.py` | Serve any materialised `metric_*` sub-db as an O(1) keyed lookup; `--histogram`; `--verify` against an independent recompute oracle. | Mixed: lookup is harness glue; the `recompute_*` oracle is a deliberate independent check |

The three canonical metrics are: `min_dist_to_root` (min,+),
`max_dist_to_root` (max,+), `effective_distance` (sum,×decay) — one recurrence
skeleton, three semirings (spec §3).

## Why this is a prototype (not the production path)

UnifyWeaver's premise is that you declare the **what** — here the recurrence,
boundary and semiring, via the `:- root_metric(Name/Arity, [...])` directive
(`src/unifyweaver/core/root_metric.pl`) — and the system *transpiles* it into
optimized target code. The node-DPs in `build_max_distance.py` and
`build_effective_distance.py` are exactly the ingest-time **difference-equation /
path-aggregate numeric work** that should be **generated** from that directive,
not hand-written. This is the implementation plan's **Phase 5**: *"factor into
one semiring-parameterised lowering template so a new metric is a new
`(aggregate, combine)` row, not new traversal code."* These scripts jumped ahead
of that step to validate the math first; retiring them in favour of a generated
materialiser is the intended next move.

Two deliberate exceptions to "generate everything":

- **The lookup glue** (`query_root_metric.py`'s keyed `get` / histogram) is a thin
  harness over LMDB; not worth transpiling, legitimately hand-coded.
- **The `recompute_*` verify oracles** (here and in the `*_nodedp` tests) are kept
  hand-written *on purpose*: their value is being a second, independent
  implementation that cross-checks the materialiser. Generating them from the
  same spec would defeat the check.

## Related, not moved

- `examples/benchmark/build_scoped_subtree_lmdb.py` — the demand-set **scoping**
  tool. It also materialises `min_dist_to_root` as a rider during its BFS, but its
  primary job is subtree scoping and it is wired into the matrix-bench generator
  and the enwiki ingest path, so it stays under `examples/benchmark/`. (Its
  min-dist materialiser is the same class of generation candidate.)

## Tests

Equivalence + serving tests live under `tests/` (repo convention), not here:
`tests/test_max_distance_nodedp.py`, `tests/test_effective_distance_nodedp.py`,
`tests/test_query_root_metric.py`. They build tiny hand-checked fixtures
(diamond/tree DAGs) and run in ~1.7 s total; each skips when `python-lmdb` is
absent.

## See also

`docs/design/ROOT_ANCHORED_METRICS_{PHILOSOPHY,SPECIFICATION,IMPLEMENTATION_PLAN}.md`,
`src/unifyweaver/core/root_metric.pl`.
