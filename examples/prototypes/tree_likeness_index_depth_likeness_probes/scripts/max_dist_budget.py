#!/usr/bin/env python3
"""Shared definition of the V3 "max acyclic parent distance to root" budget.

The V3 probe's budget B is the longest acyclic parent-hop distance from a node
to the root — which is exactly the `max_dist_to_root` root-anchored metric
(docs/design/ROOT_ANCHORED_METRICS_*). This module holds TWO implementations of
that one quantity, kept side by side so they cross-check:

  - `dp_max_dist(min_dist, parents_of)` — the in-probe DP (BFS-depth
    topological order), unbounded. This is what V3 used inline.
  - `load_metric_max_dist(lmdb_dir, ...)` — read the ingest-materialised
    `metric_max_dist_to_root` table written by
    benchmarks/root-metrics/prototype/build_max_distance.py, so the budget
    becomes a precomputed lookup instead of a per-run recompute.

On a DAG the two agree EXACTLY when the materialised metric was built with
`max_depth >= graph diameter` (i.e. effectively unbounded). build_max_distance
bounds the walk at `max_depth`; the DP here is unbounded, so a too-small
max_depth truncates deep nodes and the two diverge. `load_metric_max_dist`
warns when the stored table looks bounded or rooted differently.

Run `python3 max_dist_budget.py` for a synthetic equivalence self-check.
"""
import struct
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path

import lmdb

I32 = struct.Struct("<i")


def dp_max_dist(min_dist, parents_of, warn=sys.stderr):
    """True longest acyclic parent-hop distance to root for every reachable node.

    Longest-path DP over the parent-DAG in topological order (Kahn): a node is
    resolved only after ALL its reachable parents are, so a parent at a *greater*
    BFS depth still contributes. `min_dist` maps node -> BFS depth from root
    (depth 0 == root). Returns {node: max_dist}, root(s) -> 0.

    This matches the materialised `max_dist_to_root` metric on a DAG (see
    load_metric_max_dist + this module's self-check). It supersedes the V3
    probe's earlier BFS-depth-monotone DP, which assumed every parent was
    shallower and so UNDERCOUNTED any node with a near-root shortcut parent plus
    a deeper ancestor chain (on a diamond 4->{1,3}, 3->2, 2->1 it returned
    max_dist[4]=1 instead of the true 3).

    A reachable node trapped in a parent cycle has no topological position; such
    nodes (rare in a near-acyclic category graph) get a single shallow-parent
    relaxation as a degraded estimate and are reported via `warn`.
    """
    reachable = set(min_dist)
    children = defaultdict(list)
    indeg = {n: 0 for n in reachable}
    for n in reachable:
        for p in parents_of.get(n, ()):
            if p in reachable:
                children[p].append(n)
                indeg[n] += 1
    max_dist = {n: 0 for n, d in min_dist.items() if d == 0}
    q = deque(n for n in reachable if indeg[n] == 0)
    resolved = 0
    while q:
        n = q.popleft()
        resolved += 1
        base = max_dist.get(n, 0)
        for c in children[n]:
            if base + 1 > max_dist.get(c, -1):
                max_dist[c] = base + 1
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    if resolved < len(reachable):
        leftover = [n for n in reachable if indeg[n] > 0]
        if warn is not None:
            warn.write(f"WARNING: {len(leftover):,} reachable node(s) in a parent "
                       f"cycle; using a degraded shallow-parent estimate for them\n")
        for n in sorted(leftover, key=lambda x: min_dist[x]):
            cand = [max_dist[p] for p in parents_of.get(n, ()) if p in max_dist]
            best = 1 + max(cand) if cand else min_dist[n]
            if best > max_dist.get(n, -1):
                max_dist[n] = best
    return max_dist


def load_metric_max_dist(lmdb_dir, expected_root=None, warn=sys.stderr):
    """Read the materialised `metric_max_dist_to_root` table (int32 -> int32)
    from a Phase-1 scoped LMDB. Returns {node: max_dist}.

    Emits a warning (to `warn`, or None to silence) when the stored root differs
    from `expected_root` (node ids may not correspond) or when the values reach
    the stored max_depth (the table is bounded and will disagree with the
    unbounded DP on deep nodes — rebuild with a larger --max-depth).
    """
    env = lmdb.open(str(lmdb_dir), max_dbs=16, readonly=True, subdir=True, lock=False)
    try:
        m = env.open_db(b"metric_max_dist_to_root", create=False)
        meta = env.open_db(b"meta", create=False)
        out = {}
        with env.begin() as txn:
            r = txn.get(b"metric_max_dist_to_root.root", db=meta)
            md = txn.get(b"metric_max_dist_to_root.max_depth", db=meta)
            stored_root = int(r) if r is not None else None
            stored_md = int(md) if md is not None else None
            for k, v in txn.cursor(db=m):
                out[I32.unpack(k)[0]] = I32.unpack(v)[0]
    finally:
        env.close()
    if warn is not None:
        if (expected_root is not None and stored_root is not None
                and stored_root != expected_root):
            warn.write(f"WARNING: metric root {stored_root} != probe root "
                       f"{expected_root}; node ids may not correspond\n")
        if stored_md is not None and out and max(out.values()) >= stored_md:
            warn.write(f"WARNING: metric max value reaches stored max_depth="
                       f"{stored_md}; the materialised max_dist is likely BOUNDED "
                       f"and will disagree with the unbounded DP on deep nodes. "
                       f"Rebuild build_max_distance.py with a larger --max-depth.\n")
    return out


# Path to the materialiser, for the self-check and for callers that want to
# (re)build a metric before loading it.
BUILD_MAX_DISTANCE = (Path(__file__).resolve().parents[4]
                      / "benchmarks" / "root-metrics" / "prototype"
                      / "build_max_distance.py")


def _self_check():
    """Build a synthetic diamond Phase-1 LMDB, materialise max_dist_to_root with
    an unbounded depth, and assert the corrected DP EQUALS the materialised
    metric on the diamond (4->3->2->1 = 3, not the old BFS-monotone 1)."""
    import io
    import tempfile

    # --- in-memory DP checks (no LMDB) ---
    # Chain with a shortcut: 5 reaches root via 5->1 (len 1) and
    # 5->4->3->2->1 (len 4). The old BFS-monotone DP returned 1 for node 5;
    # the corrected DP must return 4.
    chain_parents = {2: [1], 3: [2], 4: [3], 5: [4, 1]}
    chain_min = {1: 0, 2: 1, 3: 2, 4: 3, 5: 1}
    assert dp_max_dist(chain_min, chain_parents) == {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

    # Parent cycle 3->4->5->3 must not hang; the fallback fills it and warns.
    cyc_parents = {2: [1], 3: [2, 5], 4: [3], 5: [4]}
    cyc_min = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    buf = io.StringIO()
    got = dp_max_dist(cyc_min, cyc_parents, warn=buf)
    assert got == {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}, got
    assert "cycle" in buf.getvalue(), buf.getvalue()

    # child -> [parents]; root = 1. Diamond: 4 reaches root via 4->1 (len 1)
    # and 4->3->2->1 (len 3); longest acyclic = 3.
    adj = {2: [1], 3: [2], 4: [1, 3]}
    root, max_depth = 1, 50
    with tempfile.TemporaryDirectory() as tmp:
        lmdb_dir = Path(tmp) / "scoped"
        env = lmdb.open(str(lmdb_dir), max_dbs=16, map_size=10 * 1024 * 1024, subdir=True)
        cp = env.open_db(b"category_parent", dupsort=True, create=True)
        meta = env.open_db(b"meta", create=True)
        with env.begin(write=True) as txn:
            for c, ps in adj.items():
                for p in ps:
                    txn.put(I32.pack(c), I32.pack(p), db=cp, dupdata=True)
            txn.put(b"scoped_root", str(root).encode(), db=meta)
            txn.put(b"scoped_max_depth", str(max_depth).encode(), db=meta)
        env.sync(); env.close()

        rc = subprocess.run(
            [sys.executable, str(BUILD_MAX_DISTANCE), "--lmdb", str(lmdb_dir),
             "--root", str(root), "--max-depth", str(max_depth)],
            capture_output=True, text=True)
        assert rc.returncode == 0, rc.stderr

        # in-probe DP over the same graph
        parents_of = {c: ps for c, ps in adj.items()}
        children_of = {}
        for c, ps in adj.items():
            for p in ps:
                children_of.setdefault(p, []).append(c)
        min_dist = {root: 0}
        q = deque([root])
        while q:
            n = q.popleft()
            for c in children_of.get(n, ()):
                if c not in min_dist:
                    min_dist[c] = min_dist[n] + 1
                    q.append(c)
        dp = dp_max_dist(min_dist, parents_of)
        loaded = load_metric_max_dist(lmdb_dir, expected_root=root)

        # Corrected DP and the materialised metric agree: true longest acyclic
        # path (4->3->2->1 = 3), root -> 0.
        assert loaded == {1: 0, 2: 1, 3: 2, 4: 3}, loaded
        assert dp == loaded, f"DP {dp} != metric {loaded}"
    print("max_dist_budget self-check OK: corrected DP == materialised metric "
          "{1:0, 2:1, 3:2, 4:3} (node 4 = 3, the true longest path)")


if __name__ == "__main__":
    _self_check()
