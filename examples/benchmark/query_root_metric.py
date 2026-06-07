#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
"""
query_root_metric.py — answer a materialised root-anchored metric as a lookup.

This is the query surface for the ingest-materialised path of
docs/design/ROOT_ANCHORED_METRICS_*. Once build_scoped_subtree_lmdb.py has
written a `metric_<name>` sub-db, answering the metric for a node is a single
keyed get — no graph traversal at all. For `min_dist_to_root` that means the
"how far is this category from the root" query is O(1) per node at full scale.

  min_dist_to_root(Root, Node, D)  ==  metric_min_dist_to_root[Node]

Usage:
  python3 query_root_metric.py --lmdb data/benchmark/enwiki_cats_correct/lmdb_scoped \
      7345184 1234 5678            # look up these node ids
  python3 query_root_metric.py --lmdb <dir> --nodes-file ids.txt
  python3 query_root_metric.py --lmdb <dir> --histogram
  python3 query_root_metric.py --lmdb <dir> --verify [--sample N]

`--verify` recomputes the metric from scratch and asserts every materialised
value matches — an independent check that the lookup table is correct, separate
from the builder that wrote it. min_dist_to_root is recomputed by BFS over
category_child; effective_distance by the length-bucketed node-DP over
category_parent (float comparison with 1e-6 relative tolerance).

Use `--metric effective_distance` to serve the materialised effective-distance
table (f64 values); the default metric is min_dist_to_root.
"""
import argparse
import collections
import struct
import sys
import time
from pathlib import Path

import lmdb

I32 = struct.Struct("<i")
F64 = struct.Struct("<d")

# Metrics whose stored value is an int32 (distance). Everything else (e.g.
# effective_distance) is an f64.
INT_METRICS = {"min_dist_to_root", "max_dist_to_root"}


def enc(i):
    return I32.pack(i)


def dec(b):
    return I32.unpack(b)[0]


def decode_value(metric, b):
    """Decode a stored metric value: int32 for distance metrics, else f64."""
    if metric in INT_METRICS:
        return I32.unpack(b)[0]
    return F64.unpack(b)[0]


def metric_meta(env, meta_db, name):
    """Read the stored provenance for metric `name` (root, max_depth, aggregate)."""
    out = {}
    with env.begin() as txn:
        for field in ("root", "max_depth", "aggregate", "exponent"):
            v = txn.get(f"metric_{name}.{field}".encode(), db=meta_db)
            if v is not None:
                out[field] = v.decode()
    return out


def recompute_min_dist(env, cc_db, root, max_depth):
    """Independent oracle: BFS down category_child from root; layer = min dist."""
    dist = {root: 0}
    frontier = [root]
    depth = 0
    with env.begin() as txn:
        cur = txn.cursor(db=cc_db)
        while frontier and depth < max_depth:
            nxt = []
            for node in frontier:
                if cur.set_key(enc(node)):
                    for v in cur.iternext_dup(keys=False, values=True):
                        child = dec(v)
                        if child not in dist:
                            dist[child] = depth + 1
                            nxt.append(child)
            frontier = nxt
            depth += 1
    return dist


def recompute_effective_distance(env, cp_db, root, max_depth, exponent):
    """Independent node-DP recompute (matches build_effective_distance.py):
    S(N) = Σ_L count[N][L]·(L+1)^(-exponent), count over walks up category_parent.
    Used as the --verify oracle for effective_distance (catches storage
    corruption / a stale table; algorithm correctness is proven on a DAG by
    tests/test_effective_distance_nodedp.py)."""
    children, parents = [], []
    with env.begin() as txn:
        for k, v in txn.cursor(db=cp_db):
            children.append(dec(k))
            parents.append(dec(v))
    prev = {root: 1.0}
    s = collections.defaultdict(float)
    for depth in range(1, max_depth + 1):
        cur = collections.defaultdict(float)
        for i in range(len(children)):
            c = prev.get(parents[i])
            if c:
                cur[children[i]] += c
        if not cur:
            break
        factor = (depth + 1) ** (-exponent)
        for node, cnt in cur.items():
            s[node] += cnt * factor
        prev = cur
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lmdb", required=True, help="scoped LMDB dir with a metric_* sub-db")
    ap.add_argument("--metric", default="min_dist_to_root", help="metric name (default min_dist_to_root)")
    ap.add_argument("nodes", nargs="*", type=int, help="node ids to look up")
    ap.add_argument("--nodes-file", help="file of node ids (one per line)")
    ap.add_argument("--histogram", action="store_true", help="print the value histogram")
    ap.add_argument("--verify", action="store_true",
                    help="cross-check the materialised table against a fresh BFS oracle")
    ap.add_argument("--sample", type=int, default=0,
                    help="with --verify, check this many table entries (0 = all)")
    args = ap.parse_args()

    metric_sub = f"metric_{args.metric}".encode()
    env = lmdb.open(str(args.lmdb), max_dbs=16, readonly=True, subdir=True, lock=False,
                    map_size=4 * 1024 ** 3)
    try:
        metric_db = env.open_db(metric_sub, create=False)
    except lmdb.NotFoundError:
        sys.stderr.write(f"error: no '{metric_sub.decode()}' sub-db in {args.lmdb}; "
                         f"rebuild with build_scoped_subtree_lmdb.py (--min-dist)\n")
        return 1
    meta_db = env.open_db(b"meta", create=False)
    meta = metric_meta(env, meta_db, args.metric)
    if meta:
        sys.stderr.write(f"[{args.metric}] root={meta.get('root')} "
                         f"max_depth={meta.get('max_depth')} aggregate={meta.get('aggregate')}\n")

    # Collect node ids to look up.
    ids = list(args.nodes)
    if args.nodes_file:
        with open(args.nodes_file) as f:
            ids += [int(x) for x in (line.split()[0] for line in f if line.strip())]

    # O(1) lookups.
    if ids:
        t0 = time.perf_counter()
        with env.begin() as txn:
            for nid in ids:
                v = txn.get(enc(nid), db=metric_db)
                if v is None:
                    print(f"{nid}\tunreachable")
                else:
                    print(f"{nid}\t{decode_value(args.metric, v)}")
        dt = time.perf_counter() - t0
        sys.stderr.write(f"{len(ids)} lookups in {dt*1e3:.3f} ms "
                         f"({len(ids)/dt:,.0f} lookups/s)\n")

    if args.histogram:
        if args.metric in INT_METRICS:
            hist = collections.Counter()
            with env.begin() as txn:
                for _k, v in txn.cursor(db=metric_db):
                    hist[dec(v)] += 1
            total = sum(hist.values())
            sys.stderr.write(f"value histogram over {total} nodes:\n")
            for val in sorted(hist):
                print(f"{val}\t{hist[val]}")
        else:
            # Continuous metric (f64): print summary stats rather than a
            # value/count histogram (every value is effectively unique).
            vals = []
            with env.begin() as txn:
                for _k, v in txn.cursor(db=metric_db):
                    vals.append(decode_value(args.metric, v))
            n = len(vals)
            if n:
                vals.sort()
                mean = sum(vals) / n
                sys.stderr.write(f"summary over {n} nodes:\n")
                print(f"count\t{n}")
                print(f"min\t{vals[0]:.6g}")
                print(f"max\t{vals[-1]:.6g}")
                print(f"mean\t{mean:.6g}")
                print(f"median\t{vals[n // 2]:.6g}")
            else:
                sys.stderr.write("summary: table is empty\n")

    rc = 0
    if args.verify:
        if not meta.get("root"):
            sys.stderr.write("error: no stored root for this metric; cannot verify\n")
            return 2
        root, max_depth = int(meta["root"]), int(meta["max_depth"])
        if args.metric == "min_dist_to_root":
            cc_db = env.open_db(b"category_child", dupsort=True, create=False)
            oracle = recompute_min_dist(env, cc_db, root, max_depth)
            def matches(stored, want):
                return want == stored
        elif args.metric == "effective_distance":
            exponent = float(meta.get("exponent", 5))
            cp_db = env.open_db(b"category_parent", dupsort=True, create=False)
            oracle = recompute_effective_distance(env, cp_db, root, max_depth, exponent)
            def matches(stored, want):
                return abs(want - stored) <= 1e-6 * max(1.0, abs(want))
        else:
            sys.stderr.write(f"--verify supports min_dist_to_root and "
                             f"effective_distance (got {args.metric})\n")
            return 2
        mismatches = 0
        checked = 0
        with env.begin() as txn:
            cur = txn.cursor(db=metric_db)
            for k, v in cur:
                node, stored = dec(k), decode_value(args.metric, v)
                want = oracle.get(node)
                if want is None or not matches(stored, want):
                    mismatches += 1
                    if mismatches <= 10:
                        sys.stderr.write(f"  mismatch node {node}: stored {stored} "
                                         f"oracle {want}\n")
                checked += 1
                if args.sample and checked >= args.sample:
                    break
        # also: every oracle node should be in the table (when checking all)
        missing = 0
        if not args.sample:
            with env.begin() as txn:
                for node in oracle:
                    if txn.get(enc(node), db=metric_db) is None:
                        missing += 1
        sys.stderr.write(f"verify: checked {checked} entries, {mismatches} mismatches, "
                         f"{missing} oracle nodes missing from table\n")
        if mismatches == 0 and missing == 0:
            sys.stderr.write("verify: OK — materialised metric matches the recompute oracle\n")
        else:
            sys.stderr.write("verify: FAILED\n")
            rc = 1

    env.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
