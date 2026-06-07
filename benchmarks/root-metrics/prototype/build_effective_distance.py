#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
"""
build_effective_distance.py — materialise the effective-distance metric via a
length-bucketed node-DP (ROOT_ANCHORED_METRICS Phase 3).

The category_ancestor effective-distance query sums, per seed, (L+1)^(-n) over
every path of length L from the seed up to the root. Enumerating paths is
exponential on a dense DAG (the enwiki content graph: ~24 ms/seed, ~17 h full).
This computes the identical quantity in O(edges * max_depth) by counting paths
bucketed by length instead of enumerating them (spec §5):

    count[N][0] = 1 if N == root else 0
    count[N][L] = Σ_{P in parents(N)} count[P][L-1]          (L = 1..max_depth)
    S(N)        = Σ_{L=1..max_depth} count[N][L] * (L+1)^(-n)

Only two length-layers are held at once (count[*][L-1] and count[*][L]) plus the
running S accumulator, so memory is O(reachable nodes), not O(nodes*max_depth).

SEMANTICS (spec §4.1): this is the *walk* count (cycles(bounded)). On a DAG it
equals the simple-path sum the per-path kernel computes; on a cyclic graph it
differs (walks may revisit). The depth bound makes it finite and exact to
max_depth either way.

Output: a `metric_effective_distance` sub-db (int32_le node -> f64 LE value)
written into the scoped LMDB next to the graph, + meta provenance. Querying the
metric is then a single keyed get.

Usage:
  python3 build_effective_distance.py --lmdb <scoped_dir> [--root ID]
      [--max-depth D] [--exponent N] [--map-size-gib G]
Root / max_depth default to the scoped DB's stored meta (scoped_root / _max_depth).
"""
import argparse
import struct
import sys
import time
from array import array
from collections import defaultdict
from pathlib import Path

import lmdb

I32 = struct.Struct("<i")
F64 = struct.Struct("<d")


def enc_i32(i):
    return I32.pack(i)


def dec_i32(b):
    return I32.unpack(b)[0]


def load_edges(env, cp_db):
    """Load category_parent (child -> parent) as two parallel int arrays."""
    children = array("i")
    parents = array("i")
    with env.begin() as txn:
        for k, v in txn.cursor(db=cp_db):
            children.append(dec_i32(k))
            parents.append(dec_i32(v))
    return children, parents


def effective_distance(children, parents, root, max_depth, exponent):
    """Length-bucketed node-DP. Returns {node: S(node)} for S(node) > 0."""
    prev = {root: 1.0}                 # count[*][0]
    s = defaultdict(float)             # running S(N)
    n_edges = len(children)
    for depth in range(1, max_depth + 1):
        cur = defaultdict(float)
        for i in range(n_edges):
            c = prev.get(parents[i])
            if c:
                cur[children[i]] += c
        if not cur:
            break                      # no longer-walks possible
        factor = (depth + 1) ** (-exponent)
        for node, cnt in cur.items():
            s[node] += cnt * factor
        prev = cur
    return s


def read_meta(env, key, default=None):
    try:
        meta = env.open_db(b"meta", create=False)
    except lmdb.NotFoundError:
        return default
    with env.begin() as txn:
        v = txn.get(key, db=meta)
    return v.decode() if v is not None else default


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lmdb", required=True, help="scoped LMDB dir (has category_parent + meta)")
    ap.add_argument("--root", type=int, default=None, help="root id (default: meta scoped_root)")
    ap.add_argument("--max-depth", type=int, default=None, help="default: meta scoped_max_depth")
    ap.add_argument("--exponent", type=float, default=5.0, help="distance exponent n (default 5)")
    ap.add_argument("--map-size-gib", type=float, default=4.0)
    args = ap.parse_args()

    t0 = time.time()
    env = lmdb.open(str(args.lmdb), max_dbs=16, subdir=True,
                    map_size=int(args.map_size_gib * 1024 ** 3))
    cp_db = env.open_db(b"category_parent", dupsort=True, create=False)

    root = args.root if args.root is not None else int(read_meta(env, b"scoped_root", "-1"))
    max_depth = args.max_depth if args.max_depth is not None else int(read_meta(env, b"scoped_max_depth", "10"))
    if root < 0:
        sys.stderr.write("error: no root (pass --root or build the scoped DB with a meta marker)\n")
        return 1

    children, parents = load_edges(env, cp_db)
    sys.stderr.write(f"loaded {len(children)} edges in {time.time()-t0:.1f}s\n")

    s = effective_distance(children, parents, root, max_depth, args.exponent)
    sys.stderr.write(f"node-DP: {len(s)} nodes with S>0 (root={root}, max_depth={max_depth}, "
                     f"n={args.exponent}) in {time.time()-t0:.1f}s\n")

    metric_db = env.open_db(b"metric_effective_distance", create=True)
    meta_db = env.open_db(b"meta", create=True)
    with env.begin(write=True) as txn:
        for node, val in s.items():
            txn.put(enc_i32(node), F64.pack(val), db=metric_db)
        txn.put(b"metric_effective_distance.root", str(root).encode(), db=meta_db)
        txn.put(b"metric_effective_distance.max_depth", str(max_depth).encode(), db=meta_db)
        txn.put(b"metric_effective_distance.exponent", str(args.exponent).encode(), db=meta_db)
        txn.put(b"metric_effective_distance.aggregate", b"sum", db=meta_db)
    env.sync()
    env.close()
    sys.stderr.write(f"wrote metric_effective_distance ({len(s)} entries) in {time.time()-t0:.1f}s\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
