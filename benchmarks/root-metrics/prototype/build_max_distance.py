#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
"""
build_max_distance.py — materialise max_dist_to_root via a length-bucketed
node-DP (ROOT_ANCHORED_METRICS Phase 4, the (max,+) longest-walk semiring).

max_dist_to_root(Root, Node, D) is the length of the *longest* walk in parent
hops from Node up to Root, under the depth bound. It is the (max,+) companion
of min_dist_to_root (min,+, materialised by build_scoped_subtree_lmdb.py) and
effective_distance (sum,×decay, by build_effective_distance.py): one recurrence
skeleton, three semirings.

    layer[0]    = {root}
    layer[L]    = { child | (child -> parent) edge, parent in layer[L-1] }   (L >= 1)
    max_dist(N) = max { L | N in layer[L] }                                  (root -> 0)

A node stays active in every later layer it can still be extended to, so the
final overwrite of max_dist(N) = L lands on the largest depth it appears at —
exactly count[N][L] > 0 for the largest L of the effective-distance DP, but
tracked as set membership instead of a float count. Only two layers are held at
once, so memory is O(reachable nodes).

SEMANTICS (spec §4.1): this is the longest *walk* (cycles(bounded)). On a DAG it
equals the longest simple path; on a cyclic graph the depth bound makes it
finite and the value saturates at max_depth.

Output: a `metric_max_dist_to_root` sub-db (int32_le node -> int32_le distance)
+ meta provenance. Querying it is then a single keyed get (query_root_metric.py).

Usage:
  python3 build_max_distance.py --lmdb <scoped_dir> [--root ID] [--max-depth D]
      [--map-size-gib G]
Root / max_depth default to the scoped DB's stored meta (scoped_root / _max_depth).
"""
import argparse
import struct
import sys
import time
from pathlib import Path

import lmdb

I32 = struct.Struct("<i")


def enc_i32(i):
    return I32.pack(i)


def dec_i32(b):
    return I32.unpack(b)[0]


def load_edges(env, cp_db):
    """Load category_parent (child -> parent) as parent -> [children]."""
    children = []
    parents = []
    with env.begin() as txn:
        for k, v in txn.cursor(db=cp_db):
            children.append(dec_i32(k))
            parents.append(dec_i32(v))
    return children, parents


def max_distance(children, parents, root, max_depth):
    """Length-bucketed node-DP. Returns {node: max_dist(node)}, root -> 0."""
    layer = {root}                 # nodes reachable by a walk of `depth-1` steps
    maxd = {root: 0}
    n_edges = len(children)
    for depth in range(1, max_depth + 1):
        nxt = set()
        for i in range(n_edges):
            if parents[i] in layer:
                nxt.add(children[i])
        if not nxt:
            break                  # no longer walks possible
        for node in nxt:
            maxd[node] = depth     # monotone depth -> ends at the largest
        layer = nxt
    return maxd


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

    maxd = max_distance(children, parents, root, max_depth)
    sys.stderr.write(f"node-DP: {len(maxd)} nodes (root={root}, max_depth={max_depth}) "
                     f"in {time.time()-t0:.1f}s\n")

    metric_db = env.open_db(b"metric_max_dist_to_root", create=True)
    meta_db = env.open_db(b"meta", create=True)
    with env.begin(write=True) as txn:
        for node, val in maxd.items():
            txn.put(enc_i32(node), enc_i32(val), db=metric_db)
        txn.put(b"metric_max_dist_to_root.root", str(root).encode(), db=meta_db)
        txn.put(b"metric_max_dist_to_root.max_depth", str(max_depth).encode(), db=meta_db)
        txn.put(b"metric_max_dist_to_root.aggregate", b"max", db=meta_db)
    env.sync()
    env.close()
    sys.stderr.write(f"wrote metric_max_dist_to_root ({len(maxd)} entries) in {time.time()-t0:.1f}s\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
