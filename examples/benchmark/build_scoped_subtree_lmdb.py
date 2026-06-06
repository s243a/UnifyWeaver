#!/usr/bin/env python3
"""Build a *scoped subtree* LMDB from a full-graph Phase-1 category LMDB.

Demand-set scoping belongs at INGEST time, not per query: when a good root is
known (e.g. the *Articles* subtree in simplewiki, or *Main topic
classifications* in full enwiki), materialise the reachable subtree once into a
self-contained LMDB.  Queries against the scoped DB then never re-run the
`reachable_to_root` BFS — `demand_set == all nodes` by construction, so the
runtime demand filters are no-ops.

Semantics mirror the F# WAM target (the cleanest existing spec):
  * node set = `reachableToRoot(root, max_depth)` — BFS DOWN `category_child`
    (parent -> child), bounded by `max_depth`
    (templates/targets/fsharp_wam/lmdb_fact_source.fs.mustache).
  * edges kept = every `category_parent` edge (child -> parent) whose BOTH
    endpoints are in the node set (cf. `loadCategoryParentDemandDict`).

The output is a drop-in fixture for the WAM matrix benches: it writes the
scoped `category_parent` / `category_child` DUPSORT sub-dbs, the restricted
`s2i` / `i2s`, plus `root_ids.txt`, `article_category.tsv`, and `metadata.json`
in the same layout `simplewiki_post_ingest.py` produces.

Usage:
  python3 build_scoped_subtree_lmdb.py \
      --src  data/benchmark/simplewiki_cats/lmdb_resident \
      --root 2 \
      --out  data/benchmark/simplewiki_cats/lmdb_scoped \
      [--max-depth 10] [--map-size-gib 2] [--fixture-dir DIR]

`--fixture-dir` (default: parent of --out) is where the sidecar txt/json files
are written.
"""
import argparse
import collections
import json
import struct
import sys
import time
from pathlib import Path

import lmdb

I32 = struct.Struct("<i")


def enc(i: int) -> bytes:
    return I32.pack(i)


def dec(b: bytes) -> int:
    return I32.unpack(b)[0]


def reachable_to_root(txn, cc_db, root: int, max_depth: int) -> set:
    """BFS DOWN category_child (parent -> children), bounded by max_depth.

    Mirrors F# `reachableToRoot`: returns every node reachable from `root`
    within `max_depth` hops, i.e. the set of nodes that can reach `root` going
    UP through category_parent.
    """
    visited = {root}
    frontier = [root]
    cur = txn.cursor(db=cc_db)
    depth = 0
    while frontier and depth < max_depth:
        nxt = []
        for node in frontier:
            if cur.set_key(enc(node)):
                for v in cur.iternext_dup(keys=False, values=True):
                    child = dec(v)
                    if child not in visited:
                        visited.add(child)
                        nxt.append(child)
        frontier = nxt
        depth += 1
    return visited


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="source full-graph LMDB dir")
    ap.add_argument("--root", required=True, type=int, help="root node id (LMDB int)")
    ap.add_argument("--out", required=True, help="output scoped LMDB dir")
    ap.add_argument("--max-depth", type=int, default=10,
                    help="BFS depth bound (must match the query's max_depth; default 10)")
    ap.add_argument("--map-size-gib", type=float, default=2.0,
                    help="output map size in GiB (default 2)")
    ap.add_argument("--fixture-dir", default=None,
                    help="dir for sidecar files (default: parent of --out)")
    args = ap.parse_args()

    src_dir = Path(args.src)
    out_dir = Path(args.out)
    fixture_dir = Path(args.fixture_dir) if args.fixture_dir else out_dir.parent
    map_size = int(args.map_size_gib * 1024 * 1024 * 1024)
    t0 = time.time()

    if not src_dir.exists():
        sys.stderr.write(f"error: source LMDB dir not found: {src_dir}\n")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    src_env = lmdb.open(str(src_dir), max_dbs=16, readonly=True, subdir=True,
                        lock=False, map_size=map_size)
    src_cp = src_env.open_db(b"category_parent", dupsort=True)
    # category_child must exist (Phase-1 layout / post-ingest builds it).
    try:
        src_cc = src_env.open_db(b"category_child", dupsort=True, create=False)
    except lmdb.NotFoundError:
        sys.stderr.write(
            "error: source LMDB has no 'category_child' sub-db; run the "
            "post-ingest reverse-edge pass first.\n")
        return 1

    # 1. demand set = reachable subtree of root (F# reachableToRoot semantics).
    with src_env.begin() as txn:
        demand = reachable_to_root(txn, src_cc, args.root, args.max_depth)
    sys.stderr.write(
        f"reachable_to_root(root={args.root}, max_depth={args.max_depth}) "
        f"= {len(demand)} nodes\n")
    if len(demand) <= 1:
        sys.stderr.write(
            "warning: demand set is trivial (root has no children within "
            "max_depth) — check the root id and that category_child is "
            "populated.\n")

    # 2. write scoped edges: keep child->parent iff BOTH endpoints in demand.
    out_env = lmdb.open(str(out_dir), max_dbs=16, subdir=True, map_size=map_size)
    out_cp = out_env.open_db(b"category_parent", dupsort=True, create=True)
    out_cc = out_env.open_db(b"category_child", dupsort=True, create=True)
    kept_edges = 0
    scanned = 0
    children_in_scope = set()
    with src_env.begin() as rtxn, out_env.begin(write=True) as wtxn:
        cur = rtxn.cursor(db=src_cp)
        for k, v in cur:
            scanned += 1
            child = dec(k)
            if child not in demand:
                continue
            parent = dec(v)
            if parent not in demand:
                continue
            wtxn.put(k, v, db=out_cp, dupdata=True, overwrite=True)
            wtxn.put(v, k, db=out_cc, dupdata=True, overwrite=True)
            kept_edges += 1
            children_in_scope.add(child)
            if scanned % 500_000 == 0:
                sys.stderr.write(f"  ...scanned {scanned} edges, kept {kept_edges}\n")
    sys.stderr.write(
        f"edges: scanned {scanned}, kept {kept_edges} (both endpoints in demand)\n")

    # 3. copy s2i/i2s entries restricted to demand nodes, so the scoped DB is
    #    self-contained for string<->int seed/root resolution + result decode.
    nid_to_str = {}
    try:
        src_i2s = src_env.open_db(b"i2s", create=False)
        src_s2i = src_env.open_db(b"s2i", create=False)
        out_i2s = out_env.open_db(b"i2s", create=True)
        out_s2i = out_env.open_db(b"s2i", create=True)
        with src_env.begin() as rtxn, out_env.begin(write=True) as wtxn:
            cur = rtxn.cursor(db=src_i2s)
            for k, v in cur:
                if dec(k) in demand:
                    wtxn.put(k, v, db=out_i2s)
                    nid_to_str[dec(k)] = v.decode("utf-8", "replace")
            cur = rtxn.cursor(db=src_s2i)
            for k, v in cur:
                if dec(v) in demand:
                    wtxn.put(k, v, db=out_s2i)
        sys.stderr.write(f"s2i/i2s: copied {len(nid_to_str)} demand-node mappings\n")
    except lmdb.NotFoundError:
        sys.stderr.write("note: source has no s2i/i2s sub-dbs; skipping (int-native fixture)\n")

    out_env.sync()
    out_env.close()
    src_env.close()

    # 4. sidecar fixture files (same layout as simplewiki_post_ingest.py).
    fixture_dir.mkdir(parents=True, exist_ok=True)
    nodes = sorted(demand)
    with open(fixture_dir / "root_ids.txt", "w") as f:
        f.write(f"{args.root}\n")
    with open(fixture_dir / "article_category.scoped.tsv", "w") as f:
        f.write("article\tcategory\n")
        for nid in nodes:
            name = nid_to_str.get(nid, str(nid))
            f.write(f"{name}\t{name}\n")
    with open(fixture_dir / "metadata.scoped.json", "w") as f:
        json.dump({
            "scoped_from": str(src_dir),
            "root_id": args.root,
            "max_depth": args.max_depth,
            "node_count": len(demand),
            "edge_count": kept_edges,
            "children_in_scope": len(children_in_scope),
            "out_lmdb": str(out_dir),
            "build_seconds": round(time.time() - t0, 2),
        }, f, indent=2)

    sys.stderr.write(
        f"scoped subtree LMDB written to {out_dir} "
        f"({len(demand)} nodes, {kept_edges} edges) in {time.time() - t0:.2f}s\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
