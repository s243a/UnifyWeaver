#!/usr/bin/env python3
"""Post-process the text-keyed simplewiki LMDB:
- add category_child reverse-edge DUPSORT sub-db
- pick root by graph topology: node with the largest reachable subtree
  (since simplewiki's newer schema makes 'Physics' un-resolvable without
   joining page+linktarget tables — see ingest notes)
- emit seed_ids.txt (all category int IDs that appear as children)
- emit root_ids.txt
- emit synthetic article_category.tsv (category -> category)
- emit metadata.json
"""
import collections
import json
import struct
import sys
import time
from pathlib import Path

import lmdb

FIXTURE = Path("data/benchmark/simplewiki_cats")
LMDB_DIR = FIXTURE / "lmdb_resident"

t0 = time.time()
env = lmdb.open(
    str(LMDB_DIR),
    max_dbs=16,
    map_size=2 * 1024 * 1024 * 1024,
    subdir=True,
    readonly=False,
)
cp_db = env.open_db(b"category_parent", dupsort=True)
cc_db = env.open_db(b"category_child", dupsort=True, create=True)

# pass 1: walk category_parent, write reversed edges to category_child,
# build adjacency in memory for root selection
cp_count = 0
children_of = collections.defaultdict(list)
parents_of = collections.defaultdict(list)
with env.begin(write=True) as txn:
    cur = txn.cursor(db=cp_db)
    for k, v in cur:
        txn.put(v, k, db=cc_db, dupdata=True, overwrite=True)
        cp_count += 1
        child_id = struct.unpack("<i", k)[0]
        parent_id = struct.unpack("<i", v)[0]
        children_of[parent_id].append(child_id)
        parents_of[child_id].append(parent_id)
        if cp_count % 100_000 == 0:
            sys.stderr.write(f"  ...{cp_count} reverse-edges written\n")

env.sync()

# pass 2: pick root by largest descendant subtree
# Candidates: nodes with no parents (true roots) — typically a handful
candidate_roots = sorted(set(children_of.keys()) - set(parents_of.keys()))
sys.stderr.write(
    f"category_parent edges = {cp_count}, "
    f"unique children = {len(parents_of)}, "
    f"unique parents = {len(children_of)}, "
    f"true roots (no parents) = {len(candidate_roots)}\n"
)


def bfs_size(start, adj, max_depth=10):
    visited = {start}
    queue = collections.deque([(start, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for child in adj.get(node, []):
            if child not in visited:
                visited.add(child)
                queue.append((child, depth + 1))
    return len(visited)


# Scan ALL true roots to find the largest subtree. With 27k candidates
# this is still <1s in practice (most roots have tiny subtrees).
best_root = None
best_size = 0
scanned = 0
top5 = []
for r in candidate_roots:
    s = bfs_size(r, children_of, max_depth=10)
    if s > best_size:
        best_size = s
        best_root = r
    top5.append((s, r))
    scanned += 1
    if scanned % 5000 == 0:
        sys.stderr.write(f"  ...scanned {scanned}/{len(candidate_roots)} roots\n")
top5.sort(reverse=True)
sys.stderr.write(
    f"best root: id={best_root} subtree_size={best_size}  "
    f"(top 5: {[(s, r) for s, r in top5[:5]]})\n"
)
env.close()

# pass 3: emit fixture files
seed_ids_path = FIXTURE / "seed_ids.txt"
with open(seed_ids_path, "w") as f:
    for sid in sorted(parents_of.keys()):
        f.write(f"{sid}\n")

with open(FIXTURE / "root_ids.txt", "w") as f:
    f.write(f"{best_root}\n")

# Synthetic article_category.tsv using the real i2s-decoded names
# (original page_id strings like "1000114").  Critical: these must
# match the strings the Rust LMDB-mode bench inserts into
# runtime_category_parents via lmdb.load_i2s(); otherwise tuple_count
# will be 0 (kernel can't match "node_<id>" seeds against "<page_id>"
# edges).  Read i2s from the SAME env we already have open above.
all_nodes = sorted(set(parents_of.keys()) | set(children_of.keys()))
env_rw = lmdb.open(str(LMDB_DIR), max_dbs=16, map_size=2 * 1024 * 1024 * 1024, subdir=True, readonly=False)
i2s_db_rw = env_rw.open_db(b"i2s")
nid_to_str = {}
with env_rw.begin() as txn:
    cur = txn.cursor(db=i2s_db_rw)
    for k, v in cur:
        nid_to_str[struct.unpack("<i", k)[0]] = v.decode("utf-8", "replace")
env_rw.close()
with open(FIXTURE / "article_category.tsv", "w") as f:
    f.write("article\tcategory\n")
    for nid in all_nodes:
        name = nid_to_str.get(nid, f"unknown_{nid}")
        f.write(f"{name}\t{name}\n")
sys.stderr.write(f"article_category.tsv: {len(all_nodes)} rows, sample name = '{nid_to_str.get(all_nodes[0], '?')}'\n")

# root_categories.tsv: emit one synthetic name matching best_root
with open(FIXTURE / "root_categories.tsv", "w") as f:
    f.write("root_category\n")
    f.write(f"node_{best_root}\n")

# metadata.json
with open(FIXTURE / "metadata.json", "w") as f:
    json.dump(
        {
            "name": "simplewiki_cats",
            "source": "simplewiki-latest-categorylinks.sql.gz",
            "schema_note": "newer MediaWiki schema (cl_to via linktarget); "
                            "nodes are page_id / linktarget_id, not category names. "
                            "Graph topology is valid for perf measurement.",
            "subcat_edges": cp_count,
            "unique_children": len(parents_of),
            "unique_parents": len(children_of),
            "true_roots": len(candidate_roots),
            "best_root_id": best_root,
            "best_root_subtree_size": best_size,
            "article_category_tsv": "synthetic (node_<id> -> node_<id>)",
            "post_ingest_seconds": round(time.time() - t0, 2),
        },
        f,
        indent=2,
    )

sys.stderr.write(f"post-ingest done in {time.time() - t0:.2f}s\n")
