#!/usr/bin/env python3
"""Post-process the integer-keyed enwiki LMDB:

- add category_child reverse-edge DUPSORT sub-db
- synthesize identity s2i + i2s sub-dbs so the Rust LmdbFactSource
  (which opens all four Phase 1 sub-dbs at startup) can attach without
  changes.  Strings are the decimal stringifications of the IDs.
- pick root by graph topology: node with the largest reachable subtree
  among true-roots (LMDB-cursor BFS, no in-memory adjacency)
- emit seed_ids.txt, root_ids.txt, root_categories.tsv,
  article_category.tsv (synthetic id -> id), metadata.json

Memory profile: a single set of unique int32 IDs (~ few million for
enwiki) + a single Counter mapping parent_id -> child_count.  No full
adjacency lists in RAM.
"""
import collections
import json
import struct
import sys
import time
from pathlib import Path

import lmdb

FIXTURE = Path("data/benchmark/enwiki_cats")
LMDB_DIR = FIXTURE / "lmdb_resident"

if not LMDB_DIR.exists():
    sys.stderr.write(f"enwiki_post_ingest: {LMDB_DIR} missing — run the ingest first\n")
    sys.exit(2)

t0 = time.time()
env = lmdb.open(
    str(LMDB_DIR),
    max_dbs=16,
    map_size=64 * 1024 * 1024 * 1024,  # 64 GB headroom, sparse-allocated
    subdir=True,
    readonly=False,
)
# The enwiki ingest declaration in examples/streaming/enwiki_category_ingest.pl
# does NOT set UW_LMDB_DBNAME, so the consumer falls through to sub-db "main"
# (because UW_LMDB_DUPSORT=1 forces a named sub-db).  The Rust LmdbFactSource
# expects "category_parent" (matching the simplewiki ingest), so we re-key
# edges into the canonical name here.  See WAM_LMDB_RESIDENT_INTERNING
# spec §1.4 for the Phase 1 layout.
main_db = env.open_db(b"main", dupsort=True)
cp_db = env.open_db(b"category_parent", dupsort=True, create=True)
cc_db = env.open_db(b"category_child", dupsort=True, create=True)


def i32(b: bytes) -> int:
    return struct.unpack("<i", b)[0]


def le32(i: int) -> bytes:
    return struct.pack("<i", i)


# pass 1: walk "main" (where enwiki ingest wrote edges), copy forward edges
# into the canonical "category_parent" sub-db, write reversed edges into
# "category_child", and accumulate unique-id + degree counters.
cp_count = 0
unique_ids: set = set()
out_degree: collections.Counter = collections.Counter()
in_degree: collections.Counter = collections.Counter()

# Cursor lives on a read txn (stable across many writes); writes go into
# their own write txn that we commit/restart periodically.  LMDB allows
# concurrent read+write txns; the read cursor sees the snapshot at the
# moment it was opened.
BATCH = 500_000
write_txn = env.begin(write=True)
read_txn = env.begin(write=False, buffers=True)
try:
    cur = read_txn.cursor(db=main_db)
    for k, v in cur:
        # Copy to canonical forward edges, write reverse edges.
        # k and v are memoryview slices; convert to bytes for put.
        kb = bytes(k)
        vb = bytes(v)
        write_txn.put(kb, vb, db=cp_db, dupdata=True, overwrite=True)
        write_txn.put(vb, kb, db=cc_db, dupdata=True, overwrite=True)
        cp_count += 1
        child_id = i32(kb)
        parent_id = i32(vb)
        unique_ids.add(child_id)
        unique_ids.add(parent_id)
        out_degree[parent_id] += 1
        in_degree[child_id] += 1
        if cp_count % 1_000_000 == 0:
            sys.stderr.write(
                f"  ...{cp_count:,} edges, |U|={len(unique_ids):,}, "
                f"t={time.time() - t0:.1f}s\n"
            )
        if cp_count % BATCH == 0:
            write_txn.commit()
            write_txn = env.begin(write=True)
    write_txn.commit()
    read_txn.abort()
except Exception:
    write_txn.abort()
    read_txn.abort()
    raise

env.sync()
sys.stderr.write(
    f"pass 1 done: {cp_count:,} edges, "
    f"{len(unique_ids):,} unique ids "
    f"({len(out_degree):,} parents, {len(in_degree):,} children) "
    f"in {time.time() - t0:.1f}s\n"
)

# Roots = nodes that appear as a parent but never as a child.
true_roots = sorted(set(out_degree.keys()) - set(in_degree.keys()))
sys.stderr.write(f"true roots (no parents) = {len(true_roots):,}\n")


# pass 2: pick root by cursor-BFS subtree size.  Avoid the 5,000-root
# brute scan we did at simplewiki — at enwiki scale that is O(M^2) edges
# touched.  Instead, take the top-K candidates by out-degree (cheap
# proxy for "likely hub") and BFS each.
def bfs_size_via_lmdb(start: int, max_depth: int = 10) -> int:
    visited = {start}
    queue = collections.deque([(start, 0)])
    with env.begin() as ro:
        cur = ro.cursor(db=cc_db)
        while queue:
            node, depth = queue.popleft()
            if depth >= max_depth:
                continue
            if not cur.set_key(le32(node)):
                continue
            for vbytes in cur.iternext_dup():
                child = i32(vbytes)
                if child not in visited:
                    visited.add(child)
                    queue.append((child, depth + 1))
    return len(visited)


TOP_K = 32
candidates = [r for r, _ in sorted(out_degree.items(), key=lambda kv: -kv[1])[:TOP_K]]
candidates = [c for c in candidates if c in set(true_roots) or in_degree[c] == 0]
if not candidates:
    candidates = sorted(true_roots, key=lambda r: -out_degree.get(r, 0))[:TOP_K]
sys.stderr.write(
    f"top-{TOP_K} candidate roots (by out-degree): {candidates[:5]}...\n"
)

best_root = None
best_size = 0
results = []
for r in candidates:
    s = bfs_size_via_lmdb(r, max_depth=10)
    results.append((s, r))
    if s > best_size:
        best_size = s
        best_root = r
results.sort(reverse=True)
sys.stderr.write(
    f"best root: id={best_root} subtree_size={best_size:,}  "
    f"(top 5: {[(s, r) for s, r in results[:5]]})\n"
)


# pass 3: write identity s2i + i2s sub-dbs.
sys.stderr.write(f"writing identity s2i + i2s for {len(unique_ids):,} ids...\n")
s2i_db = env.open_db(b"s2i", create=True)
i2s_db = env.open_db(b"i2s", create=True)
written = 0
with env.begin(write=True) as txn:
    for nid in unique_ids:
        nstr = str(nid)
        txn.put(nstr.encode("utf-8"), le32(nid), db=s2i_db, overwrite=True)
        txn.put(le32(nid), nstr.encode("utf-8"), db=i2s_db, overwrite=True)
        written += 1
        if written % 1_000_000 == 0:
            sys.stderr.write(f"  ...{written:,}/{len(unique_ids):,} s2i+i2s pairs\n")
env.sync()
sys.stderr.write(f"identity intern table written\n")

env.close()

# pass 4: emit fixture files (small)
with open(FIXTURE / "seed_ids.txt", "w") as f:
    for sid in sorted(in_degree.keys()):
        f.write(f"{sid}\n")

with open(FIXTURE / "root_ids.txt", "w") as f:
    f.write(f"{best_root}\n")

with open(FIXTURE / "article_category.tsv", "w") as f:
    f.write("article\tcategory\n")
    for nid in sorted(unique_ids):
        s = str(nid)
        f.write(f"{s}\t{s}\n")

with open(FIXTURE / "root_categories.tsv", "w") as f:
    f.write("root_category\n")
    f.write(f"{best_root}\n")

with open(FIXTURE / "metadata.json", "w") as f:
    json.dump(
        {
            "name": "enwiki_cats",
            "source": "enwiki-latest-categorylinks.sql.gz",
            "schema_note": (
                "newer MediaWiki schema (cl_target_id linktarget reference). "
                "Nodes are page_id / linktarget_id; identity intern table is "
                "synthetic for Rust LmdbFactSource compatibility."
            ),
            "subcat_edges": cp_count,
            "unique_ids": len(unique_ids),
            "unique_children": len(in_degree),
            "unique_parents": len(out_degree),
            "true_roots": len(true_roots),
            "best_root_id": best_root,
            "best_root_subtree_size": best_size,
            "best_root_out_degree": out_degree.get(best_root, 0),
            "root_selection": (
                f"top-{TOP_K} candidates by out-degree, "
                f"BFS subtree size at max_depth=10"
            ),
            "post_ingest_seconds": round(time.time() - t0, 2),
        },
        f,
        indent=2,
    )

sys.stderr.write(f"post-ingest done in {time.time() - t0:.1f}s\n")
