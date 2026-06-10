#!/usr/bin/env python3
"""Build a tiny int-native Phase-1 LMDB fixture for cross-target conformance.

The graph is keyed by RAW page ids (like the real Articles fixtures), and the
s2i sub-db is a deliberately NON-IDENTITY dense renumbering -- so a cursor path
that wrongly resolves seeds through s2i (the bug fixed in PR #2772) produces a
WRONG answer, while a correct int-native path (raw ids, no s2i) is right.

Writes <dir>/lmdb_resident/ (+ a `lmdb` symlink), article_category.tsv,
root_ids.txt, root_categories.tsv. Root = 2.

Graph (child -> parent), raw ids:
    10 -> 2          (depth 1 to root)
    11 -> 10         (depth 2)
    12 -> 11         (depth 3)
    20 -> 99         (99 -> 100 -> dead; 20 never reaches root)
    99 -> 100
Seeds (article_category col2): 10, 11, 12, 20.
=> seeds reaching root 2 (any depth, max_depth>=3): {10,11,12} = 3.
"""
import sys, os, struct, lmdb

EDGES = [(10, 2), (11, 10), (12, 11), (20, 99), (99, 100)]   # child -> parent
SEEDS = [10, 11, 12, 20]
ROOT  = 2

def i32(v): return struct.pack("<i", int(v))

def main(outdir):
    res = os.path.join(outdir, "lmdb_resident")
    os.makedirs(res, exist_ok=True)
    for f in ("data.mdb", "lock.mdb"):
        p = os.path.join(res, f)
        if os.path.exists(p): os.remove(p)
    env = lmdb.open(res, max_dbs=8, map_size=8*1024*1024)
    s2i = env.open_db(b"s2i"); i2s = env.open_db(b"i2s")
    cp  = env.open_db(b"category_parent", dupsort=True)
    cc  = env.open_db(b"category_child",  dupsort=True)
    nodes = sorted({n for e in EDGES for n in e} | set(SEEDS) | {ROOT})
    # NON-IDENTITY s2i: dense renumbering offset so s2i[str(n)] != n.
    with env.begin(write=True) as txn:
        for idx, n in enumerate(nodes):
            dense = idx            # 0,1,2,... -- deliberately != n
            txn.put(str(n).encode(), i32(dense), db=s2i)
            txn.put(i32(dense), str(n).encode(), db=i2s)
        for child, parent in EDGES:
            txn.put(i32(child),  i32(parent), db=cp)   # child -> parent
            txn.put(i32(parent), i32(child),  db=cc)   # reverse
    env.close()
    link = os.path.join(outdir, "lmdb")
    if os.path.islink(link) or os.path.exists(link): os.remove(link)
    os.symlink("lmdb_resident", link)
    with open(os.path.join(outdir, "article_category.tsv"), "w") as f:
        f.write("article\tcategory\n")   # header: F# skips non-numeric col2, Rust skips line 1
        for s in SEEDS: f.write(f"art_{s}\t{s}\n")
    with open(os.path.join(outdir, "root_ids.txt"), "w") as f:
        f.write(f"{ROOT}\n")
    with open(os.path.join(outdir, "root_categories.tsv"), "w") as f:
        f.write(f"root\n{ROOT}\n")
    print(f"tiny int-native fixture at {outdir}: {len(nodes)} nodes, {len(EDGES)} edges, {len(SEEDS)} seeds, root={ROOT}, expect 3 reach root")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/uw_tiny_intnative")
